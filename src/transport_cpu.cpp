#include "augmatch/transport.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace augmatch { namespace {
void valid_bytes(const std::uint8_t* in, std::uint8_t* out, std::size_t n) { if (!in || !out) throw std::invalid_argument("transport buffers must not be null"); if (!n) throw std::invalid_argument("transport byte count must be positive"); }
void valid_frame(const std::uint8_t* in, std::uint8_t* out, const TransportFrameConfig& c) { if (!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.block_size<=0||c.quant_step<=0||!std::isfinite(c.strength)||c.strength<0||c.strength>1||!std::isfinite(c.probability)||c.probability<0||c.probability>1) throw std::invalid_argument("invalid transport frame configuration"); }
std::uint64_t mix(std::uint64_t x) { x += 0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
float unit(std::uint64_t x) { return static_cast<float>((mix(x)>>11)*(1.0/9007199254740992.0)); }
std::uint64_t key(std::uint64_t seed, std::size_t i) { return seed ^ (i*0x632be59bd9b4e019ULL); }
int clampi(int v,int n) { return std::max(0,std::min(n-1,v)); }
std::size_t index(const TransportFrameConfig& c,int x,int y,int ch) { return (static_cast<std::size_t>(y)*c.width+x)*c.channels+ch; }
}}
namespace augmatch {
void truncated_frame_u8(const std::uint8_t* in,std::uint8_t* out,const TransportByteConfig& c,cudaStream_t) {
  valid_bytes(in,out,c.bytes); if(c.preserve_bytes>c.bytes) throw std::invalid_argument("preserve_bytes exceeds frame size");
  for(std::size_t i=0;i<c.bytes;++i) out[i]=i<c.preserve_bytes?in[i]:c.fill;
}
void bit_flips_u8(const std::uint8_t* in,std::uint8_t* out,const TransportByteConfig& c,cudaStream_t) {
  valid_bytes(in,out,c.bytes); if(!std::isfinite(c.probability)||c.probability<0||c.probability>1) throw std::invalid_argument("bit-flip probability must be in [0,1]");
  for(std::size_t i=0;i<c.bytes;++i) { std::uint8_t v=in[i]; if(unit(key(c.seed,i))<c.probability) { std::uint32_t mask=c.bit_mask; if(!mask) mask=1u << (mix(key(c.seed,i)^0xa5a5a5a5ULL)%8); v=static_cast<std::uint8_t>(v ^ static_cast<std::uint8_t>(mask)); } out[i]=v; }
}
void block_quantization_u8(const std::uint8_t* in,std::uint8_t* out,const TransportFrameConfig& c,cudaStream_t) {
  valid_frame(in,out,c); const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;
  for(std::size_t i=0;i<n;++i) { const int q=static_cast<int>((in[i]+c.quant_step/2)/c.quant_step)*c.quant_step; out[i]=static_cast<std::uint8_t>(std::min(255,q)); }
}
void deblocking_filter_mismatch_u8(const std::uint8_t* in,std::uint8_t* out,const TransportFrameConfig& c,cudaStream_t) {
  valid_frame(in,out,c);
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) for(int ch=0;ch<c.channels;++ch) {
    const std::size_t i=index(c,x,y,ch); float value=in[i];
    if(x%c.block_size==0 && x>0) value=(1.0f-c.strength)*value+c.strength*0.5f*(in[index(c,x-1,y,ch)]+in[i]);
    if(y%c.block_size==0 && y>0) value=(1.0f-c.strength)*value+c.strength*0.5f*(in[index(c,x,y-1,ch)]+in[i]);
    out[i]=static_cast<std::uint8_t>(std::floor(std::max(0.0f,std::min(255.0f,value))+0.5f));
  }
}
void packet_loss_macroblocks_u8(const std::uint8_t* in,std::uint8_t* out,const TransportFrameConfig& c,cudaStream_t) {
  valid_frame(in,out,c); const int bx=(c.width+c.block_size-1)/c.block_size, by=(c.height+c.block_size-1)/c.block_size;
  if(c.block_mask==nullptr && c.probability==0) { std::copy(in,in+static_cast<std::size_t>(c.width)*c.height*c.channels,out); return; }
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) { const int b=(y/c.block_size)*bx+x/c.block_size; const bool lost=c.block_mask?c.block_mask[b]!=0:unit(key(c.seed,static_cast<std::size_t>(b)))<c.probability; for(int ch=0;ch<c.channels;++ch) out[index(c,x,y,ch)]=lost?c.fill:in[index(c,x,y,ch)]; }
  (void)by;
}
void chroma_plane_misalignment_u8(const std::uint8_t* in,std::uint8_t* out,const TransportPlaneConfig& c,cudaStream_t) {
  if(!in||!out||c.width<=0||c.height<=0||c.stride<c.width) throw std::invalid_argument("invalid transport plane configuration");
  std::copy(in,in+static_cast<std::size_t>(c.height)*c.stride,out);
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) { const int sx=clampi(x-c.dx,c.width), sy=clampi(y-c.dy,c.height); out[static_cast<std::size_t>(y)*c.stride+x]=in[static_cast<std::size_t>(sy)*c.stride+sx]; }
}
void raw_bayer_transport_corruption_u16(const std::uint16_t* in,std::uint16_t* out,const RawBayerTransportCorruptionConfig& c,cudaStream_t) {
  if(!in||!out||c.width<=0||c.height<=0||c.stride<c.width||c.bit_depth<=0||c.bit_depth>16||!std::isfinite(c.probability)||c.probability<0||c.probability>1) throw std::invalid_argument("invalid raw Bayer transport configuration");
  const std::uint16_t maxv=c.bit_depth==16?65535u:static_cast<std::uint16_t>((1u<<c.bit_depth)-1u);
  std::copy(in,in+static_cast<std::size_t>(c.height)*c.stride,out);
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) { const std::size_t i=static_cast<std::size_t>(y)*c.stride+x; const bool corrupt=(c.mask&&c.mask[static_cast<std::size_t>(y)*c.width+x])||unit(key(c.seed,i))<c.probability; if(corrupt) out[i]=c.xor_mask?static_cast<std::uint16_t>(in[i]^c.xor_mask):c.fill; else out[i]=std::min(maxv,in[i]); }
}
void raw_bayer_transport_corruption_u8(const std::uint8_t* in,std::uint8_t* out,const TransportByteConfig& c,cudaStream_t s) { bit_flips_u8(in,out,c,s); }
}
