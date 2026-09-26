#include "video_codec_impl.hpp"
#include <cuda_runtime.h>
#include <limits>
namespace augmatch { namespace {
void ck(cudaError_t e,const char* m){if(e!=cudaSuccess)throw std::runtime_error(std::string(m)+": "+cudaGetErrorString(e));}
std::vector<std::uint8_t> host_input(const std::uint8_t* p,int w,int h,int c,cudaStream_t s){auto n=video_detail::bytes(w,h,c);if(!p)throw std::invalid_argument("null CUDA codec input");std::vector<std::uint8_t> v(n);ck(cudaMemcpyAsync(v.data(),p,n,cudaMemcpyDeviceToHost,s),"copy codec input");ck(cudaStreamSynchronize(s),"synchronize codec input");return v;}
void host_output(const std::vector<std::uint8_t>& v,std::uint8_t* p,cudaStream_t s){if(!p)throw std::invalid_argument("null CUDA codec output");ck(cudaMemcpyAsync(p,v.data(),v.size(),cudaMemcpyHostToDevice,s),"copy codec output");ck(cudaStreamSynchronize(s),"synchronize codec output");}
}
std::vector<std::uint8_t> video_codec_encode_u8(const std::uint8_t* p,const VideoCodecConfig& c,cudaStream_t s){auto h=host_input(p,c.width,c.height,c.channels,s);return video_detail::encode(h.data(),c);}
void video_codec_decode_u8(const std::uint8_t* p,std::size_t n,std::uint8_t* o,const VideoDecodeConfig& c,cudaStream_t s){std::vector<std::uint8_t> h(video_detail::bytes(c.width,c.height,c.channels));video_detail::decode(p,n,h.data(),c);host_output(h,o,s);}
void video_codec_roundtrip_u8(const std::uint8_t* p,std::uint8_t* o,const VideoCodecConfig& c,cudaStream_t s){auto h=host_input(p,c.width,c.height,c.channels,s);auto b=video_detail::encode(h.data(),c);std::vector<std::uint8_t> d(h.size());video_detail::decode(b.data(),b.size(),d.data(),{c.width,c.height,c.channels,c.codec});host_output(d,o,s);}
void av1_intra_frame_artifacts_u8(const std::uint8_t* p,std::uint8_t* o,const VideoCodecConfig& c,cudaStream_t s){if(c.codec!=VideoCodec::AV1)throw std::invalid_argument("AV1 artifact API requires AV1 codec");video_codec_roundtrip_u8(p,o,c,s);}
}
