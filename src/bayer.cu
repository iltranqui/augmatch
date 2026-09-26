#include "augmatch/bayer.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {
__device__ unsigned long long cfa_mix(unsigned long long x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
__device__ float cfa_uni(unsigned long long x){return (float)((cfa_mix(x)>>11)*(1.0/9007199254740992.0));}
__device__ float cfa_norm(unsigned long long x){float a=fmaxf(cfa_uni(x),1.0e-7f);return sqrtf(-2.0f*logf(a))*cosf(6.28318530718f*cfa_uni(x^0xd1b54a32d192ed03ULL));}
__device__ unsigned long long cfa_key(unsigned long long s,int x,int y,int c,unsigned long long k){return s^(unsigned long long)x*0x632be59bd9b4e019ULL^(unsigned long long)y*0x8cb92baa2f2f6f7dULL^(unsigned long long)c*0x9e3779b97f4a7c15ULL^k;}
#define norm cfa_norm
#define uni cfa_uni
#define key cfa_key
__device__ int cfa_plane_at_device(int y,int x,CfaPlaneMap map) { return map.plane[((y&1)<<1)|(x&1)]; }
__device__ int cfa_nearest(const float* in,int w,int h,int x,int y,int plane,CfaPlaneMap map) {
  for (int r=0;r<=2;++r) for (int dy=-r;dy<=r;++dy) for (int dx=-r;dx<=r;++dx) {
    if (abs(dx)+abs(dy)!=r) continue;
    const int xx=max(0,min(w-1,x+dx)), yy=max(0,min(h-1,y+dy));
    if (cfa_plane_at_device(yy,xx,map)==plane) return yy*w+xx;
  }
  return y*w+x;
}
__global__ void cfa_response_kernel(const float* in,float* out,CfaChannelResponseVariationConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width,p=cfa_plane_at_device(y,x,c.plane_map);const float f=c.response[p]+c.variation_stddev*norm(key(c.seed,p,0,p,0x43524153504f4e53ULL));out[i]=fminf(c.clip_max,fmaxf(c.clip_min,in[i]*f));
}
__global__ void cfa_leakage_kernel(const float* in,float* out,CfaLeakageConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width,p=cfa_plane_at_device(y,x,c.plane_map);float value=0.0f;for(int q=0;q<4;++q)value+=c.leakage[p*4+q]*in[cfa_nearest(in,c.width,c.height,x,y,q,c.plane_map)];out[i]=fminf(c.clip_max,fmaxf(c.clip_min,value));
}
__global__ void cfa_misregistration_kernel(const float* in,float* out,CfaMisregistrationConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width,p=cfa_plane_at_device(y,x,c.plane_map);const int xx=max(0,min(c.width-1,x+c.dx[p])),yy=max(0,min(c.height-1,y+c.dy[p]));out[i]=fminf(c.clip_max,fmaxf(c.clip_min,in[cfa_nearest(in,c.width,c.height,xx,yy,p,c.plane_map)]));
}
__global__ void cfa_missing_kernel(const float* in,float* out,CfaMissingSamplesConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width,p=cfa_plane_at_device(y,x,c.plane_map);const bool missing=c.missing_mask?(c.missing_mask[i]!=0):(uni(key(c.seed,x,y,0,0x4346414d49535349ULL))<c.missing_probability);float value=in[i];if(missing){if(c.replacement==CfaMissingReplacement::Nearest)value=in[cfa_nearest(in,c.width,c.height,x,y,p,c.plane_map)];else value=c.replacement==CfaMissingReplacement::Fill?c.replacement_value:0.0f;}out[i]=fminf(c.clip_max,fmaxf(c.clip_min,value));
}
__global__ void bayer_plane_noise_kernel(const float* in,float* out,BayerPlaneNoiseConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width,p=cfa_plane_at_device(y,x,c.plane_map);out[i]=fminf(c.clip_max,fmaxf(c.clip_min,in[i]+c.stddev[p]*norm(key(c.seed,x,y,p,0x42504e4f49534531ULL))));
}
__global__ void bayer_plane_gain_kernel(const float* in,float* out,BayerPlaneGainConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width,p=cfa_plane_at_device(y,x,c.plane_map);out[i]=fminf(c.clip_max,fmaxf(c.clip_min,in[i]*c.gain[p]));
}
void validate_plane_config(int w,int h,const float* in,const float* out,const CfaPlaneMap& map,const char* name) {
  if(!in||!out||w<=0||h<=0) throw std::invalid_argument(std::string("invalid ")+name+" configuration"); bool seen[4]={false,false,false,false}; for(int i=0;i<4;++i){if(map.plane[i]>=4||seen[map.plane[i]])throw std::invalid_argument("CFA plane map must be a permutation of 0..3");seen[map.plane[i]]=true;}
}
void validate_clip(float lo,float hi){if(!isfinite(lo)||!isfinite(hi)||hi<lo)throw std::invalid_argument("invalid CFA clipping range");}
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__device__ int bayer_channel_device(int y, int x, int pattern) {
  const bool ye = (y & 1) != 0, xe = (x & 1) != 0;
  if (pattern == 0) return ye ? (xe ? 2 : 1) : (xe ? 1 : 0);
  if (pattern == 1) return ye ? (xe ? 0 : 1) : (xe ? 1 : 2);
  if (pattern == 2) return ye ? (xe ? 1 : 2) : (xe ? 0 : 1);
  return ye ? (xe ? 1 : 0) : (xe ? 2 : 1);
}

__device__ float bayer_raw_at_device(const unsigned char* in,int w,int h,int x,int y) {
  x=max(0,min(w-1,x)); y=max(0,min(h-1,y)); return (float)in[y*w+x];
}
__device__ float bayer_mhc_value(const unsigned char* in,int w,int h,int x,int y,int target,int pattern) {
  const int center=bayer_channel_device(y,x,pattern); if(center==target)return bayer_raw_at_device(in,w,h,x,y);
  int kind=0; bool horizontal=true;
  if(target!=1) { if(center==1) { horizontal=bayer_channel_device(y,max(0,min(w-1,x-1)),pattern)==target||bayer_channel_device(y,max(0,min(w-1,x+1)),pattern)==target; kind=1; } else kind=2; }
  const float green[5][5]={{0,0,-1,0,0},{0,0,2,0,0},{-1,2,4,2,-1},{0,0,2,0,0},{0,0,-1,0,0}};
  const float same[5][5]={{0,0,.5f,0,0},{0,-1,0,-1,0},{-1,4,5,4,-1},{0,-1,0,-1,0},{0,0,.5f,0,0}};
  const float diagonal[5][5]={{0,0,-1.5f,0,0},{0,2,0,2,0},{-1.5f,0,6,0,-1.5f},{0,2,0,2,0},{0,0,-1.5f,0,0}};
  float sum=0.0f;
  for(int ky=-2;ky<=2;++ky)for(int kx=-2;kx<=2;++kx){const float coefficient=kind==0?green[ky+2][kx+2]:(kind==2?diagonal[ky+2][kx+2]:(horizontal?same[ky+2][kx+2]:same[kx+2][ky+2]));sum+=coefficient*bayer_raw_at_device(in,w,h,x+kx,y+ky);}
  return sum/8.0f;
}
__device__ float bayer_edge_directional(const unsigned char* in,int w,int h,int x,int y,int target,int pattern,int dx,int dy) {
  float sum=0.0f; int count=0;
  for(int sign=-1;sign<=1;sign+=2) for(int distance=1;distance<=2;++distance){const int xx=max(0,min(w-1,x+sign*distance*dx)),yy=max(0,min(h-1,y+sign*distance*dy));if(bayer_channel_device(yy,xx,pattern)==target){sum+=bayer_raw_at_device(in,w,h,xx,yy);++count;break;}}
  return count?sum/count:bayer_raw_at_device(in,w,h,x,y);
}
__device__ float bayer_edge_value(const unsigned char* in,int w,int h,int x,int y,int target,int pattern) {
  const int center=bayer_channel_device(y,x,pattern);if(center==target)return bayer_raw_at_device(in,w,h,x,y);
  if(center==1){const float gh=fabsf(bayer_raw_at_device(in,w,h,x-1,y)-bayer_raw_at_device(in,w,h,x+1,y));const float gv=fabsf(bayer_raw_at_device(in,w,h,x,y-1)-bayer_raw_at_device(in,w,h,x,y+1));const float horizontal=bayer_edge_directional(in,w,h,x,y,target,pattern,1,0),vertical=bayer_edge_directional(in,w,h,x,y,target,pattern,0,1);if(gh<gv)return horizontal;if(gh>gv)return vertical;return (horizontal+vertical)*.5f;}
  const float gd1=fabsf(bayer_raw_at_device(in,w,h,x-1,y-1)-bayer_raw_at_device(in,w,h,x+1,y+1));const float gd2=fabsf(bayer_raw_at_device(in,w,h,x-1,y+1)-bayer_raw_at_device(in,w,h,x+1,y-1));const float d1=bayer_edge_directional(in,w,h,x,y,target,pattern,1,1),d2=bayer_edge_directional(in,w,h,x,y,target,pattern,1,-1);if(gd1<gd2)return d1;if(gd1>gd2)return d2;return (d1+d2)*.5f;
}
__device__ unsigned char bayer_clip_round(float value){return (unsigned char)floorf(fminf(255.0f,fmaxf(0.0f,value))+0.5f);}

__global__ void demosaic_malvar_kernel(const unsigned char* in,unsigned char* out,BayerConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*3;if(i>=n)return;const int ch=i%3,pix=i/3,x=pix%c.width,y=pix/c.width;out[i]=bayer_clip_round(bayer_mhc_value(in,c.width,c.height,x,y,ch,(int)c.pattern));}
__global__ void demosaic_edge_kernel(const unsigned char* in,unsigned char* out,BayerConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*3;if(i>=n)return;const int ch=i%3,pix=i/3,x=pix%c.width,y=pix/c.width;out[i]=bayer_clip_round(bayer_edge_value(in,c.width,c.height,x,y,ch,(int)c.pattern));}

__device__ int rgbw_channel_device(int y, int x, int pattern) {
  const bool ye = (y & 1) != 0, xe = (x & 1) != 0;
  if (pattern == 0) return ye ? (xe ? 2 : 3) : (xe ? 1 : 0);
  if (pattern == 1) return ye ? (xe ? 1 : 2) : (xe ? 3 : 0);
  if (pattern == 2) return ye ? (xe ? 0 : 3) : (xe ? 2 : 1);
  return ye ? (xe ? 2 : 0) : (xe ? 1 : 3);
}

__global__ void bayer_kernel(const unsigned char* in, unsigned char* out, BayerConfig c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = c.width * c.height;
  if (i >= n) return;
  const int x = i % c.width, y = i / c.width;
  out[i] = in[i * 3 + bayer_channel_device(y, x, static_cast<int>(c.pattern))];
}

__global__ void quad_bayer_kernel(const unsigned char* in, unsigned char* out, QuadBayerConfig c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = c.width * c.height;
  if (i >= n) return;
  const int x = i % c.width, y = i / c.width;
  const int channel = bayer_channel_device(y >> 1, x >> 1, static_cast<int>(c.pattern));
  out[i] = in[i * 3 + channel];
}

__global__ void rgbw_kernel(const unsigned char* in, unsigned char* out, RGBWConfig c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = c.width * c.height;
  if (i >= n) return;
  const int x = i % c.width, y = i / c.width;
  out[i] = in[i * 4 + rgbw_channel_device(y, x, static_cast<int>(c.pattern))];
}

__global__ void custom_cfa_kernel(const unsigned char* in, unsigned char* out, CustomCfaConfig c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = c.width * c.height;
  if (i >= n) return;
  const int x = i % c.width, y = i / c.width;
  const int channel = c.mask[(y % c.mask_height) * c.mask_width + (x % c.mask_width)];
  out[i] = in[i * c.channels + channel];
}

__device__ int plane_at(int y, int x, BayerPattern p) {
  return bayer_channel_device(y, x, static_cast<int>(p));
}

__global__ void demosaic_kernel(const unsigned char* in, unsigned char* out, BayerConfig c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = c.width * c.height * 3;
  if (i >= n) return;
  const int ch = i % 3, pix = i / 3, x = pix % c.width, y = pix / c.width;
  int sx = x, sy = y;
  for (int r = 0; r <= 2; ++r) {
    bool found = false;
    for (int dy = -r; dy <= r && !found; ++dy)
      for (int dx = -r; dx <= r; ++dx)
        if (abs(dx) + abs(dy) == r) {
          const int xx = max(0, min(c.width - 1, x + dx));
          const int yy = max(0, min(c.height - 1, y + dy));
          if (plane_at(yy, xx, c.pattern) == ch) {
            sx = xx; sy = yy; found = true; break;
          }
        }
    if (found) break;
  }
  out[i] = in[sy * c.width + sx];
}

__global__ void demosaic_bilinear_kernel(const unsigned char* in, unsigned char* out, BayerConfig c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = c.width * c.height * 3;
  if (i >= n) return;
  const int ch = i % 3, pix = i / 3, x = pix % c.width, y = pix / c.width;
  int sum = 0, count = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const int xx = max(0, min(c.width - 1, x + dx));
      const int yy = max(0, min(c.height - 1, y + dy));
      if (plane_at(yy, xx, c.pattern) == ch) { sum += in[yy * c.width + xx]; ++count; }
    }
  out[i] = static_cast<unsigned char>((sum + count / 2) / count);
}

void validate(const std::uint8_t* in, const std::uint8_t* out, int width, int height,
              const char* name) {
  if (!in || !out || width <= 0 || height <= 0)
    throw std::invalid_argument(std::string("invalid ") + name + " configuration");
}
}

CfaPlaneMap bayer_plane_map(BayerPattern pattern) {
  switch (pattern) {
    case BayerPattern::RGGB: return {{0,1,2,3}};
    case BayerPattern::BGGR: return {{3,2,1,0}};
    case BayerPattern::GRBG: return {{1,0,3,2}};
    case BayerPattern::GBRG: return {{2,3,0,1}};
  }
  throw std::invalid_argument("invalid Bayer pattern");
}
int bayer_plane_at(int y,int x,const CfaPlaneMap& map) {
  if(x<0||y<0) throw std::invalid_argument("negative CFA coordinate");
  return map.plane[((y&1)<<1)|(x&1)];
}

void bayer_sample_u8(const std::uint8_t* in, std::uint8_t* out, const BayerConfig& c, cudaStream_t s) {
  validate(in, out, c.width, c.height, "Bayer");
  const int n = c.width * c.height;
  bayer_kernel<<<(n + 255) / 256, 256, 0, s>>>(in, out, c); check(cudaGetLastError());
}

void quad_bayer_sample_u8(const std::uint8_t* in, std::uint8_t* out, const QuadBayerConfig& c, cudaStream_t s) {
  validate(in, out, c.width, c.height, "Quad-Bayer");
  const int n = c.width * c.height;
  quad_bayer_kernel<<<(n + 255) / 256, 256, 0, s>>>(in, out, c); check(cudaGetLastError());
}

void rgbw_sample_u8(const std::uint8_t* in, std::uint8_t* out, const RGBWConfig& c, cudaStream_t s) {
  validate(in, out, c.width, c.height, "RGBW");
  const int n = c.width * c.height;
  rgbw_kernel<<<(n + 255) / 256, 256, 0, s>>>(in, out, c); check(cudaGetLastError());
}

void custom_cfa_sample_u8(const std::uint8_t* in, std::uint8_t* out, const CustomCfaConfig& c, cudaStream_t s) {
  validate(in, out, c.width, c.height, "custom CFA");
  if (c.channels <= 0 || c.mask_width <= 0 || c.mask_height <= 0 || !c.mask)
    throw std::invalid_argument("invalid custom CFA mask configuration");
  const int n = c.width * c.height;
  custom_cfa_kernel<<<(n + 255) / 256, 256, 0, s>>>(in, out, c); check(cudaGetLastError());
}

void bayer_demosaic_nearest_u8(const std::uint8_t* in, std::uint8_t* out, const BayerConfig& c, cudaStream_t s) {
  validate(in, out, c.width, c.height, "Bayer");
  const int n = c.width * c.height * 3;
  demosaic_kernel<<<(n + 255) / 256, 256, 0, s>>>(in, out, c); check(cudaGetLastError());
}

void bayer_demosaic_bilinear_u8(const std::uint8_t* in, std::uint8_t* out, const BayerConfig& c, cudaStream_t s) {
  validate(in, out, c.width, c.height, "Bayer");
  const int n = c.width * c.height * 3;
  demosaic_bilinear_kernel<<<(n + 255) / 256, 256, 0, s>>>(in, out, c); check(cudaGetLastError());
}
void bayer_demosaic_malvar_he_cutler_u8(const std::uint8_t* in, std::uint8_t* out, const BayerConfig& c, cudaStream_t s) {
  validate(in,out,c.width,c.height,"Malvar-He-Cutler Bayer"); const int n=c.width*c.height*3; demosaic_malvar_kernel<<<(n+255)/256,256,0,s>>>(in,out,c); check(cudaGetLastError());
}
void bayer_demosaic_edge_aware_u8(const std::uint8_t* in, std::uint8_t* out, const BayerConfig& c, cudaStream_t s) {
  validate(in,out,c.width,c.height,"edge-aware Bayer"); const int n=c.width*c.height*3; demosaic_edge_kernel<<<(n+255)/256,256,0,s>>>(in,out,c); check(cudaGetLastError());
}
void bayer_demosaic_malvar_he_cutler_rggb_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,cudaStream_t s){BayerConfig c{w,h,BayerPattern::RGGB};bayer_demosaic_malvar_he_cutler_u8(in,out,c,s);}
void bayer_demosaic_malvar_he_cutler_bggr_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,cudaStream_t s){BayerConfig c{w,h,BayerPattern::BGGR};bayer_demosaic_malvar_he_cutler_u8(in,out,c,s);}
void bayer_demosaic_malvar_he_cutler_grbg_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,cudaStream_t s){BayerConfig c{w,h,BayerPattern::GRBG};bayer_demosaic_malvar_he_cutler_u8(in,out,c,s);}
void bayer_demosaic_malvar_he_cutler_gbrg_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,cudaStream_t s){BayerConfig c{w,h,BayerPattern::GBRG};bayer_demosaic_malvar_he_cutler_u8(in,out,c,s);}
void bayer_demosaic_edge_aware_rggb_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,cudaStream_t s){BayerConfig c{w,h,BayerPattern::RGGB};bayer_demosaic_edge_aware_u8(in,out,c,s);}
void bayer_demosaic_edge_aware_bggr_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,cudaStream_t s){BayerConfig c{w,h,BayerPattern::BGGR};bayer_demosaic_edge_aware_u8(in,out,c,s);}
void bayer_demosaic_edge_aware_grbg_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,cudaStream_t s){BayerConfig c{w,h,BayerPattern::GRBG};bayer_demosaic_edge_aware_u8(in,out,c,s);}
void bayer_demosaic_edge_aware_gbrg_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,cudaStream_t s){BayerConfig c{w,h,BayerPattern::GBRG};bayer_demosaic_edge_aware_u8(in,out,c,s);}

void cfa_channel_response_variation_f32(const float* in,float* out,const CfaChannelResponseVariationConfig& c,cudaStream_t s){validate_plane_config(c.width,c.height,in,out,c.plane_map,"CFA response");validate_clip(c.clip_min,c.clip_max);if(!isfinite(c.variation_stddev)||c.variation_stddev<0)throw std::invalid_argument("invalid CFA response variation");for(int p=0;p<4;++p)if(!isfinite(c.response[p]))throw std::invalid_argument("non-finite CFA response");const int n=c.width*c.height;cfa_response_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void cfa_leakage_f32(const float* in,float* out,const CfaLeakageConfig& c,cudaStream_t s){validate_plane_config(c.width,c.height,in,out,c.plane_map,"CFA leakage");validate_clip(c.clip_min,c.clip_max);for(int i=0;i<16;++i)if(!isfinite(c.leakage[i])||c.leakage[i]<0)throw std::invalid_argument("invalid CFA leakage");const int n=c.width*c.height;cfa_leakage_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void cfa_misregistration_f32(const float* in,float* out,const CfaMisregistrationConfig& c,cudaStream_t s){validate_plane_config(c.width,c.height,in,out,c.plane_map,"CFA misregistration");validate_clip(c.clip_min,c.clip_max);for(int p=0;p<4;++p)if(abs(c.dx[p])>c.width||abs(c.dy[p])>c.height)throw std::invalid_argument("CFA misregistration offset is too large");const int n=c.width*c.height;cfa_misregistration_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void cfa_missing_samples_f32(const float* in,float* out,const CfaMissingSamplesConfig& c,cudaStream_t s){validate_plane_config(c.width,c.height,in,out,c.plane_map,"CFA missing samples");validate_clip(c.clip_min,c.clip_max);if(!isfinite(c.missing_probability)||c.missing_probability<0||c.missing_probability>1||!isfinite(c.replacement_value)||(c.replacement!=CfaMissingReplacement::Zero&&c.replacement!=CfaMissingReplacement::Fill&&c.replacement!=CfaMissingReplacement::Nearest))throw std::invalid_argument("invalid CFA missing samples");const int n=c.width*c.height;cfa_missing_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void bayer_plane_noise_f32(const float* in,float* out,const BayerPlaneNoiseConfig& c,cudaStream_t s){validate_plane_config(c.width,c.height,in,out,c.plane_map,"Bayer plane noise");validate_clip(c.clip_min,c.clip_max);for(int p=0;p<4;++p)if(!isfinite(c.stddev[p])||c.stddev[p]<0)throw std::invalid_argument("invalid Bayer plane noise");const int n=c.width*c.height;bayer_plane_noise_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void bayer_plane_gain_f32(const float* in,float* out,const BayerPlaneGainConfig& c,cudaStream_t s){validate_plane_config(c.width,c.height,in,out,c.plane_map,"Bayer plane gain");validate_clip(c.clip_min,c.clip_max);for(int p=0;p<4;++p)if(!isfinite(c.gain[p])||c.gain[p]<0)throw std::invalid_argument("invalid Bayer plane gain");const int n=c.width*c.height;bayer_plane_gain_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}

__device__ unsigned char artifact_clip_u8(float v,int lo,int hi){return (unsigned char)floorf(fminf((float)hi,fmaxf((float)lo,v))+.5f);}
__device__ float directional_delta_device(const unsigned char* in,int w,int h,int x,int y,int target,int pattern,int direction){
  if(bayer_channel_device(y,x,pattern)==target||bayer_channel_device(y,x,pattern)!=1)return 0.0f;
  const float gh=fabsf(bayer_raw_at_device(in,w,h,x-1,y)-bayer_raw_at_device(in,w,h,x+1,y)),gv=fabsf(bayer_raw_at_device(in,w,h,x,y-1)-bayer_raw_at_device(in,w,h,x,y+1));
  const float horizontal=bayer_edge_directional(in,w,h,x,y,target,pattern,1,0),vertical=bayer_edge_directional(in,w,h,x,y,target,pattern,0,1);
  const float selected=direction==1?horizontal:(direction==2?vertical:(gh<gv?horizontal:(gh>gv?vertical:(horizontal+vertical)*.5f)));
  const float other=direction==1?vertical:(direction==2?horizontal:(gh<gv?vertical:(gh>gv?horizontal:(horizontal+vertical)*.5f))); return selected-other;
}
__global__ void directional_artifact_kernel(const unsigned char* in,unsigned char* out,DirectionalDemosaicingArtifactsConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*3;if(i>=n)return;const int ch=i%3,p=i/3,x=p%c.width,y=p/c.width;out[i]=artifact_clip_u8(bayer_edge_value(in,c.width,c.height,x,y,ch,(int)c.pattern)+c.strength*directional_delta_device(in,c.width,c.height,x,y,ch,(int)c.pattern,c.direction),c.clip_min,c.clip_max);}
__global__ void zipper_artifact_kernel(const unsigned char* in,unsigned char* out,FalseColorZipperArtifactsConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*3;if(i>=n)return;const int ch=i%3,p=i/3,x=p%c.width,y=p/c.width;const float edge=fabsf(bayer_raw_at_device(in,c.width,c.height,x-1,y)-bayer_raw_at_device(in,c.width,c.height,x+1,y));const float d=c.strength*edge*(((x+y)&1)?-1.0f:1.0f);const float sign=ch==0?d:(ch==2?-d:0.0f);const float add=bayer_channel_device(y,x,(int)c.pattern)==ch?0.0f:sign;out[i]=artifact_clip_u8(bayer_edge_value(in,c.width,c.height,x,y,ch,(int)c.pattern)+add,c.clip_min,c.clip_max);}
__global__ void aliasing_artifact_kernel(const unsigned char* in,unsigned char* out,DemosaicingAliasingConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*3;if(i>=n)return;const int ch=i%3,p=i/3,x=p%c.width,y=p/c.width;const float wave=sinf(6.28318530718f*((float)(x+y)/(float)c.period)+c.phase)*c.strength*127.5f;const float sign=ch==0?wave:(ch==2?-wave:0.0f);const float add=bayer_channel_device(y,x,(int)c.pattern)==ch?0.0f:sign;out[i]=artifact_clip_u8(bayer_edge_value(in,c.width,c.height,x,y,ch,(int)c.pattern)+add,c.clip_min,c.clip_max);}
__global__ void ringing_artifact_kernel(const unsigned char* in,unsigned char* out,DemosaicingRingingConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*3;if(i>=n)return;const int ch=i%3,p=i/3,x=p%c.width,y=p/c.width;const float center=bayer_raw_at_device(in,c.width,c.height,x,y),lap=center-(bayer_raw_at_device(in,c.width,c.height,x-1,y)+bayer_raw_at_device(in,c.width,c.height,x+1,y)+bayer_raw_at_device(in,c.width,c.height,x,y-1)+bayer_raw_at_device(in,c.width,c.height,x,y+1))*.25f;const float add=bayer_channel_device(y,x,(int)c.pattern)==ch?0.0f:c.strength*lap;out[i]=artifact_clip_u8(bayer_edge_value(in,c.width,c.height,x,y,ch,(int)c.pattern)+add,c.clip_min,c.clip_max);}
__global__ void noise_amplification_kernel(const unsigned char* in,unsigned char* out,DemosaicingNoiseAmplificationConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*3;if(i>=n)return;const int ch=i%3,p=i/3,x=p%c.width,y=p/c.width;const float gradient=fabsf(bayer_raw_at_device(in,c.width,c.height,x-1,y)-bayer_raw_at_device(in,c.width,c.height,x+1,y))+fabsf(bayer_raw_at_device(in,c.width,c.height,x,y-1)-bayer_raw_at_device(in,c.width,c.height,x,y+1));const float gain=1.0f+c.amplification*gradient/510.0f;const float noise=c.noise_stddev*gain*norm(key(c.seed,x,y,ch,0x444e4f4953454152ULL));out[i]=artifact_clip_u8(bayer_edge_value(in,c.width,c.height,x,y,ch,(int)c.pattern)+noise,c.clip_min,c.clip_max);}
void validate_artifact_u8(const std::uint8_t* in,const std::uint8_t* out,int w,int h,int lo,int hi,const char* name){if(!in||!out||w<=0||h<=0||lo<0||hi>255||hi<lo)throw std::invalid_argument(std::string("invalid ")+name+" configuration");}
void bayer_directional_demosaicing_artifacts_u8(const std::uint8_t* in,std::uint8_t* out,const DirectionalDemosaicingArtifactsConfig& c,cudaStream_t s){validate_artifact_u8(in,out,c.width,c.height,c.clip_min,c.clip_max,"directional demosaicing artifacts");if(c.direction<0||c.direction>2||!isfinite(c.strength)||c.strength<0)throw std::invalid_argument("invalid directional demosaicing artifact parameters");const int n=c.width*c.height*3;directional_artifact_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void bayer_false_color_zipper_artifacts_u8(const std::uint8_t* in,std::uint8_t* out,const FalseColorZipperArtifactsConfig& c,cudaStream_t s){validate_artifact_u8(in,out,c.width,c.height,c.clip_min,c.clip_max,"false-color zipper artifacts");if(!isfinite(c.strength)||c.strength<0)throw std::invalid_argument("invalid false-color zipper strength");const int n=c.width*c.height*3;zipper_artifact_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void bayer_demosaicing_aliasing_u8(const std::uint8_t* in,std::uint8_t* out,const DemosaicingAliasingConfig& c,cudaStream_t s){validate_artifact_u8(in,out,c.width,c.height,c.clip_min,c.clip_max,"demosaicing aliasing");if(c.period<=0||!isfinite(c.phase)||!isfinite(c.strength)||c.strength<0)throw std::invalid_argument("invalid demosaicing aliasing parameters");const int n=c.width*c.height*3;aliasing_artifact_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void bayer_demosaicing_ringing_u8(const std::uint8_t* in,std::uint8_t* out,const DemosaicingRingingConfig& c,cudaStream_t s){validate_artifact_u8(in,out,c.width,c.height,c.clip_min,c.clip_max,"demosaicing ringing");if(!isfinite(c.strength)||c.strength<0)throw std::invalid_argument("invalid demosaicing ringing strength");const int n=c.width*c.height*3;ringing_artifact_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
void bayer_demosaicing_noise_amplification_u8(const std::uint8_t* in,std::uint8_t* out,const DemosaicingNoiseAmplificationConfig& c,cudaStream_t s){validate_artifact_u8(in,out,c.width,c.height,c.clip_min,c.clip_max,"demosaicing noise amplification");if(!isfinite(c.noise_stddev)||c.noise_stddev<0||!isfinite(c.amplification)||c.amplification<0)throw std::invalid_argument("invalid demosaicing noise amplification parameters");const int n=c.width*c.height*3;noise_amplification_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError());}
}
