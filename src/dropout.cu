#include "augmatch/geometry/dropout.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__host__ __device__ unsigned long long mix(unsigned long long x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
__host__ __device__ float uni(unsigned long long x){return (float)((mix(x)>>11)*(1.0/9007199254740992.0));}
__global__ void pixel_kernel(const unsigned char* in,unsigned char* out,PixelDropoutConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=uni(c.seed+(unsigned long long)i*0x9e3779b97f4a7c15ULL)<c.probability?c.replacement:in[i];}
__global__ void channel_kernel(const unsigned char* in,unsigned char* out,ChannelDropoutConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n){int ch=i%c.channels;out[i]=uni(c.seed+(unsigned long long)ch*0x9e3779b97f4a7c15ULL)<c.probability?c.replacement:in[i];}}
__global__ void grid_kernel(const unsigned char* in,unsigned char* out,GridDropoutConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n){int x=(i/c.channels)%c.width,y=i/(c.width*c.channels);bool drop=((float)(x%c.cell_width)/c.cell_width<c.ratio)&&((float)(y%c.cell_height)/c.cell_height<c.ratio);out[i]=drop?c.fill:in[i];}}
__global__ void coarse_kernel(const unsigned char* in,unsigned char* out,CoarseDropoutConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int x=(i/c.channels)%c.width,y=i/(c.width*c.channels);bool drop=false;for(int h=0;h<c.holes;++h){int sx=(int)(uni(c.seed+2*h)*c.width),sy=(int)(uni(c.seed+2*h+1)*c.height);drop=drop||(x>=sx&&x<sx+c.hole_width&&y>=sy&&y<sy+c.hole_height);}out[i]=drop?c.fill:in[i];}
__global__ void mask_kernel(const unsigned char* in,const unsigned char* mask,unsigned char* out,MaskDropoutConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;const int pixel=i/c.channels;out[i]=mask[pixel]!=0?c.fill:in[i];}
__global__ void xy_masking_kernel(const unsigned char* in,unsigned char* out,XYMaskingConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;
  if(i>=n)return;
  const int pixel=i/c.channels,x=pixel%c.width,y=pixel/c.width;
  bool drop=false;
  for(int k=0;k<c.row_interval_count;++k){const XYMaskInterval interval=c.row_intervals[k];drop=drop||(y>=interval.begin&&y<interval.end);}
  for(int k=0;k<c.column_interval_count;++k){const XYMaskInterval interval=c.column_intervals[k];drop=drop||(x>=interval.begin&&x<interval.end);}
  out[i]=drop?c.fill:in[i];
}
__global__ void cutout_kernel(const unsigned char* in,unsigned char* out,CutoutConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;
  if(i>=n)return;
  const int pixel=i/c.channels,x=pixel%c.width,y=pixel/c.width;
  bool drop=false;
  for(int k=0;k<c.rectangle_count;++k){const CutoutRectangle r=c.rectangles[k];drop=drop||(x>=r.x0&&x<r.x1&&y>=r.y0&&y<r.y1);}
  out[i]=drop?c.fill:in[i];
}
__global__ void total_dropout_kernel(const unsigned char* in,unsigned char* out,TotalDropoutConfig c,bool random_drop){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;
  if(i<n){const bool drop=c.mask ? (c.mask[0]!=0) : random_drop;out[i]=drop?c.fill:in[i];}
}
void check(cudaError_t e,const char* p){if(e!=cudaSuccess)throw std::runtime_error(std::string(p)+": "+cudaGetErrorString(e));}
void valid(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid dropout dimensions");}
}
void pixel_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const PixelDropoutConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.probability<0||c.probability>1)throw std::invalid_argument("invalid pixel dropout configuration");int n=c.width*c.height*c.channels;pixel_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"pixel_dropout_kernel");}
void channel_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const ChannelDropoutConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.probability<0||c.probability>1)throw std::invalid_argument("invalid channel dropout configuration");int n=c.width*c.height*c.channels;channel_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"channel_dropout_kernel");}
void grid_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const GridDropoutConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.cell_width<=0||c.cell_height<=0||c.ratio<0||c.ratio>1)throw std::invalid_argument("invalid grid dropout configuration");int n=c.width*c.height*c.channels;grid_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"grid_dropout_kernel");}
void coarse_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const CoarseDropoutConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.holes<0||c.hole_width<=0||c.hole_height<=0)throw std::invalid_argument("invalid coarse dropout configuration");int n=c.width*c.height*c.channels;coarse_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"coarse_dropout_kernel");}
void mask_dropout_u8(const std::uint8_t* in,const std::uint8_t* mask,std::uint8_t* out,const MaskDropoutConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!mask||!out)throw std::invalid_argument("invalid mask dropout buffers");int n=c.width*c.height*c.channels;mask_kernel<<<(n+255)/256,256,0,s>>>(in,mask,out,c);check(cudaGetLastError(),"mask_dropout_kernel");}
void xy_masking_u8(const std::uint8_t* in,std::uint8_t* out,const XYMaskingConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||c.row_interval_count<0||c.column_interval_count<0||
     (c.row_interval_count>0&&!c.row_intervals)||(c.column_interval_count>0&&!c.column_intervals))
    throw std::invalid_argument("invalid XY masking configuration");
  const int n=c.width*c.height*c.channels;
  xy_masking_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);
  check(cudaGetLastError(),"xy_masking_kernel");
}
void cutout_u8(const std::uint8_t* in,std::uint8_t* out,const CutoutConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||c.rectangle_count<0||(c.rectangle_count>0&&!c.rectangles))
    throw std::invalid_argument("invalid Cutout configuration");
  const int n=c.width*c.height*c.channels;
  cutout_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);
  check(cudaGetLastError(),"cutout_kernel");
}
void total_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const TotalDropoutConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.probability)||c.probability<0.0f||c.probability>1.0f)
    throw std::invalid_argument("invalid TotalDropout configuration");
  const bool random_drop=uni(c.seed)<c.probability;
  const int n=c.width*c.height*c.channels;
  total_dropout_kernel<<<(n+255)/256,256,0,s>>>(in,out,c,random_drop);
  check(cudaGetLastError(),"total_dropout_kernel");
}
}
