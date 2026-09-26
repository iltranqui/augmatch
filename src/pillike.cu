#include "augmatch/pillike.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__device__ int pillike_luma(const unsigned char* p){return (299*(int)p[0]+587*(int)p[1]+114*(int)p[2]+500)/1000;}
__device__ unsigned char pillike_blend(float base,float image,float factor){int value=(int)(base+factor*(image-base));return (unsigned char)max(0,min(255,value));}
__device__ int pillike_round_ties_even(int sum,int scale){const int quotient=sum/scale,remainder=sum%scale,absolute_remainder=remainder<0?-remainder:remainder;const bool increment=absolute_remainder*2>scale||(absolute_remainder*2==scale&&(quotient&1));return quotient+(increment?(sum<0?-1:1):0);}
__global__ void enhance_kernel(const unsigned char* in,unsigned char* out,int w,int h,int c,float factor,int mode){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=w*h;if(i>=n)return;const unsigned char* p=in+(size_t)i*c;unsigned char* q=out+(size_t)i*c;int limit=c>=3?3:1;float base=0;
  if(mode==0)base=(float)pillike_luma(p); else if(mode==1){long long sum=0;for(int j=0;j<n;++j){const unsigned char* s=in+(size_t)j*c;sum+=c>=3?pillike_luma(s):s[0];}base=(float)((sum+n/2)/n);} 
  if(mode==2)base=0;
  if(mode==3){for(int ch=0;ch<limit;++ch){float smooth=(float)p[ch];if(i/w>0&&i/w+1<h&&i%w>0&&i%w+1<w){const int x=i%w,y=i/w;const int k[9]={1,1,1,1,5,1,1,1,1};int sum=0;for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)sum+=k[(dy+1)*3+dx+1]*in[((size_t)(y+dy)*w+x+dx)*c+ch];smooth=(float)((sum+6)/13);}q[ch]=pillike_blend(smooth,p[ch],factor);}for(int ch=limit;ch<c;++ch)q[ch]=p[ch];return;}
  for(int ch=0;ch<limit;++ch)q[ch]=pillike_blend(base,p[ch],factor);for(int ch=limit;ch<c;++ch)q[ch]=p[ch];
}
__global__ void filter_kernel(const unsigned char* in,unsigned char* out,int w,int h,int c,int mode){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=w*h*c;if(i>=n)return;const int ch=i%c,x=(i/c)%w,y=i/(w*c),radius=mode<=2&& (mode==0||mode==2)?2:1;const size_t pos=(size_t)i;
  if(x<radius||y<radius||x+radius>=w||y+radius>=h){out[pos]=in[pos];return;}
  int sum=0,scale=16,offset=0;for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){int weight=0;
    if(mode==0){const int ax=abs(dx),ay=abs(dy);weight=(ax==2||ay==2)?1:0;scale=16;}
    else if(mode==1){weight=(dx==0&&dy==0)?5:1;scale=13;}
    else if(mode==2){const int ax=abs(dx),ay=abs(dy);if(ax==2||ay==2)weight=1;else if(ax==1||ay==1)weight=(dx==0&&dy==0)?44:5;else weight=44;scale=100;}
    else if(mode==3){weight=(dx==0&&dy==0)?10:-1;scale=2;}
    else if(mode==4){weight=(dx==0&&dy==0)?9:-1;scale=1;}
    else if(mode==5){weight=(dx==0&&dy==0)?8:-1;scale=1;}
    else if(mode==6){weight=(dx==0&&dy==0)?8:-1;scale=1;offset=255;}
    else if(mode==7){weight=(dx==-1&&dy==1)?-1:(dx==0&&dy==0)?1:0;scale=1;offset=128;}
    else if(mode==8){weight=(dx==0&&dy==0)?32:-2;scale=16;}
    else {weight=((dx==0&&dy==0)?10:((dx==0||dy==0)?-1:0));scale=6;}
    sum+=weight*in[((size_t)(y+dy)*w+x+dx)*c+ch];}
  const int rounded=(mode==2?pillike_round_ties_even(sum,scale):(sum+scale/2)/scale)+offset;out[pos]=(unsigned char)max(0,min(255,rounded));
}
void check(const unsigned char* in,const unsigned char* out,int w,int h,int c,float factor,bool rgb){if(!in||!out||w<=0||h<=0||c<=0||!isfinite(factor)||(rgb&&c<3))throw std::invalid_argument("invalid Pillow enhancement configuration");}
void check_filter(const unsigned char* in,const unsigned char* out,int w,int h,int c){if(!in||!out||w<=0||h<=0||c<=0)throw std::invalid_argument("invalid Pillow filter configuration");}
void launch_enhance(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,float factor,int mode,cudaStream_t s){const int n=w*h;enhance_kernel<<<(n+255)/256,256,0,s>>>(in,out,w,h,c,factor,mode);if(cudaError_t e=cudaGetLastError();e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void launch_filter(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,int mode,cudaStream_t s){const int n=w*h*c;filter_kernel<<<(n+255)/256,256,0,s>>>(in,out,w,h,c,mode);if(cudaError_t e=cudaGetLastError();e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
void enhance_color_u8(const std::uint8_t* in,std::uint8_t* out,const EnhanceColorConfig& c,cudaStream_t s){check(in,out,c.width,c.height,c.channels,c.factor,true);launch_enhance(in,out,c.width,c.height,c.channels,c.factor,0,s);}
void enhance_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const EnhanceContrastConfig& c,cudaStream_t s){check(in,out,c.width,c.height,c.channels,c.factor,false);launch_enhance(in,out,c.width,c.height,c.channels,c.factor,1,s);}
void enhance_brightness_u8(const std::uint8_t* in,std::uint8_t* out,const EnhanceBrightnessConfig& c,cudaStream_t s){check(in,out,c.width,c.height,c.channels,c.factor,false);launch_enhance(in,out,c.width,c.height,c.channels,c.factor,2,s);}
void enhance_sharpness_u8(const std::uint8_t* in,std::uint8_t* out,const EnhanceSharpnessConfig& c,cudaStream_t s){check(in,out,c.width,c.height,c.channels,c.factor,false);launch_enhance(in,out,c.width,c.height,c.channels,c.factor,3,s);}
void filter_blur_u8(const std::uint8_t* in,std::uint8_t* out,const FilterBlurConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,0,s);}
void filter_smooth_u8(const std::uint8_t* in,std::uint8_t* out,const FilterSmoothConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,1,s);}
void filter_smooth_more_u8(const std::uint8_t* in,std::uint8_t* out,const FilterSmoothMoreConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,2,s);}
void filter_edge_enhance_u8(const std::uint8_t* in,std::uint8_t* out,const FilterEdgeEnhanceConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,3,s);}
void filter_edge_enhance_more_u8(const std::uint8_t* in,std::uint8_t* out,const FilterEdgeEnhanceMoreConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,4,s);}
void filter_find_edges_u8(const std::uint8_t* in,std::uint8_t* out,const FilterFindEdgesConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,5,s);}
void filter_contour_u8(const std::uint8_t* in,std::uint8_t* out,const FilterContourConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,6,s);}
void filter_emboss_u8(const std::uint8_t* in,std::uint8_t* out,const FilterEmbossConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,7,s);}
void filter_sharpen_u8(const std::uint8_t* in,std::uint8_t* out,const FilterSharpenConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,8,s);}
void filter_detail_u8(const std::uint8_t* in,std::uint8_t* out,const FilterDetailConfig& c,cudaStream_t s){check_filter(in,out,c.width,c.height,c.channels);launch_filter(in,out,c.width,c.height,c.channels,9,s);}
}
