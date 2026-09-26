#include "augmatch/transforms.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__global__ void transform_kernel(const std::uint8_t* in,std::uint8_t* out,int iw,int channels,int cx,int cy,int ow,int oh,bool fh,bool fv) {
  int i=blockIdx.x*blockDim.x+threadIdx.x, n=ow*oh*channels; if(i>=n)return;
  int c=i%channels,x=(i/channels)%ow,y=i/(channels*ow);
  int sx=cx+(fh?ow-1-x:x), sy=cy+(fv?oh-1-y:y);
  out[i]=in[(sy*iw+sx)*channels+c];
}
__device__ int kth_value(const std::uint8_t* in,int x,int y,int c,int width,int channels,int kw,int kh,int rank) {
  int lo=0,hi=255;
  while(lo<hi){int mid=(lo+hi)/2,count=0;for(int dy=0;dy<kh;++dy)for(int dx=0;dx<kw;++dx)count+=(in[((y*kh+dy)*width+x*kw+dx)*channels+c]<=mid);if(count>rank)hi=mid;else lo=mid+1;}
  return lo;
}
__device__ std::uint8_t pooled_value(const std::uint8_t* in,int x,int y,int c,int width,int channels,int kw,int kh,int mode) {
  const int n=kw*kh;
  if(mode==0){unsigned sum=0;for(int dy=0;dy<kh;++dy)for(int dx=0;dx<kw;++dx)sum+=in[((y*kh+dy)*width+x*kw+dx)*channels+c];return static_cast<std::uint8_t>(sum/n);}
  if(mode==1||mode==2){unsigned char a=mode==1?0:255;for(int dy=0;dy<kh;++dy)for(int dx=0;dx<kw;++dx){auto v=in[((y*kh+dy)*width+x*kw+dx)*channels+c];a=mode==1?(v>a?v:a):(v<a?v:a);}return a;}
  // Binary-search the value at a rank; this handles duplicates exactly.
  int upper=kth_value(in,x,y,c,width,channels,kw,kh,n/2);if(n%2)return static_cast<std::uint8_t>(upper);int lower=kth_value(in,x,y,c,width,channels,kw,kh,n/2-1);return static_cast<std::uint8_t>((lower+upper)/2);
}
__global__ void pool_kernel(const std::uint8_t* in,std::uint8_t* out,int width,int height,int channels,int kw,int kh,int pw,int ph,int ow,int oh,int mode,bool keep) {
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=ow*oh*channels;if(i>=n)return;
  int c=i%channels,x=(i/channels)%ow,y=i/(channels*ow);int px=keep?(x*pw)/ow:x,py=keep?(y*ph)/oh:y;
  out[i]=pooled_value(in,px,py,c,width,channels,kw,kh,mode);
}
void check(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void validate(const Geometry& g){if(g.input_width<=0||g.input_height<=0||g.channels<=0||g.output_width<=0||g.output_height<=0||g.crop_x<0||g.crop_y<0||g.crop_x+g.output_width>g.input_width||g.crop_y+g.output_height>g.input_height)throw std::invalid_argument("invalid image or crop geometry");}
}
void transform_u8(const std::uint8_t* in,std::uint8_t* out,const Geometry& g,cudaStream_t stream){validate(g);int n=g.output_width*g.output_height*g.channels;transform_kernel<<<(n+255)/256,256,0,stream>>>(in,out,g.input_width,g.channels,g.crop_x,g.crop_y,g.output_width,g.output_height,g.flip_horizontal,g.flip_vertical);check(cudaGetLastError(),"transform_kernel launch");}
void horizontal_flip_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,cudaStream_t s){transform_u8(in,out,{w,h,c,0,0,w,h,true,false},s);}
void vertical_flip_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,cudaStream_t s){transform_u8(in,out,{w,h,c,0,0,w,h,false,true},s);}
void flip_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,int axis,cudaStream_t s){if(axis<0||axis>1)throw std::invalid_argument("flip axis must be 0 or 1");transform_u8(in,out,{w,h,c,0,0,w,h,axis==1,axis==0},s);}
void pool_u8(const std::uint8_t* in,std::uint8_t* out,const Pooling& p,cudaStream_t stream){if(!in||!out||p.width<=0||p.height<=0||p.channels<=0||p.kernel_width<=0||p.kernel_height<=0||p.width%p.kernel_width||p.height%p.kernel_height||static_cast<int>(p.mode)<0||static_cast<int>(p.mode)>3)throw std::invalid_argument("invalid pooling parameters");int pw=p.width/p.kernel_width,ph=p.height/p.kernel_height,ow=p.keep_size?p.width:pw,oh=p.keep_size?p.height:ph,n=ow*oh*p.channels;pool_kernel<<<(n+255)/256,256,0,stream>>>(in,out,p.width,p.height,p.channels,p.kernel_width,p.kernel_height,pw,ph,ow,oh,static_cast<int>(p.mode),p.keep_size);check(cudaGetLastError(),"pool_kernel launch");}
void transform_mask_u8(const std::uint8_t* in,std::uint8_t* out,const Geometry& g,
                       MaskInterpolation interpolation,cudaStream_t stream) {
  if (static_cast<int>(interpolation) < 0 || static_cast<int>(interpolation) > 1)
    throw std::invalid_argument("unknown mask interpolation policy");
  // Geometry has crop and flips only, so both policies sample exact pixel
  // centres. A future scale must implement linear interpolation here.
  transform_u8(in, out, g, stream);
}
}
