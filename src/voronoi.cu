#include "augmatch/filter/voronoi.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace augmatch { namespace {
__device__ unsigned long long mix_device(unsigned long long x) { x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
__device__ VoronoiPoint random_point_device(int w,int h,unsigned long long seed,int i) { const auto a=mix_device(seed+static_cast<unsigned long long>(i)*0x9e3779b97f4a7c15ULL),b=mix_device(a+0x632be59bd9b4e019ULL); return {static_cast<int>(a%static_cast<unsigned long long>(w)),static_cast<int>(b%static_cast<unsigned long long>(h))}; }
__device__ VoronoiPoint uniform_point_device(int w,int h,unsigned long long seed,int i,int n) { const int gx=static_cast<int>(ceil(sqrt(static_cast<double>(n)))),gy=(n+gx-1)/gx,col=i%gx,row=i/gx; const auto r=mix_device(seed+static_cast<unsigned long long>(i)*0x9e3779b97f4a7c15ULL),q=mix_device(r+0x632be59bd9b4e019ULL); int x=static_cast<int>((static_cast<unsigned long long>(col)*w+r%static_cast<unsigned long long>(w/gx+1))/gx),y=static_cast<int>((static_cast<unsigned long long>(row)*h+q%static_cast<unsigned long long>(h/gy+1))/gy); return {min(w-1,max(0,x)),min(h-1,max(0,y))}; }
__device__ VoronoiPoint grid_point_device(int w,int h,int gx,int gy,int i) { const int col=i%gx,row=i/gx,x0=col*w/gx,x1=(col+1)*w/gx,y0=row*h/gy,y1=(row+1)*h/gy; return {min(w-1,(x0+x1)/2),min(h-1,(y0+y1)/2)}; }
template<int Kind> __global__ void labels_kernel(std::uint32_t* out,int w,int h,int n,unsigned long long seed,const VoronoiPoint* points,int gx,int gy) {
  const std::size_t at=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=static_cast<std::size_t>(w)*h)return; const int x=static_cast<int>(at%w),y=static_cast<int>(at/w); unsigned long long best=~0ULL; int winner=0;
  for(int i=0;i<n;++i){VoronoiPoint p; if(Kind==0)p=points?points[i]:random_point_device(w,h,seed,i); else if(Kind==1)p=uniform_point_device(w,h,seed,i,n); else p=grid_point_device(w,h,gx,gy,i); long long dx=static_cast<long long>(x)-p.x,dy=static_cast<long long>(y)-p.y; auto d=static_cast<unsigned long long>(dx*dx+dy*dy);if(d<best){best=d;winner=i;}}
  out[at]=static_cast<std::uint32_t>(winner);
}
template<int Kind> __global__ void mask_kernel(const std::uint8_t* in,std::uint8_t* out,int w,int h,int n,unsigned long long seed,const VoronoiPoint* points,int gx,int gy) {
  const std::size_t at=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=static_cast<std::size_t>(w)*h)return; const int x=static_cast<int>(at%w),y=static_cast<int>(at/w); unsigned long long best=~0ULL; int winner=0;
  for(int i=0;i<n;++i){VoronoiPoint p; if(Kind==0)p=points?points[i]:random_point_device(w,h,seed,i); else if(Kind==1)p=uniform_point_device(w,h,seed,i,n); else p=grid_point_device(w,h,gx,gy,i); long long dx=static_cast<long long>(x)-p.x,dy=static_cast<long long>(y)-p.y; auto d=static_cast<unsigned long long>(dx*dx+dy*dy);if(d<best){best=d;winner=i;}}
  VoronoiPoint p; if(Kind==0)p=points?points[winner]:random_point_device(w,h,seed,winner); else if(Kind==1)p=uniform_point_device(w,h,seed,winner,n); else p=grid_point_device(w,h,gx,gy,winner); out[at]=in[static_cast<std::size_t>(p.y)*w+p.x];
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("Voronoi CUDA: ")+cudaGetErrorString(e));}
void dims(int w,int h){if(w<=0||h<=0)throw std::invalid_argument("invalid Voronoi dimensions");}
void count(int w,int h,int n,const VoronoiPoint* p){dims(w,h);if(n<=0)throw std::invalid_argument("Voronoi point_count must be positive");(void)p;}
void relative(const RelativeRegularGridVoronoiConfig& c){dims(c.width,c.height);if(!(c.relative_grid_width>0&&c.relative_grid_width<=1&&c.relative_grid_height>0&&c.relative_grid_height<=1)||!std::isfinite(c.relative_grid_width)||!std::isfinite(c.relative_grid_height))throw std::invalid_argument("relative Voronoi grid fractions must be finite in (0,1]");}
template<int K> void launch_mask(const std::uint8_t* in,std::uint8_t* out,int w,int h,int n,unsigned long long seed,const VoronoiPoint* p,int gx,int gy,cudaStream_t s){mask_kernel<K><<<static_cast<unsigned int>((static_cast<std::size_t>(w)*h+255)/256),256,0,s>>>(in,out,w,h,n,seed,p,gx,gy);check(cudaGetLastError());}
template<int K> void launch_labels(std::uint32_t* out,int w,int h,int n,unsigned long long seed,const VoronoiPoint* p,int gx,int gy,cudaStream_t s){labels_kernel<K><<<static_cast<unsigned int>((static_cast<std::size_t>(w)*h+255)/256),256,0,s>>>(out,w,h,n,seed,p,gx,gy);check(cudaGetLastError());}
}}
namespace augmatch {
void voronoi_u8(const std::uint8_t* i,std::uint8_t* o,const VoronoiConfig& c,cudaStream_t s){if(!i||!o)throw std::invalid_argument("null Voronoi buffer");count(c.width,c.height,c.point_count,c.points);launch_mask<0>(i,o,c.width,c.height,c.point_count,c.seed,c.points,0,0,s);}
void voronoi_labels_u32(std::uint32_t* o,const VoronoiConfig& c,cudaStream_t s){if(!o)throw std::invalid_argument("null Voronoi output");count(c.width,c.height,c.point_count,c.points);launch_labels<0>(o,c.width,c.height,c.point_count,c.seed,c.points,0,0,s);}
void uniform_voronoi_u8(const std::uint8_t* i,std::uint8_t* o,const UniformVoronoiConfig& c,cudaStream_t s){if(!i||!o)throw std::invalid_argument("null UniformVoronoi buffer");count(c.width,c.height,c.point_count,nullptr);launch_mask<1>(i,o,c.width,c.height,c.point_count,c.seed,nullptr,0,0,s);}
void uniform_voronoi_labels_u32(std::uint32_t* o,const UniformVoronoiConfig& c,cudaStream_t s){if(!o)throw std::invalid_argument("null UniformVoronoi output");count(c.width,c.height,c.point_count,nullptr);launch_labels<1>(o,c.width,c.height,c.point_count,c.seed,nullptr,0,0,s);}
void regular_grid_voronoi_u8(const std::uint8_t* i,std::uint8_t* o,const RegularGridVoronoiConfig& c,cudaStream_t s){if(!i||!o)throw std::invalid_argument("null RegularGridVoronoi buffer");dims(c.width,c.height);if(c.grid_width<=0||c.grid_height<=0)throw std::invalid_argument("invalid Voronoi grid");launch_mask<2>(i,o,c.width,c.height,c.grid_width*c.grid_height,0,nullptr,c.grid_width,c.grid_height,s);}
void regular_grid_voronoi_labels_u32(std::uint32_t* o,const RegularGridVoronoiConfig& c,cudaStream_t s){if(!o)throw std::invalid_argument("null RegularGridVoronoi output");dims(c.width,c.height);if(c.grid_width<=0||c.grid_height<=0)throw std::invalid_argument("invalid Voronoi grid");launch_labels<2>(o,c.width,c.height,c.grid_width*c.grid_height,0,nullptr,c.grid_width,c.grid_height,s);}
void relative_regular_grid_voronoi_u8(const std::uint8_t* i,std::uint8_t* o,const RelativeRegularGridVoronoiConfig& c,cudaStream_t s){if(!i||!o)throw std::invalid_argument("null RelativeRegularGridVoronoi buffer");relative(c);const int gx=std::max(1,static_cast<int>(std::ceil(1.0f/c.relative_grid_width))),gy=std::max(1,static_cast<int>(std::ceil(1.0f/c.relative_grid_height)));launch_mask<2>(i,o,c.width,c.height,gx*gy,0,nullptr,gx,gy,s);}
void relative_regular_grid_voronoi_labels_u32(std::uint32_t* o,const RelativeRegularGridVoronoiConfig& c,cudaStream_t s){if(!o)throw std::invalid_argument("null RelativeRegularGridVoronoi output");relative(c);const int gx=std::max(1,static_cast<int>(std::ceil(1.0f/c.relative_grid_width))),gy=std::max(1,static_cast<int>(std::ceil(1.0f/c.relative_grid_height)));launch_labels<2>(o,c.width,c.height,gx*gy,0,nullptr,gx,gy,s);}
}
