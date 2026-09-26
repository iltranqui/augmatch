#include "augmatch/voronoi.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace augmatch { namespace {
std::uint64_t mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
void valid(int w,int h) { if(w<=0||h<=0) throw std::invalid_argument("invalid Voronoi dimensions"); }
void valid_points(int w,int h,const VoronoiPoint* p,int n) {
  valid(w,h); if(n<=0) throw std::invalid_argument("Voronoi point_count must be positive");
  if(p) for(int i=0;i<n;++i) if(p[i].x<0||p[i].x>=w||p[i].y<0||p[i].y>=h) throw std::invalid_argument("Voronoi point outside image");
}
VoronoiPoint random_point(int w,int h,std::uint64_t seed,int i) {
  const std::uint64_t a=mix(seed+static_cast<std::uint64_t>(i)*0x9e3779b97f4a7c15ULL);
  const std::uint64_t b=mix(a+0x632be59bd9b4e019ULL);
  return {static_cast<int>(a%static_cast<std::uint64_t>(w)),static_cast<int>(b%static_cast<std::uint64_t>(h))};
}
VoronoiPoint uniform_point(int w,int h,std::uint64_t seed,int i,int n) {
  const int gx=static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
  const int gy=(n+gx-1)/gx;
  const int col=i%gx,row=i/gx;
  const std::uint64_t r=mix(seed+static_cast<std::uint64_t>(i)*0x9e3779b97f4a7c15ULL);
  const std::uint64_t q=mix(r+0x632be59bd9b4e019ULL);
  int x=static_cast<int>((static_cast<std::uint64_t>(col)*w + (r%static_cast<std::uint64_t>(w/gx+1)))/gx);
  int y=static_cast<int>((static_cast<std::uint64_t>(row)*h + (q%static_cast<std::uint64_t>(h/gy+1)))/gy);
  return {std::min(w-1,std::max(0,x)),std::min(h-1,std::max(0,y))};
}
void grid_points(std::vector<VoronoiPoint>& p,int w,int h,int gx,int gy) {
  for(int row=0;row<gy;++row) for(int col=0;col<gx;++col) {
    const int x0=col*w/gx,x1=(col+1)*w/gx,y0=row*h/gy,y1=(row+1)*h/gy;
    p.push_back({std::min(w-1,(x0+x1)/2),std::min(h-1,(y0+y1)/2)});
  }
}
template<class PointFn, class Out>
void partition(Out* out,int w,int h,int n,PointFn point,const std::uint8_t* in) {
  for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
    std::uint64_t best=std::numeric_limits<std::uint64_t>::max(); int winner=0;
    for(int i=0;i<n;++i) { const VoronoiPoint p=point(i); const std::int64_t dx=x-p.x,dy=y-p.y; const std::uint64_t d=static_cast<std::uint64_t>(dx*dx+dy*dy); if(d<best){best=d;winner=i;} }
    const std::size_t at=static_cast<std::size_t>(y)*w+x;
    out[at]=in?static_cast<Out>(in[static_cast<std::size_t>(point(winner).y)*w+point(winner).x]):static_cast<Out>(winner);
  }
}
void check_relative(const RelativeRegularGridVoronoiConfig& c) { valid(c.width,c.height); if(!(c.relative_grid_width>0&&c.relative_grid_width<=1&&c.relative_grid_height>0&&c.relative_grid_height<=1)||!std::isfinite(c.relative_grid_width)||!std::isfinite(c.relative_grid_height)) throw std::invalid_argument("relative Voronoi grid fractions must be finite in (0,1]"); }
}}

namespace augmatch {
void voronoi_u8(const std::uint8_t* in,std::uint8_t* out,const VoronoiConfig& c,cudaStream_t) {
  if(!in||!out) throw std::invalid_argument("null Voronoi buffer"); valid_points(c.width,c.height,c.points,c.point_count);
  partition(out,c.width,c.height,c.point_count,[&](int i){return c.points?c.points[i]:random_point(c.width,c.height,c.seed,i);},in);
}
void voronoi_labels_u32(std::uint32_t* out,const VoronoiConfig& c,cudaStream_t) {
  if(!out) throw std::invalid_argument("null Voronoi output"); valid_points(c.width,c.height,c.points,c.point_count);
  partition(out,c.width,c.height,c.point_count,[&](int i){return c.points?c.points[i]:random_point(c.width,c.height,c.seed,i);},nullptr);
}
void uniform_voronoi_u8(const std::uint8_t* in,std::uint8_t* out,const UniformVoronoiConfig& c,cudaStream_t) {
  if(!in||!out) throw std::invalid_argument("null UniformVoronoi buffer"); valid_points(c.width,c.height,nullptr,c.point_count);
  partition(out,c.width,c.height,c.point_count,[&](int i){return uniform_point(c.width,c.height,c.seed,i,c.point_count);},in);
}
void uniform_voronoi_labels_u32(std::uint32_t* out,const UniformVoronoiConfig& c,cudaStream_t) {
  if(!out) throw std::invalid_argument("null UniformVoronoi output"); valid_points(c.width,c.height,nullptr,c.point_count);
  partition(out,c.width,c.height,c.point_count,[&](int i){return uniform_point(c.width,c.height,c.seed,i,c.point_count);},nullptr);
}
void regular_grid_voronoi_u8(const std::uint8_t* in,std::uint8_t* out,const RegularGridVoronoiConfig& c,cudaStream_t) {
  if(!in||!out) throw std::invalid_argument("null RegularGridVoronoi buffer"); valid(c.width,c.height); if(c.grid_width<=0||c.grid_height<=0) throw std::invalid_argument("invalid Voronoi grid"); std::vector<VoronoiPoint> p; p.reserve(static_cast<std::size_t>(c.grid_width)*c.grid_height); grid_points(p,c.width,c.height,c.grid_width,c.grid_height); partition(out,c.width,c.height,static_cast<int>(p.size()),[&](int i){return p[i];},in);
}
void regular_grid_voronoi_labels_u32(std::uint32_t* out,const RegularGridVoronoiConfig& c,cudaStream_t) {
  if(!out) throw std::invalid_argument("null RegularGridVoronoi output"); valid(c.width,c.height); if(c.grid_width<=0||c.grid_height<=0) throw std::invalid_argument("invalid Voronoi grid"); std::vector<VoronoiPoint> p; p.reserve(static_cast<std::size_t>(c.grid_width)*c.grid_height); grid_points(p,c.width,c.height,c.grid_width,c.grid_height); partition(out,c.width,c.height,static_cast<int>(p.size()),[&](int i){return p[i];},nullptr);
}
void relative_regular_grid_voronoi_u8(const std::uint8_t* in,std::uint8_t* out,const RelativeRegularGridVoronoiConfig& c,cudaStream_t) {
  if(!in||!out) throw std::invalid_argument("null RelativeRegularGridVoronoi buffer"); check_relative(c); const int gx=std::max(1,static_cast<int>(std::ceil(1.0f/c.relative_grid_width))),gy=std::max(1,static_cast<int>(std::ceil(1.0f/c.relative_grid_height))); std::vector<VoronoiPoint> p; grid_points(p,c.width,c.height,gx,gy); partition(out,c.width,c.height,static_cast<int>(p.size()),[&](int i){return p[i];},in);
}
void relative_regular_grid_voronoi_labels_u32(std::uint32_t* out,const RelativeRegularGridVoronoiConfig& c,cudaStream_t) {
  if(!out) throw std::invalid_argument("null RelativeRegularGridVoronoi output"); check_relative(c); const int gx=std::max(1,static_cast<int>(std::ceil(1.0f/c.relative_grid_width))),gy=std::max(1,static_cast<int>(std::ceil(1.0f/c.relative_grid_height))); std::vector<VoronoiPoint> p; grid_points(p,c.width,c.height,gx,gy); partition(out,c.width,c.height,static_cast<int>(p.size()),[&](int i){return p[i];},nullptr);
}
}
