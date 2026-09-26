#include "augmatch/annotations/transforms.hpp"
#include <algorithm>
#include <vector>
#include <stdexcept>
namespace augmatch {
namespace {
void validate(const Geometry& g) {
  if(g.input_width<=0||g.input_height<=0||g.channels<=0||g.output_width<=0||g.output_height<=0||g.crop_x<0||g.crop_y<0||g.crop_x+g.output_width>g.input_width||g.crop_y+g.output_height>g.input_height) throw std::invalid_argument("invalid image or crop geometry");
}
}
void transform_u8(const std::uint8_t* in,std::uint8_t* out,const Geometry& g,cudaStream_t) {
  validate(g);
  for(int y=0;y<g.output_height;++y) for(int x=0;x<g.output_width;++x) for(int c=0;c<g.channels;++c) {
    int sx=g.crop_x+(g.flip_horizontal?g.output_width-1-x:x);
    int sy=g.crop_y+(g.flip_vertical?g.output_height-1-y:y);
    out[(y*g.output_width+x)*g.channels+c]=in[(sy*g.input_width+sx)*g.channels+c];
  }
}
void horizontal_flip_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,cudaStream_t s){transform_u8(in,out,{w,h,c,0,0,w,h,true,false},s);}
void vertical_flip_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,cudaStream_t s){transform_u8(in,out,{w,h,c,0,0,w,h,false,true},s);}
void flip_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,int axis,cudaStream_t s){if(axis<0||axis>1)throw std::invalid_argument("flip axis must be 0 or 1");transform_u8(in,out,{w,h,c,0,0,w,h,axis==1,axis==0},s);}
void pool_u8(const std::uint8_t* in,std::uint8_t* out,const Pooling& p,cudaStream_t) {
  if(!in||!out||p.width<=0||p.height<=0||p.channels<=0||p.kernel_width<=0||p.kernel_height<=0||p.width%p.kernel_width||p.height%p.kernel_height||static_cast<int>(p.mode)<0||static_cast<int>(p.mode)>3)
    throw std::invalid_argument("pooling expects non-null buffers, positive sizes, and dimensions divisible by kernel size");
  const int pw=p.width/p.kernel_width, ph=p.height/p.kernel_height;
  std::vector<std::uint8_t> pooled(static_cast<size_t>(pw)*ph*p.channels);
  std::vector<std::uint8_t> values; values.reserve(static_cast<size_t>(p.kernel_width)*p.kernel_height);
  for(int y=0;y<ph;++y) for(int x=0;x<pw;++x) for(int c=0;c<p.channels;++c) {
    values.clear(); unsigned sum=0; unsigned char acc=p.mode==PoolMode::Minimum?255:0;
    for(int dy=0;dy<p.kernel_height;++dy) for(int dx=0;dx<p.kernel_width;++dx) {
      auto v=in[((y*p.kernel_height+dy)*p.width+x*p.kernel_width+dx)*p.channels+c];
      if(p.mode==PoolMode::Average) sum+=v;
      else if(p.mode==PoolMode::Maximum) acc=std::max(acc,v);
      else if(p.mode==PoolMode::Minimum) acc=std::min(acc,v);
      else values.push_back(v);
    }
    std::uint8_t result;
    if(p.mode==PoolMode::Average) result=static_cast<std::uint8_t>(sum/(p.kernel_width*p.kernel_height));
    else if(p.mode==PoolMode::Median) { std::sort(values.begin(),values.end()); const size_t n=values.size(); result=n%2?values[n/2]:static_cast<std::uint8_t>((static_cast<unsigned>(values[n/2-1])+values[n/2])/2); }
    else result=acc;
    pooled[(y*pw+x)*p.channels+c]=result;
  }
  const int ow=p.keep_size?p.width:pw, oh=p.keep_size?p.height:ph;
  for(int y=0;y<oh;++y) for(int x=0;x<ow;++x) for(int c=0;c<p.channels;++c) {
    const int sx=p.keep_size?(x*pw)/ow:x, sy=p.keep_size?(y*ph)/oh:y;
    out[(y*ow+x)*p.channels+c]=pooled[(sy*pw+sx)*p.channels+c];
  }
}
void transform_mask_u8(const std::uint8_t* in,std::uint8_t* out,const Geometry& g,
                       MaskInterpolation interpolation,cudaStream_t s) {
  if (static_cast<int>(interpolation) < 0 || static_cast<int>(interpolation) > 1)
    throw std::invalid_argument("unknown mask interpolation policy");
  // Geometry has crop and flips only, so the source sample is an exact pixel
  // centre. Linear and nearest therefore coincide until a scale is introduced.
  transform_u8(in, out, g, s);
}
}
