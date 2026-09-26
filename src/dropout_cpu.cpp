#include "augmatch/dropout.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace augmatch { namespace { void check(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid dropout dimensions");} std::uint64_t mix(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} float uni(std::uint64_t x){return static_cast<float>((mix(x)>>11)*(1.0/9007199254740992.0));} }
void pixel_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const PixelDropoutConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.probability<0||c.probability>1)throw std::invalid_argument("invalid pixel dropout configuration");for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=uni(c.seed+static_cast<std::uint64_t>(i)*0x9e3779b97f4a7c15ULL)<c.probability?c.replacement:in[i];}
void channel_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const ChannelDropoutConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.probability<0||c.probability>1)throw std::invalid_argument("invalid channel dropout configuration");for(int ch=0;ch<c.channels;++ch){bool drop=uni(c.seed+static_cast<std::uint64_t>(ch)*0x9e3779b97f4a7c15ULL)<c.probability;for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){int i=(y*c.width+x)*c.channels+ch;out[i]=drop?c.replacement:in[i];}}}
void grid_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const GridDropoutConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.cell_width<=0||c.cell_height<=0||c.ratio<0||c.ratio>1)throw std::invalid_argument("invalid grid dropout configuration");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){bool drop=(static_cast<float>(x%c.cell_width)/c.cell_width<c.ratio)&&(static_cast<float>(y%c.cell_height)/c.cell_height<c.ratio);for(int ch=0;ch<c.channels;++ch){int i=(y*c.width+x)*c.channels+ch;out[i]=drop?c.fill:in[i];}}}
void coarse_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const CoarseDropoutConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.holes<0||c.hole_width<=0||c.hole_height<=0)throw std::invalid_argument("invalid coarse dropout configuration");std::copy(in,in+static_cast<std::size_t>(c.width)*c.height*c.channels,out);for(int h=0;h<c.holes;++h){int x=static_cast<int>(uni(c.seed+2*h)*c.width),y=static_cast<int>(uni(c.seed+2*h+1)*c.height);for(int yy=y;yy<std::min(c.height,y+c.hole_height);++yy)for(int xx=x;xx<std::min(c.width,x+c.hole_width);++xx)for(int ch=0;ch<c.channels;++ch)out[(yy*c.width+xx)*c.channels+ch]=c.fill;}}
void mask_dropout_u8(const std::uint8_t* in,const std::uint8_t* mask,std::uint8_t* out,const MaskDropoutConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!mask||!out)throw std::invalid_argument("invalid mask dropout buffers");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const bool drop=mask[y*c.width+x]!=0;for(int ch=0;ch<c.channels;++ch){const int i=(y*c.width+x)*c.channels+ch;out[i]=drop?c.fill:in[i];}}}
void xy_masking_u8(const std::uint8_t* in,std::uint8_t* out,const XYMaskingConfig& c,cudaStream_t){
  check(c.width,c.height,c.channels);
  if(!in||!out||c.row_interval_count<0||c.column_interval_count<0||
     (c.row_interval_count>0&&!c.row_intervals)||(c.column_interval_count>0&&!c.column_intervals))
    throw std::invalid_argument("invalid XY masking configuration");
  for(int i=0;i<c.row_interval_count;++i)
    if(c.row_intervals[i].begin<0||c.row_intervals[i].begin>c.row_intervals[i].end||c.row_intervals[i].end>c.height)
      throw std::invalid_argument("row masking interval is outside the image");
  for(int i=0;i<c.column_interval_count;++i)
    if(c.column_intervals[i].begin<0||c.column_intervals[i].begin>c.column_intervals[i].end||c.column_intervals[i].end>c.width)
      throw std::invalid_argument("column masking interval is outside the image");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x){
    bool drop=false;
    for(int i=0;i<c.row_interval_count;++i) drop=drop||(y>=c.row_intervals[i].begin&&y<c.row_intervals[i].end);
    for(int i=0;i<c.column_interval_count;++i) drop=drop||(x>=c.column_intervals[i].begin&&x<c.column_intervals[i].end);
    for(int ch=0;ch<c.channels;++ch){const int p=(y*c.width+x)*c.channels+ch;out[p]=drop?c.fill:in[p];}
  }
}
void cutout_u8(const std::uint8_t* in,std::uint8_t* out,const CutoutConfig& c,cudaStream_t){
  check(c.width,c.height,c.channels);
  if(!in||!out||c.rectangle_count<0||(c.rectangle_count>0&&!c.rectangles))
    throw std::invalid_argument("invalid Cutout configuration");
  for(int i=0;i<c.rectangle_count;++i){
    const CutoutRectangle& r=c.rectangles[i];
    if(r.x0<0||r.y0<0||r.x1<r.x0||r.y1<r.y0||r.x1>c.width||r.y1>c.height)
      throw std::invalid_argument("Cutout rectangle is outside the image");
  }
  std::copy(in,in+static_cast<std::size_t>(c.width)*c.height*c.channels,out);
  for(int i=0;i<c.rectangle_count;++i){
    const CutoutRectangle& r=c.rectangles[i];
    for(int y=r.y0;y<r.y1;++y) for(int x=r.x0;x<r.x1;++x)
      for(int ch=0;ch<c.channels;++ch) out[(y*c.width+x)*c.channels+ch]=c.fill;
  }
}
void total_dropout_u8(const std::uint8_t* in,std::uint8_t* out,const TotalDropoutConfig& c,cudaStream_t){
  check(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.probability)||c.probability<0.0f||c.probability>1.0f)
    throw std::invalid_argument("invalid TotalDropout configuration");
  const bool drop=c.mask ? (*c.mask!=0) : (uni(c.seed)<c.probability);
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;
  if(drop) std::fill(out,out+n,c.fill); else std::copy(in,in+n,out);
}
}
