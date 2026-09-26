#include "augmatch/pillike.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace augmatch { namespace {
void check(const std::uint8_t* in,const std::uint8_t* out,int w,int h,int c,float factor,bool rgb){
  if(!in||!out||w<=0||h<=0||c<=0||!std::isfinite(factor)||(rgb&&c<3)) throw std::invalid_argument("invalid Pillow enhancement configuration");
}
void check_filter(const std::uint8_t* in,const std::uint8_t* out,int w,int h,int c){if(!in||!out||w<=0||h<=0||c<=0)throw std::invalid_argument("invalid Pillow filter configuration");}
int luma(const std::uint8_t* p){return (299*static_cast<int>(p[0])+587*static_cast<int>(p[1])+114*static_cast<int>(p[2])+500)/1000;}
std::uint8_t blend(float base,float image,float factor){const float value=base+factor*(image-base);return static_cast<std::uint8_t>(std::max(0,std::min(255,static_cast<int>(value))));}
int round_ties_even(int sum,int scale){const int quotient=sum/scale,remainder=sum%scale,absolute_remainder=remainder<0?-remainder:remainder;const bool increment=absolute_remainder*2>scale||(absolute_remainder*2==scale&&(quotient&1));return quotient+(increment?(sum<0?-1:1):0);}
void filter_impl(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,const int* kernel,int side,int scale,int offset=0,int rounding=0){
  check_filter(in,out,w,h,c); const int radius=side/2;
  for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int ch=0;ch<c;++ch){
    if(x<radius||y<radius||x+radius>=w||y+radius>=h){out[(static_cast<std::size_t>(y)*w+x)*c+ch]=in[(static_cast<std::size_t>(y)*w+x)*c+ch];continue;}
    int sum=0;for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx)sum+=kernel[(dy+radius)*side+dx+radius]*in[(static_cast<std::size_t>(y+dy)*w+x+dx)*c+ch];
    int divided=(sum+scale/2)/scale;if(rounding==1)divided=(sum+scale/2-1)/scale;else if(rounding==2)divided=round_ties_even(sum,scale);const int rounded=divided+offset;out[(static_cast<std::size_t>(y)*w+x)*c+ch]=static_cast<std::uint8_t>(std::max(0,std::min(255,rounded)));
  }
}
}
void enhance_color_u8(const std::uint8_t* in,std::uint8_t* out,const EnhanceColorConfig& c,cudaStream_t){
  check(in,out,c.width,c.height,c.channels,c.factor,true);for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+static_cast<std::size_t>(i)*c.channels;auto* q=out+static_cast<std::size_t>(i)*c.channels;const float gray=static_cast<float>(luma(p));for(int ch=0;ch<3;++ch)q[ch]=blend(gray,p[ch],c.factor);for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}
}
void enhance_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const EnhanceContrastConfig& c,cudaStream_t){
  check(in,out,c.width,c.height,c.channels,c.factor,false);const int n=c.width*c.height;long long sum=0;for(int i=0;i<n;++i){const auto* p=in+static_cast<std::size_t>(i)*c.channels;sum+=c.channels>=3?luma(p):p[0];}const float mean=static_cast<float>((sum+n/2)/n);for(int i=0;i<n;++i){const auto* p=in+static_cast<std::size_t>(i)*c.channels;auto* q=out+static_cast<std::size_t>(i)*c.channels;const int limit=c.channels>=3?3:1;for(int ch=0;ch<limit;++ch)q[ch]=blend(mean,p[ch],c.factor);for(int ch=limit;ch<c.channels;++ch)q[ch]=p[ch];}
}
void enhance_brightness_u8(const std::uint8_t* in,std::uint8_t* out,const EnhanceBrightnessConfig& c,cudaStream_t){
  check(in,out,c.width,c.height,c.channels,c.factor,false);for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+static_cast<std::size_t>(i)*c.channels;auto* q=out+static_cast<std::size_t>(i)*c.channels;const int limit=c.channels>=3?3:1;for(int ch=0;ch<limit;++ch)q[ch]=blend(0,p[ch],c.factor);for(int ch=limit;ch<c.channels;++ch)q[ch]=p[ch];}
}
void enhance_sharpness_u8(const std::uint8_t* in,std::uint8_t* out,const EnhanceSharpnessConfig& c,cudaStream_t){
  check(in,out,c.width,c.height,c.channels,c.factor,false);const int kernel[9]={1,1,1,1,5,1,1,1,1};const int n=c.width*c.height;for(int i=0;i<n;++i){const int y=i/c.width,x=i%c.width;const auto* p=in+static_cast<std::size_t>(i)*c.channels;auto* q=out+static_cast<std::size_t>(i)*c.channels;const int limit=c.channels>=3?3:1;for(int ch=0;ch<limit;++ch){float smooth=static_cast<float>(p[ch]);if(x>0&&y>0&&x+1<c.width&&y+1<c.height){int sum=0;for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)sum+=kernel[(dy+1)*3+dx+1]*in[(static_cast<std::size_t>(y+dy)*c.width+x+dx)*c.channels+ch];smooth=static_cast<float>((sum+6)/13);}q[ch]=blend(smooth,p[ch],c.factor);}for(int ch=limit;ch<c.channels;++ch)q[ch]=p[ch];}
}
void filter_blur_u8(const std::uint8_t* in,std::uint8_t* out,const FilterBlurConfig& c,cudaStream_t){static const int k[25]={1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1};filter_impl(in,out,c.width,c.height,c.channels,k,5,16);}
void filter_smooth_u8(const std::uint8_t* in,std::uint8_t* out,const FilterSmoothConfig& c,cudaStream_t){static const int k[9]={1,1,1,1,5,1,1,1,1};filter_impl(in,out,c.width,c.height,c.channels,k,3,13);}
void filter_smooth_more_u8(const std::uint8_t* in,std::uint8_t* out,const FilterSmoothMoreConfig& c,cudaStream_t){static const int k[25]={1,1,1,1,1,1,5,5,5,1,1,5,44,5,1,1,5,5,5,1,1,1,1,1,1};filter_impl(in,out,c.width,c.height,c.channels,k,5,100,0,2);}
void filter_edge_enhance_u8(const std::uint8_t* in,std::uint8_t* out,const FilterEdgeEnhanceConfig& c,cudaStream_t){static const int k[9]={-1,-1,-1,-1,10,-1,-1,-1,-1};filter_impl(in,out,c.width,c.height,c.channels,k,3,2);}
void filter_edge_enhance_more_u8(const std::uint8_t* in,std::uint8_t* out,const FilterEdgeEnhanceMoreConfig& c,cudaStream_t){static const int k[9]={-1,-1,-1,-1,9,-1,-1,-1,-1};filter_impl(in,out,c.width,c.height,c.channels,k,3,1);}
void filter_find_edges_u8(const std::uint8_t* in,std::uint8_t* out,const FilterFindEdgesConfig& c,cudaStream_t){static const int k[9]={-1,-1,-1,-1,8,-1,-1,-1,-1};filter_impl(in,out,c.width,c.height,c.channels,k,3,1);}
void filter_contour_u8(const std::uint8_t* in,std::uint8_t* out,const FilterContourConfig& c,cudaStream_t){static const int k[9]={-1,-1,-1,-1,8,-1,-1,-1,-1};filter_impl(in,out,c.width,c.height,c.channels,k,3,1,255);}
void filter_emboss_u8(const std::uint8_t* in,std::uint8_t* out,const FilterEmbossConfig& c,cudaStream_t){static const int k[9]={0,0,0,0,1,0,-1,0,0};filter_impl(in,out,c.width,c.height,c.channels,k,3,1,128);}
void filter_sharpen_u8(const std::uint8_t* in,std::uint8_t* out,const FilterSharpenConfig& c,cudaStream_t){static const int k[9]={-2,-2,-2,-2,32,-2,-2,-2,-2};filter_impl(in,out,c.width,c.height,c.channels,k,3,16);}
void filter_detail_u8(const std::uint8_t* in,std::uint8_t* out,const FilterDetailConfig& c,cudaStream_t){static const int k[9]={0,-1,0,-1,10,-1,0,-1,0};filter_impl(in,out,c.width,c.height,c.channels,k,3,6);}
}
