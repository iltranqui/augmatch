#include "augmatch/derivative.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace augmatch {
namespace {
float sample(const std::uint8_t* in,int x,int y,int ch,int w,int h,int c,BorderPolicy border,std::uint8_t fill){if(x<0||x>=w||y<0||y>=h){if(border==BorderPolicy::Constant)return fill;x=std::max(0,std::min(w-1,x));y=std::max(0,std::min(h-1,y));}return in[(static_cast<std::size_t>(y)*w+x)*c+ch];}
void apply(const std::uint8_t* in,std::uint8_t* out,const DerivativeConfig& c,const int* kx,const int* ky,bool magnitude,cudaStream_t){
 if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.scale<0) throw std::invalid_argument("invalid derivative configuration");
 for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) for(int ch=0;ch<c.channels;++ch){float gx=0,gy=0;
  for(int j=-1;j<=1;++j) for(int i=-1;i<=1;++i){float v=sample(in,x+i,y+j,ch,c.width,c.height,c.channels,c.border,c.border_value);gx+=v*kx[(j+1)*3+i+1];gy+=v*ky[(j+1)*3+i+1];}
  float v=magnitude?std::sqrt(gx*gx+gy*gy):std::fabs(gx+gy);out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(std::max(0.f,std::min(255.f,v*c.scale))));
 }
}
const int zero[9]={0,0,0,0,0,0,0,0,0};
}
void sobel_x_u8(const std::uint8_t* i,std::uint8_t* o,const DerivativeConfig& c,cudaStream_t s){const int x[9]={-1,0,1,-2,0,2,-1,0,1};apply(i,o,c,x,zero,false,s);}
void sobel_y_u8(const std::uint8_t* i,std::uint8_t* o,const DerivativeConfig& c,cudaStream_t s){const int y[9]={-1,-2,-1,0,0,0,1,2,1};apply(i,o,c,zero,y,false,s);}
void scharr_x_u8(const std::uint8_t* i,std::uint8_t* o,const DerivativeConfig& c,cudaStream_t s){const int x[9]={-3,0,3,-10,0,10,-3,0,3};apply(i,o,c,x,zero,false,s);}
void scharr_y_u8(const std::uint8_t* i,std::uint8_t* o,const DerivativeConfig& c,cudaStream_t s){const int y[9]={-3,-10,-3,0,0,0,3,10,3};apply(i,o,c,zero,y,false,s);}
void laplacian_u8(const std::uint8_t* i,std::uint8_t* o,const DerivativeConfig& c,cudaStream_t s){const int l[9]={0,1,0,1,-4,1,0,1,0};apply(i,o,c,l,zero,false,s);}
void gradient_magnitude_u8(const std::uint8_t* i,std::uint8_t* o,const DerivativeConfig& c,cudaStream_t s){const int x[9]={-1,0,1,-2,0,2,-1,0,1};const int y[9]={-1,-2,-1,0,0,0,1,2,1};apply(i,o,c,x,y,true,s);}
void local_variance_u8(const std::uint8_t* in,std::uint8_t* out,const LocalStatsConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.radius<1||c.scale<0)throw std::invalid_argument("invalid local variance configuration");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0,sq=0;int n=0;for(int j=-c.radius;j<=c.radius;++j)for(int i=-c.radius;i<=c.radius;++i){float v=sample(in,x+i,y+j,ch,c.width,c.height,c.channels,c.border,c.border_value);sum+=v;sq+=v*v;++n;}float v=std::max(0.f,sq/n-(sum/n)*(sum/n))*c.scale;out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(std::min(255.f,v)));}}
void local_entropy_u8(const std::uint8_t* in,std::uint8_t* out,const LocalStatsConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.radius<1||c.scale<0)throw std::invalid_argument("invalid local entropy configuration");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){int hist[256]={};int n=0;for(int j=-c.radius;j<=c.radius;++j)for(int i=-c.radius;i<=c.radius;++i){int idx=static_cast<int>(sample(in,x+i,y+j,ch,c.width,c.height,c.channels,c.border,c.border_value));++hist[idx];++n;}float e=0;for(int b:hist)if(b){float p=static_cast<float>(b)/n;e-=p*std::log2(p);}out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(std::min(255.f,e*c.scale)));}}
}
