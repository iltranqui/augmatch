#include "augmatch/color/colorspace.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
void valid(int w,int h,int c){if(w<=0||h<=0||c<3) throw std::invalid_argument("colorspace requires positive RGB dimensions");}
void buffers(const std::uint8_t* in,std::uint8_t* out){if(!in||!out) throw std::invalid_argument("null colorspace buffer");}
int bin(float x){return std::max(0,std::min(255,static_cast<int>(std::floor(x+0.5f))));}
int hue_bin(float x){int h=static_cast<int>(std::floor(x+0.5f))%180;return h<0?h+180:h;}
void finite(float x,const char* n){if(!std::isfinite(x)) throw std::invalid_argument(std::string(n)+" must be finite");}
void validate_space(ColorSpace s){if(s!=ColorSpace::RGB&&s!=ColorSpace::HSV&&s!=ColorSpace::LAB) throw std::invalid_argument("unsupported colorspace");}
struct HSV { int h,s,v; };
HSV to_hsv(const std::uint8_t* p){
  const float r=p[0]/255.0f,g=p[1]/255.0f,b=p[2]/255.0f,mx=std::max(r,std::max(g,b)),mn=std::min(r,std::min(g,b)),d=mx-mn;
  float h=0.0f; if(d>1.0e-6f){if(mx==r) h=60.0f*std::fmod((g-b)/d+6.0f,6.0f);else if(mx==g) h=60.0f*((b-r)/d+2.0f);else h=60.0f*((r-g)/d+4.0f);}
  return {hue_bin(h/2.0f),bin(mx<=1.0e-6f?0.0f:d/mx*255.0f),bin(mx*255.0f)};
}
void from_hsv(std::uint8_t* q,int h,int s,int v){
  const float value=v/255.0f,saturation=s/255.0f,sector=(h*2.0f)/60.0f,part=sector-std::floor(sector),pv=value*(1-saturation),qv=value*(1-saturation*part),tv=value*(1-saturation*(1-part));float r,g,b;
  switch(h/30){case 0:r=value;g=tv;b=pv;break;case 1:r=qv;g=value;b=pv;break;case 2:r=pv;g=value;b=tv;break;case 3:r=pv;g=qv;b=value;break;case 4:r=tv;g=pv;b=value;break;default:r=value;g=pv;b=qv;break;}
  q[0]=static_cast<std::uint8_t>(bin(r*255.0f));q[1]=static_cast<std::uint8_t>(bin(g*255.0f));q[2]=static_cast<std::uint8_t>(bin(b*255.0f));
}
struct Lab { float l,a,b; };
float pivot_rgb(float x){return x<=0.04045f?x/12.92f:std::pow((x+0.055f)/1.055f,2.4f);}
float pivot_xyz(float x){return x>0.0088564517f?std::cbrt(x):7.787037f*x+16.0f/116.0f;}
float inv_xyz(float x){const float x3=x*x*x;return x3>0.0088564517f?x3:(x-16.0f/116.0f)/7.787037f;}
float inv_rgb(float x){return x<=0.0031308f?12.92f*x:1.055f*std::pow(std::max(0.0f,x),1.0f/2.4f)-0.055f;}
Lab to_lab(const std::uint8_t* p){
  const float r=pivot_rgb(p[0]/255.0f),g=pivot_rgb(p[1]/255.0f),b=pivot_rgb(p[2]/255.0f);
  const float x=pivot_xyz((0.4124564f*r+0.3575761f*g+0.1804375f*b)/0.95047f),y=pivot_xyz((0.2126729f*r+0.7151522f*g+0.0721750f*b)/1.0f),z=pivot_xyz((0.0193339f*r+0.1191920f*g+0.9503041f*b)/1.08883f);
  return {116.0f*y-16.0f,500.0f*(x-y),200.0f*(y-z)};
}
void from_lab(std::uint8_t* q,float l,float a,float b){
  const float y=(l+16.0f)/116.0f,x=a/500.0f+y,z=y-b/200.0f;
  const float X=0.95047f*inv_xyz(x),Y=inv_xyz(y),Z=1.08883f*inv_xyz(z);
  const float r=inv_rgb( 3.2404542f*X-1.5371385f*Y-0.4985314f*Z),g=inv_rgb(-0.9692660f*X+1.8760108f*Y+0.0415560f*Z),bb=inv_rgb(0.0556434f*X-0.2040259f*Y+1.0572252f*Z);
  q[0]=static_cast<std::uint8_t>(bin(std::max(0.0f,std::min(1.0f,r))*255.0f));q[1]=static_cast<std::uint8_t>(bin(std::max(0.0f,std::min(1.0f,g))*255.0f));q[2]=static_cast<std::uint8_t>(bin(std::max(0.0f,std::min(1.0f,bb))*255.0f));
}
void validate_with(const WithColorspaceConfig& c){valid(c.width,c.height,c.channels);validate_space(c.colorspace);const float v[]={c.multiplier0,c.multiplier1,c.multiplier2,c.addend0,c.addend1,c.addend2};for(float x:v)finite(x,"colorspace parameter");if(c.multiplier0<0||c.multiplier1<0||c.multiplier2<0)throw std::invalid_argument("colorspace multipliers must be nonnegative");}
void validate_brightness(const WithBrightnessChannelsConfig& c){valid(c.width,c.height,c.channels);if(c.colorspace!=ColorSpace::RGB&&c.colorspace!=ColorSpace::HSV)throw std::invalid_argument("brightness channels supports RGB or HSV");finite(c.multiplier,"brightness multiplier");finite(c.addend,"brightness addend");if(c.multiplier<0)throw std::invalid_argument("brightness multiplier must be nonnegative");const int ids[]={c.channel0,c.channel1,c.channel2};for(int id:ids)if(id!=-1&&(id<0||id>2||(c.colorspace==ColorSpace::HSV&&id!=2)))throw std::invalid_argument("invalid brightness channel");}
}
void with_colorspace_u8(const std::uint8_t* in,std::uint8_t* out,const WithColorspaceConfig& c,cudaStream_t){validate_with(c);buffers(in,out);for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+static_cast<std::size_t>(i)*c.channels;auto* q=out+static_cast<std::size_t>(i)*c.channels;for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];if(c.colorspace==ColorSpace::RGB){q[0]=bin(p[0]*c.multiplier0+c.addend0);q[1]=bin(p[1]*c.multiplier1+c.addend1);q[2]=bin(p[2]*c.multiplier2+c.addend2);}else if(c.colorspace==ColorSpace::HSV){HSV x=to_hsv(p);x.h=hue_bin(x.h*c.multiplier0+c.addend0);x.s=bin(x.s*c.multiplier1+c.addend1);x.v=bin(x.v*c.multiplier2+c.addend2);from_hsv(q,x.h,x.s,x.v);}else{Lab x=to_lab(p);const float l=std::max(0.0f,std::min(255.0f,(x.l*2.55f)*c.multiplier0+c.addend0)),a=std::max(0.0f,std::min(255.0f,(x.a+128.0f)*c.multiplier1+c.addend1)),b=std::max(0.0f,std::min(255.0f,(x.b+128.0f)*c.multiplier2+c.addend2));from_lab(q,l/2.55f,a-128.0f,b-128.0f);}}}
void with_brightness_channels_u8(const std::uint8_t* in,std::uint8_t* out,const WithBrightnessChannelsConfig& c,cudaStream_t){validate_brightness(c);buffers(in,out);for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+static_cast<std::size_t>(i)*c.channels;auto* q=out+static_cast<std::size_t>(i)*c.channels;for(int ch=0;ch<c.channels;++ch)q[ch]=p[ch];const int ids[]={c.channel0,c.channel1,c.channel2};if(c.colorspace==ColorSpace::RGB){for(int k:ids)if(k>=0)q[k]=bin(p[k]*c.multiplier+c.addend);}else{if(c.channel0==2||c.channel1==2||c.channel2==2){HSV x=to_hsv(p);x.v=bin(x.v*c.multiplier+c.addend);from_hsv(q,x.h,x.s,x.v);for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}}}
}
}
