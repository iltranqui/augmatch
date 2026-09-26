#include "augmatch/catalog/imgcorruptlike.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {
std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
float uniform(std::uint64_t seed, std::uint64_t index) {
  return static_cast<float>(splitmix64(seed + index) >> 40) / 16777216.0f;
}
float normal(std::uint64_t seed, std::uint64_t index) {
  const float u1 = std::max(uniform(seed, index * 2), 1.0e-7f);
  const float u2 = uniform(seed, index * 2 + 1);
  return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853071795864769f * u2);
}
void check_dims(int w,int h,int c) {
  if (w<=0 || h<=0 || c<=0) throw std::invalid_argument("imgcorruptlike dimensions and channels must be positive");
}
void check_io(const std::uint8_t* in,std::uint8_t* out) {
  if (!in || !out) throw std::invalid_argument("imgcorruptlike input and output are required");
}
std::uint8_t byte(float value) {
  return static_cast<std::uint8_t>(std::max(0.0f,std::min(255.0f,std::floor(value+0.5f))));
}
void check_unit(float value,const char* name) {
  if (!std::isfinite(value) || value<0.0f || value>1.0f) throw std::invalid_argument(std::string(name)+" must be finite and in [0,1]");
}
float field_value(const float* field,std::uint64_t seed,std::size_t pixel) {
  return field ? field[pixel] : uniform(seed,static_cast<std::uint64_t>(pixel));
}
void validate_field(const float* field,int w,int h) {
  if (!field) return;
  const std::size_t n=static_cast<std::size_t>(w)*h;
  for (std::size_t i=0;i<n;++i) check_unit(field[i],"imgcorruptlike field value");
}
void hsv_from_rgb(float r,float g,float b,float& h,float& s,float& v) {
  const float mx=std::max(r,std::max(g,b)), mn=std::min(r,std::min(g,b)), d=mx-mn;
  v=mx; s=mx==0.0f?0.0f:d/mx; h=0.0f;
  if (d!=0.0f) { if(mx==r) h=std::fmod((g-b)/d,6.0f); else if(mx==g) h=(b-r)/d+2.0f; else h=(r-g)/d+4.0f; h/=6.0f; if(h<0.0f)h+=1.0f; }
}
void rgb_from_hsv(float h,float s,float v,float& r,float& g,float& b) {
  if(s==0.0f){r=g=b=v;return;} const float q=h*6.0f; const int i=static_cast<int>(std::floor(q)); const float f=q-i;
  const float p=v*(1.0f-s), t=v*(1.0f-s*f), u=v*(1.0f-s*(1.0f-f));
  switch(i%6){case 0:r=v;g=u;b=p;break;case 1:r=t;g=v;b=p;break;case 2:r=p;g=v;b=u;break;case 3:r=p;g=t;b=v;break;case 4:r=u;g=p;b=v;break;default:r=v;g=p;b=t;break;}
}
void validate_fog_common(int w,int h,int c,float a,float b,const float* field,const char* name) {
  check_dims(w,h,c); check_unit(a,(std::string(name)+" density/intensity").c_str()); check_unit(b,(std::string(name)+" opacity").c_str()); validate_field(field,w,h);
}
} // namespace

void speckle_noise_u8(const std::uint8_t* in,std::uint8_t* out,const SpeckleNoiseConfig& c,cudaStream_t) {
  check_dims(c.width,c.height,c.channels); check_io(in,out);
  if(!std::isfinite(c.mean)||!std::isfinite(c.stddev)||c.stddev<0.0f) throw std::invalid_argument("SpeckleNoise mean/stddev must be finite and stddev nonnegative");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;
  for(std::size_t i=0;i<n;++i) out[i]=byte(static_cast<float>(in[i])*(1.0f+c.mean+c.stddev*normal(c.seed,i)));
}
void fog_u8(const std::uint8_t* in,std::uint8_t* out,const FogConfig& c,cudaStream_t) {
  validate_fog_common(c.width,c.height,c.channels,c.density,c.opacity,c.field,"Fog"); check_io(in,out);
  const std::size_t pixels=static_cast<std::size_t>(c.width)*c.height;
  for(std::size_t p=0;p<pixels;++p){const float a=std::min(1.0f,c.density*c.opacity*field_value(c.field,c.seed,p));for(int ch=0;ch<c.channels;++ch)out[p*c.channels+ch]=byte((1.0f-a)*in[p*c.channels+ch]+a*255.0f);}
}
void frost_u8(const std::uint8_t* in,std::uint8_t* out,const FrostConfig& c,cudaStream_t) {
  validate_fog_common(c.width,c.height,c.channels,c.intensity,c.opacity,c.field,"Frost"); check_io(in,out);
  const std::size_t pixels=static_cast<std::size_t>(c.width)*c.height;
  for(std::size_t p=0;p<pixels;++p){const float a=std::min(1.0f,c.intensity*c.opacity*field_value(c.field,c.seed,p));for(int ch=0;ch<c.channels;++ch){const float frost=ch==0?230.0f:(ch==1?240.0f:255.0f);out[p*c.channels+ch]=byte((1.0f-a)*in[p*c.channels+ch]+a*frost);}}
}
void snow_u8(const std::uint8_t* in,std::uint8_t* out,const SnowConfig& c,cudaStream_t) {
  validate_fog_common(c.width,c.height,c.channels,c.density,c.opacity,c.field,"Snow"); check_io(in,out);
  const std::size_t pixels=static_cast<std::size_t>(c.width)*c.height;
  for(std::size_t p=0;p<pixels;++p){const float f=field_value(c.field,c.seed,p);const float a=f>=1.0f-c.density?c.opacity:0.0f;for(int ch=0;ch<c.channels;++ch)out[p*c.channels+ch]=byte((1.0f-a)*in[p*c.channels+ch]+a*255.0f);}
}
void contrast_u8(const std::uint8_t* in,std::uint8_t* out,const ContrastConfig& c,cudaStream_t) {
  check_dims(c.width,c.height,c.channels);check_io(in,out);if(!std::isfinite(c.factor)||c.factor<0.0f)throw std::invalid_argument("Contrast factor must be finite and nonnegative");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;for(std::size_t i=0;i<n;++i)out[i]=byte((static_cast<float>(in[i])-127.5f)*c.factor+127.5f);
}
void brightness_u8(const std::uint8_t* in,std::uint8_t* out,const BrightnessConfig& c,cudaStream_t) {
  check_dims(c.width,c.height,c.channels);check_io(in,out);if(!std::isfinite(c.factor)||c.factor<0.0f)throw std::invalid_argument("Brightness factor must be finite and nonnegative");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;for(std::size_t i=0;i<n;++i)out[i]=byte(static_cast<float>(in[i])*c.factor);
}
void saturate_u8(const std::uint8_t* in,std::uint8_t* out,const SaturateConfig& c,cudaStream_t) {
  check_dims(c.width,c.height,c.channels);check_io(in,out);if(!std::isfinite(c.factor)||c.factor<0.0f)throw std::invalid_argument("Saturate factor must be finite and nonnegative");
  const std::size_t pixels=static_cast<std::size_t>(c.width)*c.height;
  for(std::size_t p=0;p<pixels;++p){const std::uint8_t* s=in+p*c.channels;std::uint8_t* d=out+p*c.channels; if(c.channels<3){for(int ch=0;ch<c.channels;++ch)d[ch]=s[ch];continue;} float h,ss,v,r,g,b;hsv_from_rgb(s[0]/255.0f,s[1]/255.0f,s[2]/255.0f,h,ss,v);rgb_from_hsv(h,std::min(1.0f,ss*c.factor),v,r,g,b);d[0]=byte(r*255.0f);d[1]=byte(g*255.0f);d[2]=byte(b*255.0f);for(int ch=3;ch<c.channels;++ch)d[ch]=s[ch];}
}
void pixelate_u8(const std::uint8_t* in,std::uint8_t* out,const PixelateConfig& c,cudaStream_t) {
  check_dims(c.width,c.height,c.channels);check_io(in,out);if(c.block_size<=0)throw std::invalid_argument("Pixelate block size must be positive");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const int sx=std::min(c.width-1,(x/c.block_size)*c.block_size+c.block_size/2);const int sy=std::min(c.height-1,(y/c.block_size)*c.block_size+c.block_size/2);for(int ch=0;ch<c.channels;++ch)out[(y*c.width+x)*c.channels+ch]=in[(sy*c.width+sx)*c.channels+ch];}
}
} // namespace augmatch
