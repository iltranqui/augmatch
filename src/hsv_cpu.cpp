#include "augmatch/color/hsv.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
namespace augmatch { namespace { void check(int w,int h,int c){if(w<=0||h<=0||c<3)throw std::invalid_argument("HSV requires RGB dimensions");} unsigned char sat(float v){return static_cast<unsigned char>(std::lround(std::max(0.0f,std::min(255.0f,v))));} }
void hsv_shift_u8(const std::uint8_t* in,std::uint8_t* out,const HSVShiftConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null HSV buffer");for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;float r=p[0]/255.f,g=p[1]/255.f,b=p[2]/255.f,mx=std::max(r,std::max(g,b)),mn=std::min(r,std::min(g,b)),d=mx-mn,h=0;if(d>1e-6f){if(mx==r)h=60.f*std::fmod((g-b)/d+6.f,6.f);else if(mx==g)h=60.f*((b-r)/d+2.f);else h=60.f*((r-g)/d+4.f);}float s=mx<=1e-6f?0.f:d/mx,v=mx;int hb=(static_cast<int>(std::lround(h/2.f))+c.hue_shift)%180;if(hb<0)hb+=180;float sb=std::max(0.f,std::min(255.f,s*255.f+c.saturation_shift))/255.f,vb=std::max(0.f,std::min(255.f,v*255.f+c.value_shift))/255.f;float hf=hb*2.f,sector=hf/60.f,ff=sector-std::floor(sector),pp=vb*(1-sb),qq=vb*(1-sb*ff),tt=vb*(1-sb*(1-ff)),rr=0,gg=0,bb=0;int si=static_cast<int>(std::floor(sector))%6;if(si==0){rr=vb;gg=tt;bb=pp;}else if(si==1){rr=qq;gg=vb;bb=pp;}else if(si==2){rr=pp;gg=vb;bb=tt;}else if(si==3){rr=pp;gg=qq;bb=vb;}else if(si==4){rr=tt;gg=pp;bb=vb;}else{rr=vb;gg=pp;bb=qq;}q[0]=sat(rr*255);q[1]=sat(gg*255);q[2]=sat(bb*255);for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}}
namespace {
enum class HsvBatchOp { MultiplyBrightness, AddBrightness, MultiplyAddBrightness, SetHueSaturation, MultiplyHueSaturation, MultiplyHue, MultiplySaturation, RemoveSaturation, AddHueSaturation, AddHue, AddSaturation };
int half_up(float value) { return static_cast<int>(std::floor(value + 0.5f)); }
int clip_bin(float value) { return std::max(0, std::min(255, half_up(value))); }
int wrap_hue(float value) { int result=half_up(value)%180; return result<0?result+180:result; }
void rgb_to_opencv_hsv(const std::uint8_t* p, int& hue, int& saturation, int& value) {
  const int r=p[0],g=p[1],b=p[2];
  value=std::max(r,std::max(g,b));
  const int minimum=std::min(r,std::min(g,b)), delta=value-minimum;
  const int saturation_div=value ? static_cast<int>(std::lrint((255 << 12)/static_cast<double>(value))) : 0;
  saturation=(delta*saturation_div+(1 << 11)) >> 12;
  int numerator;
  if(value==r) numerator=g-b;
  else if(value==g) numerator=b-r+2*delta;
  else numerator=r-g+4*delta;
  const int hue_div=delta ? static_cast<int>(std::lrint((180 << 12)/(6.0*delta))) : 0;
  hue=(numerator*hue_div+(1 << 11)) >> 12;
  if(hue<0) hue+=180;
}
void hsv_batch(const std::uint8_t* in,std::uint8_t* out,int width,int height,int channels,HsvBatchOp op,float a,float b) {
  check(width,height,channels); if(!in||!out) throw std::invalid_argument("null HSV buffer");
  for(int i=0,n=width*height;i<n;++i) {
    const auto* p=in+static_cast<std::size_t>(i)*channels; auto* q=out+static_cast<std::size_t>(i)*channels;
    int h,s,v; rgb_to_opencv_hsv(p,h,s,v);
    switch(op) { case HsvBatchOp::MultiplyBrightness:v=clip_bin(v*a);break; case HsvBatchOp::AddBrightness:v=clip_bin(v+a);break; case HsvBatchOp::MultiplyAddBrightness:v=clip_bin(v*a+b);break; case HsvBatchOp::SetHueSaturation:h=wrap_hue(a);s=clip_bin(b);break; case HsvBatchOp::MultiplyHueSaturation:h=wrap_hue(h*a);s=clip_bin(s*b);break; case HsvBatchOp::MultiplyHue:h=wrap_hue(h*a);break; case HsvBatchOp::MultiplySaturation:s=clip_bin(s*a);break; case HsvBatchOp::RemoveSaturation:s=0;break; case HsvBatchOp::AddHueSaturation:h=wrap_hue(h+a);s=clip_bin(s+b);break; case HsvBatchOp::AddHue:h=wrap_hue(h+a);break; case HsvBatchOp::AddSaturation:s=clip_bin(s+a);break; }
    const float value=v/255.0f,saturation=s/255.0f,sector=(h*2.0f)/60.0f,part=sector-std::floor(sector),pv=value*(1.0f-saturation),qv=value*(1.0f-saturation*part),tv=value*(1.0f-saturation*(1.0f-part)); float rr=0.0f,gg=0.0f,bb=0.0f;
    switch(h/30) { case 0:rr=value;gg=tv;bb=pv;break; case 1:rr=qv;gg=value;bb=pv;break; case 2:rr=pv;gg=value;bb=tv;break; case 3:rr=pv;gg=qv;bb=value;break; case 4:rr=tv;gg=pv;bb=value;break; default:rr=value;gg=pv;bb=qv;break; }
    q[0]=static_cast<std::uint8_t>(clip_bin(rr*255.0f)); q[1]=static_cast<std::uint8_t>(clip_bin(gg*255.0f)); q[2]=static_cast<std::uint8_t>(clip_bin(bb*255.0f)); for(int ch=3;ch<channels;++ch) q[ch]=p[ch];
  }
}
void validate_factor(float value,const char* name) { if(!std::isfinite(value)||value<0.0f) throw std::invalid_argument(std::string(name)+" must be finite and nonnegative"); }
void validate_value(float value,const char* name) { if(!std::isfinite(value)) throw std::invalid_argument(std::string(name)+" must be finite"); }
}
void multiply_brightness_u8(const std::uint8_t* i,std::uint8_t* o,const MultiplyBrightnessConfig& c,cudaStream_t){validate_factor(c.multiplier,"brightness multiplier");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::MultiplyBrightness,c.multiplier,0);}
void add_to_brightness_u8(const std::uint8_t* i,std::uint8_t* o,const AddToBrightnessConfig& c,cudaStream_t){validate_value(c.amount,"brightness amount");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::AddBrightness,c.amount,0);}
void multiply_and_add_to_brightness_u8(const std::uint8_t* i,std::uint8_t* o,const MultiplyAndAddToBrightnessConfig& c,cudaStream_t){validate_factor(c.multiplier,"brightness multiplier");validate_value(c.addend,"brightness addend");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::MultiplyAddBrightness,c.multiplier,c.addend);}
void with_hue_and_saturation_u8(const std::uint8_t* i,std::uint8_t* o,const WithHueAndSaturationConfig& c,cudaStream_t){validate_value(c.hue,"hue");validate_value(c.saturation,"saturation");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::SetHueSaturation,c.hue,c.saturation);}
void multiply_hue_and_saturation_u8(const std::uint8_t* i,std::uint8_t* o,const MultiplyHueAndSaturationConfig& c,cudaStream_t){validate_factor(c.hue_multiplier,"hue multiplier");validate_factor(c.saturation_multiplier,"saturation multiplier");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::MultiplyHueSaturation,c.hue_multiplier,c.saturation_multiplier);}
void multiply_hue_u8(const std::uint8_t* i,std::uint8_t* o,const MultiplyHueConfig& c,cudaStream_t){validate_factor(c.multiplier,"hue multiplier");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::MultiplyHue,c.multiplier,0);}
void multiply_saturation_u8(const std::uint8_t* i,std::uint8_t* o,const MultiplySaturationConfig& c,cudaStream_t){validate_factor(c.multiplier,"saturation multiplier");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::MultiplySaturation,c.multiplier,0);}
void remove_saturation_u8(const std::uint8_t* i,std::uint8_t* o,const RemoveSaturationConfig& c,cudaStream_t){hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::RemoveSaturation,0,0);}
void add_to_hue_and_saturation_u8(const std::uint8_t* i,std::uint8_t* o,const AddToHueAndSaturationConfig& c,cudaStream_t){validate_value(c.hue_amount,"hue amount");validate_value(c.saturation_amount,"saturation amount");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::AddHueSaturation,c.hue_amount,c.saturation_amount);}
void add_to_hue_u8(const std::uint8_t* i,std::uint8_t* o,const AddToHueConfig& c,cudaStream_t){validate_value(c.amount,"hue amount");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::AddHue,c.amount,0);}
void add_to_saturation_u8(const std::uint8_t* i,std::uint8_t* o,const AddToSaturationConfig& c,cudaStream_t){validate_value(c.amount,"saturation amount");hsv_batch(i,o,c.width,c.height,c.channels,HsvBatchOp::AddSaturation,c.amount,0);}
}
