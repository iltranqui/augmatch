#include "augmatch/color/tone.hpp"
#include "augmatch/color/arithmetic.hpp"
#include "clahe_impl.hpp"
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
void check_channels(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("tone transforms require positive image dimensions and channels");}
void check_rgb(int w,int h,int c){if(w<=0||h<=0||c<3)throw std::invalid_argument("tone transforms require RGB dimensions");}
std::uint8_t sat(float x){return static_cast<std::uint8_t>(std::lround(std::max(0.f,std::min(255.f,x))));}
std::uint64_t splitmix64(std::uint64_t value){value+=0x9e3779b97f4a7c15ULL;value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;value=(value^(value>>27))*0x94d049bb133111ebULL;return value^(value>>31);}
int curve_value(int channel,int input,std::uint64_t seed){
  const int segment=input==255?3:input/64;
  const int x0=segment==0?0:segment==1?64:segment==2?128:192;
  const int x1=segment==0?64:segment==1?128:segment==2?192:255;
  int y[5]={0,0,0,0,255};
  for(int point=1;point<4;++point){
    const int x=point*64;
    const int offset=static_cast<int>(splitmix64(seed+static_cast<std::uint64_t>(channel*4+point))%65ULL)-32;
    y[point]=std::max(y[point-1],std::min(255,x+offset));
  }
  const int distance=input-x0,span=x1-x0;
  return (y[segment]*(span-distance)+y[segment+1]*distance+span/2)/span;
}
}
void make_random_tone_curve_lut(std::uint8_t* lut,int channels,std::uint64_t seed){
  if(!lut||channels<=0)throw std::invalid_argument("random tone curve LUT requires positive channels and non-null storage");
  for(int channel=0;channel<channels;++channel)for(int input=0;input<256;++input)
    lut[static_cast<std::size_t>(channel)*256+input]=static_cast<std::uint8_t>(curve_value(channel,input,seed));
}
void random_tone_curve_u8(const std::uint8_t* in,std::uint8_t* out,const RandomToneCurveConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid random tone curve configuration");
  std::uint8_t generated[256];
  for(int channel=0;channel<c.channels;++channel){
    const std::uint8_t* table=c.lut?c.lut+static_cast<std::size_t>(channel)*256:generated;
    if(!c.lut)for(int input=0;input<256;++input)generated[input]=static_cast<std::uint8_t>(curve_value(channel,input,c.seed));
    for(int pixel=0,n=c.width*c.height;pixel<n;++pixel)out[static_cast<std::size_t>(pixel)*c.channels+channel]=table[in[static_cast<std::size_t>(pixel)*c.channels+channel]];
  }
}
void clahe_u8(const std::uint8_t* in,std::uint8_t* out,const CLAHEConfig& c,cudaStream_t){detail::clahe_u8_host(in,out,c.width,c.height,c.channels,c.clip_limit,c.tiles_x,c.tiles_y);}
void all_channels_clahe_u8(const std::uint8_t* in,std::uint8_t* out,const AllChannelsCLAHEConfig& c,cudaStream_t stream){clahe_u8(in,out,{c.width,c.height,c.channels,c.clip_limit,c.tiles_x,c.tiles_y},stream);}
void sepia_u8(const std::uint8_t* in,std::uint8_t* out,const ToneConfig& c,cudaStream_t){check_rgb(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null sepia buffer");for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;q[0]=sat(.393f*p[0]+.769f*p[1]+.189f*p[2]);q[1]=sat(.349f*p[0]+.686f*p[1]+.168f*p[2]);q[2]=sat(.272f*p[0]+.534f*p[1]+.131f*p[2]);for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}}
void autocontrast_u8(const std::uint8_t* in,std::uint8_t* out,const ToneConfig& c,cudaStream_t){check_channels(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null autocontrast buffer");for(int ch=0;ch<c.channels;++ch){int lo=255,hi=0;for(int i=0,n=c.width*c.height;i<n;++i){lo=std::min(lo,(int)in[i*c.channels+ch]);hi=std::max(hi,(int)in[i*c.channels+ch]);}for(int i=0,n=c.width*c.height;i<n;++i)out[i*c.channels+ch]=hi==lo?in[i*c.channels+ch]:sat((in[i*c.channels+ch]-lo)*255.f/(hi-lo));}}
void equalize_u8(const std::uint8_t* in,std::uint8_t* out,const ToneConfig& c,cudaStream_t){check_channels(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null equalize buffer");const int n=c.width*c.height;for(int ch=0;ch<c.channels;++ch){int hist[256]={};for(int i=0;i<n;++i)++hist[in[i*c.channels+ch]];int cdf=0,cdf_min=0;while(cdf_min<256&&hist[cdf_min]==0)++cdf_min;for(int v=0;v<256;++v){cdf+=hist[v];int mapped=(n==hist[cdf_min]?v:static_cast<int>(std::lround((cdf-hist[cdf_min])*255.0/(n-hist[cdf_min]))));for(int i=0;i<n;++i)if(in[i*c.channels+ch]==v)out[i*c.channels+ch]=static_cast<std::uint8_t>(std::max(0,std::min(255,mapped)));}}}
void all_channels_histogram_equalization_u8(const std::uint8_t* in,std::uint8_t* out,const AllChannelsHistogramEqualizationConfig& c,cudaStream_t stream){equalize_u8(in,out,{c.width,c.height,c.channels},stream);}
namespace {
void check_sigmoid(const std::uint8_t* in,const std::uint8_t* out,const SigmoidContrastConfig& c){
  if(c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid contrast dimensions");
  if(!in||!out||!std::isfinite(c.cutoff)||c.cutoff<0.0f||c.cutoff>1.0f||!std::isfinite(c.gain)||c.gain<0.0f)
    throw std::invalid_argument("invalid sigmoid contrast configuration");
}
void check_log(const std::uint8_t* in,const std::uint8_t* out,const LogContrastConfig& c){
  if(c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid contrast dimensions");
  if(!in||!out||!std::isfinite(c.gain)||c.gain<0.0f||!std::isfinite(c.base)||c.base<=1.0f)
    throw std::invalid_argument("invalid log contrast configuration");
}
std::uint8_t contrast_round(float value){return static_cast<std::uint8_t>(std::floor(std::max(0.0f,std::min(255.0f,value))+0.5f));}
}
void sigmoid_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const SigmoidContrastConfig& c,cudaStream_t){
  check_sigmoid(in,out,c);
  for(int i=0,n=c.width*c.height*c.channels;i<n;++i){
    const float x=static_cast<float>(in[i])/255.0f;
    const float y=1.0f/(1.0f+std::exp(c.gain*(c.cutoff-x)));
    out[i]=contrast_round(y*255.0f);
  }
}
void log_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const LogContrastConfig& c,cudaStream_t){
  check_log(in,out,c);
  const float scale=std::pow(c.base,c.gain)-1.0f;
  const float denominator=std::log(c.base);
  for(int i=0,n=c.width*c.height*c.channels;i<n;++i){
    const float x=static_cast<float>(in[i])/255.0f;
    const float y=std::log1p(x*scale)/denominator;
    out[i]=contrast_round(y*255.0f);
  }
}
namespace {
void check_variation(const std::uint8_t* in,const std::uint8_t* out,int w,int h,int channels){
  if(!in||!out||w<=0||h<=0||channels<=0)throw std::invalid_argument("invalid tone variation dimensions or buffers");
}
void check_finite(float value,const char* name){if(!std::isfinite(value))throw std::invalid_argument(std::string("invalid ")+name);}
float clamp01(float x){return std::max(0.0f,std::min(1.0f,x));}
std::uint8_t variation_round(float y){return static_cast<std::uint8_t>(std::floor(clamp01(y)*255.0f+0.5f));}
}
void gamma_variation_u8(const std::uint8_t* in,std::uint8_t* out,const GammaVariationConfig& c,cudaStream_t){
  check_variation(in,out,c.width,c.height,c.channels);check_finite(c.gamma,"gamma");if(c.gamma<=0.0f)throw std::invalid_argument("gamma must be positive");
  for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=variation_round(std::pow(in[i]/255.0f,c.gamma));
}
void tone_curve_variation_u8(const std::uint8_t* in,std::uint8_t* out,const ToneCurveVariationConfig& c,cudaStream_t){
  check_variation(in,out,c.width,c.height,c.channels);if(!c.lut)throw std::invalid_argument("tone curve LUT must not be null");
  for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=c.lut[(i%c.channels)*256+in[i]];
}
void s_curve_contrast_variation_u8(const std::uint8_t* in,std::uint8_t* out,const SCurveContrastVariationConfig& c,cudaStream_t){
  check_variation(in,out,c.width,c.height,c.channels);check_finite(c.amount,"S-curve amount");if(c.amount<-1.0f||c.amount>1.0f)throw std::invalid_argument("S-curve amount must be in [-1,1]");
  for(int i=0,n=c.width*c.height*c.channels;i<n;++i){const float x=in[i]/255.0f;out[i]=variation_round(x+c.amount*4.0f*x*(1.0f-x)*(2.0f*x-1.0f));}
}
void highlight_rolloff_variation_u8(const std::uint8_t* in,std::uint8_t* out,const HighlightRolloffVariationConfig& c,cudaStream_t){
  check_variation(in,out,c.width,c.height,c.channels);check_finite(c.threshold,"highlight threshold");check_finite(c.strength,"highlight strength");if(c.threshold<0.0f||c.threshold>1.0f||c.strength<0.0f)throw std::invalid_argument("invalid highlight roll-off configuration");
  for(int i=0,n=c.width*c.height*c.channels;i<n;++i){const float x=in[i]/255.0f;const float d=std::max(0.0f,x-c.threshold);const float den=std::max(1e-12f,1.0f-c.threshold);const float y=x<=c.threshold?x:c.threshold+d/(1.0f+c.strength*d/den);out[i]=variation_round(y);}
}
void shadow_lift_u8(const std::uint8_t* in,std::uint8_t* out,const ShadowLiftConfig& c,cudaStream_t){
  check_variation(in,out,c.width,c.height,c.channels);check_finite(c.amount,"shadow lift amount");check_finite(c.threshold,"shadow threshold");if(c.amount<0.0f||c.amount>1.0f||c.threshold<=0.0f||c.threshold>1.0f)throw std::invalid_argument("invalid shadow lift configuration");
  for(int i=0,n=c.width*c.height*c.channels;i<n;++i){const float x=in[i]/255.0f;const float weight=clamp01((c.threshold-x)/c.threshold);out[i]=variation_round(x+c.amount*weight*(1.0f-x));}
}
void shadow_crush_u8(const std::uint8_t* in,std::uint8_t* out,const ShadowCrushConfig& c,cudaStream_t){
  check_variation(in,out,c.width,c.height,c.channels);check_finite(c.amount,"shadow crush amount");check_finite(c.threshold,"shadow threshold");if(c.amount<0.0f||c.amount>1.0f||c.threshold<=0.0f||c.threshold>1.0f)throw std::invalid_argument("invalid shadow crush configuration");
  for(int i=0,n=c.width*c.height*c.channels;i<n;++i){const float x=in[i]/255.0f;const float weight=clamp01((c.threshold-x)/c.threshold);out[i]=variation_round(x*(1.0f-c.amount*weight));}
}
void posterization_u8(const std::uint8_t* in,std::uint8_t* out,const PosterizationConfig& c,cudaStream_t s){
  posterize_u8(in,out,{c.width,c.height,c.channels,c.bits},s);
}
void low_bit_depth_banding_u8(const std::uint8_t* in,std::uint8_t* out,const LowBitDepthBandingConfig& c,cudaStream_t){posterization_u8(in,out,{c.width,c.height,c.channels,c.bits},nullptr);}
}
