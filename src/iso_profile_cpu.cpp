#include "augmatch/sensor/iso_profile.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
namespace augmatch { namespace {
void valid(const IsoNoiseApplicationConfig& c) {
  if (c.width<=0||c.height<=0||c.channels<=0||!std::isfinite(c.iso)||c.iso<=0||!std::isfinite(c.shot_scale)||c.shot_scale<=0||!std::isfinite(c.read_noise_stddev)||c.read_noise_stddev<0||!std::isfinite(c.fpn_stddev)||c.fpn_stddev<0||!std::isfinite(c.clip_min)||!std::isfinite(c.clip_max)||c.clip_max<c.clip_min||c.profile.points==nullptr||c.profile.point_count==0) throw std::invalid_argument("invalid ISO noise profile configuration");
  float previous=0;
  for(std::size_t i=0;i<c.profile.point_count;++i){const auto& p=c.profile.points[i];if(!std::isfinite(p.iso)||p.iso<=0|| (i&&p.iso<=previous)||!std::isfinite(p.gain)||p.gain<0||!std::isfinite(p.shot_noise_scale)||p.shot_noise_scale<0||!std::isfinite(p.read_noise_scale)||p.read_noise_scale<0||!std::isfinite(p.fpn_scale)||p.fpn_scale<0||!std::isfinite(p.black_level)||!std::isfinite(p.saturation_level)||p.black_level<0||p.black_level>1||p.saturation_level<0||p.saturation_level>1||p.black_level>p.saturation_level) throw std::invalid_argument("invalid ISO noise profile point");previous=p.iso;}
}
std::uint64_t mix(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
float uni(std::uint64_t x){return static_cast<float>((mix(x)>>11)*(1.0/9007199254740992.0));}
float normal(std::uint64_t x){const float a=std::max(uni(x),std::numeric_limits<float>::min());return std::sqrt(-2.0f*std::log(a))*std::cos(6.28318530718f*uni(x^0xd1b54a32d192ed03ULL));}
std::uint64_t key(std::uint64_t s,int x,int y,int ch,std::uint64_t tag){return s^static_cast<std::uint64_t>(x)*0x632be59bd9b4e019ULL^static_cast<std::uint64_t>(y)*0x8cb92baa2f2f6f7dULL^static_cast<std::uint64_t>(ch)*0x9e3779b97f4a7c15ULL^tag;}
float poisson(float lambda,std::uint64_t seed){if(lambda<=0)return 0; if(lambda<64){const float u=uni(seed^0x53484f545f554e49ULL);float p=std::exp(-lambda),cdf=p;int n=0;while(u>cdf&&n<512){++n;p*=lambda/n;cdf+=p;}return static_cast<float>(n);}return std::floor(std::max(0.0f,lambda+std::sqrt(lambda)*normal(seed^0x53484f545f4e4f52ULL))+0.5f);}
void check_buffers(const void* i,void* o){if(!i||!o)throw std::invalid_argument("null ISO noise buffer");}
}}
namespace augmatch {
void iso_shot_noise_u8(const std::uint8_t* in,std::uint8_t* out,const IsoNoiseApplicationConfig& c,cudaStream_t){valid(c);check_buffers(in,out);const float gain=iso_to_gain(c.iso,c.profile),scale=c.shot_scale*iso_shot_noise_scale(c.iso,c.profile);if(!std::isfinite(gain)||gain<0||!std::isfinite(scale)||scale<=0)throw std::invalid_argument("invalid ISO shot-noise profile values");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;const float lambda=(in[i]/255.0f)*gain*scale;const float v=poisson(lambda,key(c.seed,x,y,ch,0x49534f5f53484f54ULL))/scale;out[i]=static_cast<std::uint8_t>(std::floor(std::max(0.0f,std::min(1.0f,v))*255.0f+0.5f));}}
void iso_read_noise_f32(const float* in,float* out,const IsoNoiseApplicationConfig& c,cudaStream_t){valid(c);check_buffers(in,out);const float sigma=c.read_noise_stddev*iso_read_noise_scale(c.iso,c.profile);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]+sigma*normal(key(c.seed,x,y,ch,0x49534f5f52454144ULL))));}}
void iso_fpn_f32(const float* in,float* out,const IsoNoiseApplicationConfig& c,cudaStream_t){valid(c);check_buffers(in,out);const float sigma=c.fpn_stddev*iso_fpn_scale(c.iso,c.profile);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]+sigma*normal(key(c.seed,x,y,ch,0x49534f5f46504e4eULL))));}}
void iso_black_level_u8(const std::uint8_t* in,std::uint8_t* out,const IsoNoiseApplicationConfig& c,cudaStream_t){valid(c);check_buffers(in,out);const int level=static_cast<int>(std::lround(255.0f*iso_black_level(c.iso,c.profile)));for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=static_cast<std::uint8_t>(std::max<int>(in[i],level));}
void iso_saturation_level_u8(const std::uint8_t* in,std::uint8_t* out,const IsoNoiseApplicationConfig& c,cudaStream_t){valid(c);check_buffers(in,out);const int level=static_cast<int>(std::lround(255.0f*iso_saturation_level(c.iso,c.profile)));for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=static_cast<std::uint8_t>(std::min<int>(in[i],level));}
void iso_signal_levels_u8(const std::uint8_t* in,std::uint8_t* out,const IsoNoiseApplicationConfig& c,cudaStream_t){valid(c);check_buffers(in,out);const int lo=static_cast<int>(std::lround(255.0f*iso_black_level(c.iso,c.profile))),hi=static_cast<int>(std::lround(255.0f*iso_saturation_level(c.iso,c.profile)));for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=static_cast<std::uint8_t>(std::max(lo,std::min(hi,static_cast<int>(in[i]))));}
void exposure_time_dark_current_f32(const float* in,float* out,const ExposureTimeDarkCurrentConfig& c,cudaStream_t){
  IsoNoiseApplicationConfig base{}; base.width=c.width;base.height=c.height;base.channels=c.channels;base.iso=c.iso;base.profile=c.profile;
  base.shot_scale=1.0f; base.read_noise_stddev=0.0f; base.fpn_stddev=0.0f; base.clip_min=c.clip_min;base.clip_max=c.clip_max;base.seed=c.seed;
  valid(base); check_buffers(in,out);
  if(!std::isfinite(c.dark_current_electrons_per_second)||c.dark_current_electrons_per_second<0.0f||!std::isfinite(c.exposure_seconds)||c.exposure_seconds<0.0f||!std::isfinite(c.electrons_per_unit)||c.electrons_per_unit<=0.0f) throw std::invalid_argument("invalid exposure-time dark-current configuration");
  const float gain=iso_to_gain(c.iso,c.profile);
  const float lambda=c.dark_current_electrons_per_second*c.exposure_seconds*gain;
  if(!std::isfinite(lambda)||lambda<0.0f) throw std::invalid_argument("invalid exposure-time dark-current rate");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
    const float dark=poisson(lambda,key(c.seed,x,y,ch,0x4441524b5f54494dULL));
    const float value=in[i]+dark/c.electrons_per_unit;
    out[i]=std::max(c.clip_min,std::min(c.clip_max,value));
  }
}
void temperature_noise_scaling_f32(const float* in,float* out,const TemperatureNoiseScalingConfig& c,cudaStream_t){
  IsoNoiseApplicationConfig base{}; base.width=c.width;base.height=c.height;base.channels=c.channels;base.iso=c.iso;base.profile=c.profile;
  base.shot_scale=1.0f;base.read_noise_stddev=0.0f;base.fpn_stddev=0.0f;base.clip_min=c.clip_min;base.clip_max=c.clip_max;base.seed=c.seed;
  valid(base); check_buffers(in,out);
  if(!std::isfinite(c.noise_stddev_electrons)||c.noise_stddev_electrons<0.0f||!std::isfinite(c.temperature_celsius)||!std::isfinite(c.reference_temperature_celsius)||!std::isfinite(c.temperature_noise_coefficient_per_celsius)||!std::isfinite(c.electrons_per_unit)||c.electrons_per_unit<=0.0f) throw std::invalid_argument("invalid temperature noise-scaling configuration");
  const float thermal=std::exp(c.temperature_noise_coefficient_per_celsius*(c.temperature_celsius-c.reference_temperature_celsius));
  const float sigma=c.noise_stddev_electrons*thermal*iso_to_gain(c.iso,c.profile)*iso_read_noise_scale(c.iso,c.profile)/c.electrons_per_unit;
  if(!std::isfinite(sigma)||sigma<0.0f) throw std::invalid_argument("invalid temperature noise-scaling result");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
    out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]+sigma*normal(key(c.seed,x,y,ch,0x54454d505f4e4f49ULL))));
  }
}
namespace {
float transition_weight(float gain,float low,float high,float width) {
  if (high <= low) return gain >= high ? 1.0f : 0.0f;
  const float t=(gain-low)/(high-low);
  if (width<=0.0f) return std::max(0.0f,std::min(1.0f,t));
  if(width*2.0f>=high-low){const float q=std::max(0.0f,std::min(1.0f,t));return q*q*(3.0f-2.0f*q);}
  const float w=width/(high-low);
  const float q=std::max(0.0f,std::min(1.0f,(t-w)/(1.0f-2.0f*w)));
  return q*q*(3.0f-2.0f*q);
}
void validate_gain_bounds(float low,float high,float width,float hysteresis,float clip_min,float clip_max) {
  if(!std::isfinite(low)||!std::isfinite(high)||!std::isfinite(width)||!std::isfinite(hysteresis)||low<0.0f||high<low||width<0.0f||hysteresis<0.0f||!std::isfinite(clip_min)||!std::isfinite(clip_max)||clip_max<clip_min) throw std::invalid_argument("invalid gain transition thresholds");
}
}
void dual_conversion_gain_f32(const float* in,float* out,const DualConversionGainConfig& c,cudaStream_t){
  IsoNoiseApplicationConfig base{};base.width=c.width;base.height=c.height;base.channels=c.channels;base.iso=c.iso;base.profile=c.profile;base.shot_scale=1.0f;base.clip_min=c.clip_min;base.clip_max=c.clip_max;valid(base);check_buffers(in,out);validate_gain_bounds(c.low_gain_threshold,c.high_gain_threshold,0.0f,0.0f,c.clip_min,c.clip_max);
  if(!std::isfinite(c.low_conversion_gain)||!std::isfinite(c.high_conversion_gain)||c.low_conversion_gain<0.0f||c.high_conversion_gain<0.0f||!std::isfinite(c.low_read_noise_electrons)||!std::isfinite(c.high_read_noise_electrons)||c.low_read_noise_electrons<0.0f||c.high_read_noise_electrons<0.0f||!std::isfinite(c.electrons_per_unit)||c.electrons_per_unit<=0.0f) throw std::invalid_argument("invalid dual-conversion-gain configuration");
  const float w=transition_weight(iso_to_gain(c.iso,c.profile),c.low_gain_threshold,c.high_gain_threshold,0.0f);const float conversion=c.low_conversion_gain+(c.high_conversion_gain-c.low_conversion_gain)*w;const float sigma=(c.low_read_noise_electrons+(c.high_read_noise_electrons-c.low_read_noise_electrons)*w)/c.electrons_per_unit;
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;const float v=in[i]*conversion+sigma*normal(key(c.seed,x,y,ch,0x4443475f4e4f4953ULL));out[i]=std::max(c.clip_min,std::min(c.clip_max,v));}
}
void gain_switch_transition_f32(const float* in,float* out,const GainSwitchTransitionConfig& c,cudaStream_t){
  IsoNoiseApplicationConfig base{};base.width=c.width;base.height=c.height;base.channels=c.channels;base.iso=c.iso;base.profile=c.profile;base.shot_scale=1.0f;base.clip_min=c.clip_min;base.clip_max=c.clip_max;valid(base);check_buffers(in,out);validate_gain_bounds(c.low_gain_threshold,c.high_gain_threshold,c.transition_width,c.hysteresis,c.clip_min,c.clip_max);
  if(!std::isfinite(c.low_signal_gain)||!std::isfinite(c.high_signal_gain)||!std::isfinite(c.transition_strength)||c.low_signal_gain<0.0f||c.high_signal_gain<0.0f||c.transition_strength<0.0f) throw std::invalid_argument("invalid gain-switch transition configuration");
  const float gain=iso_to_gain(c.iso,c.profile),lo=c.low_gain_threshold-c.hysteresis,hi=c.high_gain_threshold+c.hysteresis;float w=transition_weight(gain,lo,hi,c.transition_width);if(c.initial_high_gain&&gain>lo&&gain<c.low_gain_threshold)w=1.0f;if(!c.initial_high_gain&&gain>c.high_gain_threshold&&gain<hi)w=0.0f;
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;const float v=in[i]*(c.low_signal_gain+(c.high_signal_gain-c.low_signal_gain)*w)+c.transition_strength*w*(1.0f-w);out[i]=std::max(c.clip_min,std::min(c.clip_max,v));}
}
}

namespace augmatch { namespace {
std::uint64_t camera_mix(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
float camera_uniform(std::uint64_t x){return static_cast<float>((camera_mix(x)>>11)*(1.0/9007199254740992.0));}
float camera_normal(std::uint64_t x){const float a=std::max(camera_uniform(x),std::numeric_limits<float>::min());return std::sqrt(-2.0f*std::log(a))*std::cos(6.28318530718f*camera_uniform(x^0xd1b54a32d192ed03ULL));}
std::uint64_t camera_key(std::uint64_t s,int x,int y,int ch,std::uint64_t tag){return s^static_cast<std::uint64_t>(x)*0x632be59bd9b4e019ULL^static_cast<std::uint64_t>(y)*0x8cb92baa2f2f6f7dULL^static_cast<std::uint64_t>(ch)*0x9e3779b97f4a7c15ULL^tag;}
float camera_poisson(float lambda,std::uint64_t seed){if(lambda<=0)return 0;if(lambda<64){const float u=camera_uniform(seed^0x53484f545f554e49ULL);float p=std::exp(-lambda),cdf=p;int n=0;while(u>cdf&&n<512){++n;p*=lambda/n;cdf+=p;}return static_cast<float>(n);}return std::floor(std::max(0.0f,lambda+std::sqrt(lambda)*camera_normal(seed^0x53484f545f4e4f52ULL))+0.5f);}
void valid_camera(const CameraProfileLookupConfig& c,const void* in,const void* out){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!std::isfinite(c.iso)||c.iso<=0||!std::isfinite(c.clip_min)||!std::isfinite(c.clip_max)||c.clip_max<c.clip_min||!c.profile.points||c.profile.point_count==0) throw std::invalid_argument("invalid camera profile lookup configuration");
  float previous=0.0f; for(std::size_t i=0;i<c.profile.point_count;++i){const auto& p=c.profile.points[i];if(!std::isfinite(p.iso)||p.iso<=0||(i&&p.iso<=previous)||!std::isfinite(p.gain)||p.gain<0||!std::isfinite(p.shot_noise_scale)||p.shot_noise_scale<0||!std::isfinite(p.read_noise_scale)||p.read_noise_scale<0||!std::isfinite(p.fpn_scale)||p.fpn_scale<0||!std::isfinite(p.shot_scale)||p.shot_scale<0||!std::isfinite(p.read_noise_stddev)||p.read_noise_stddev<0||!std::isfinite(p.fpn_stddev)||p.fpn_stddev<0) throw std::invalid_argument("invalid camera profile lookup point");previous=p.iso;}
}
void valid_calibration(const CalibrationFrameNoiseConfig& c,const void* in,const void* out){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!std::isfinite(c.clip_min)||!std::isfinite(c.clip_max)||c.clip_max<c.clip_min||!std::isfinite(c.calibration.shot_noise_scale)||c.calibration.shot_noise_scale<0||!std::isfinite(c.calibration.read_noise_stddev)||c.calibration.read_noise_stddev<0||!std::isfinite(c.calibration.fpn_stddev)||c.calibration.fpn_stddev<0) throw std::invalid_argument("invalid calibration-frame noise configuration");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels; if((c.calibration.read_noise_map||c.calibration.fpn_map||c.calibration.gain_map)&&c.calibration.map_count!=n) throw std::invalid_argument("calibration noise maps must contain width*height*channels values");
}
}}
namespace augmatch {
void camera_profile_lookup_f32(const float* in,float* out,const CameraProfileLookupConfig& c,cudaStream_t){
  valid_camera(c,in,out); const float gain=camera_profile_gain(c.iso,c.profile); const float shot_scale=camera_profile_lookup_value(c.iso,c.profile,CameraProfileLookupParameter::ShotScale)*camera_profile_lookup_value(c.iso,c.profile,CameraProfileLookupParameter::ShotNoiseScale); const float read_sigma=camera_profile_lookup_value(c.iso,c.profile,CameraProfileLookupParameter::ReadNoiseStddev)*camera_profile_lookup_value(c.iso,c.profile,CameraProfileLookupParameter::ReadNoiseScale); const float fpn_sigma=camera_profile_lookup_value(c.iso,c.profile,CameraProfileLookupParameter::FPNStddev)*camera_profile_lookup_value(c.iso,c.profile,CameraProfileLookupParameter::FPNScale);
  if(!std::isfinite(gain)||!std::isfinite(shot_scale)||!std::isfinite(read_sigma)||!std::isfinite(fpn_sigma)||shot_scale<0||read_sigma<0||fpn_sigma<0) throw std::invalid_argument("invalid camera profile lookup values");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;float value=in[i]*gain;const auto key=camera_key(c.seed,x,y,ch,0x43414d4552415f53ULL);if(shot_scale>0){value=camera_poisson(std::max(0.0f,value*shot_scale),key)/shot_scale;}value+=read_sigma*camera_normal(key^0x524541445f4e4f49ULL)+fpn_sigma*camera_normal(key^0x46504e5f4e4f49ULL);out[i]=std::max(c.clip_min,std::min(c.clip_max,value));}
}
void calibration_frame_noise_f32(const float* in,float* out,const CalibrationFrameNoiseConfig& c,cudaStream_t){
  valid_calibration(c,in,out);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;const float gain=c.calibration.gain_map?c.calibration.gain_map[i]:1.0f;const float sigma=c.calibration.read_noise_map?c.calibration.read_noise_map[i]:c.calibration.read_noise_stddev;const float fpn=c.calibration.fpn_map?c.calibration.fpn_map[i]:c.calibration.fpn_stddev*camera_normal(camera_key(c.seed,x,y,ch,0x43414c5f46504e4eULL));const auto key=camera_key(c.seed,x,y,ch,0x43414c5f53484f54ULL);float value=in[i]*gain;if(c.calibration.shot_noise_scale>0)value=camera_poisson(std::max(0.0f,value*c.calibration.shot_noise_scale),key)/c.calibration.shot_noise_scale;value+=sigma*camera_normal(key^0x524541445f4e4f49ULL)+fpn;out[i]=std::max(c.clip_min,std::min(c.clip_max,value));}
}
}

