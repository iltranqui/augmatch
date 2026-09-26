#include "augmatch/environmental.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace augmatch {
namespace {
void valid_image(int w,int h,int c,const float* in,const float* out,float lo,float hi) {
  if (w<=0||h<=0||c<=0||!in||!out||!std::isfinite(lo)||!std::isfinite(hi)||hi<lo)
    throw std::invalid_argument("invalid environmental image configuration");
}
void valid_map(const EnvironmentalVeilConfig& c) {
  if (c.map_channels!=1&&c.map_channels!=c.channels)
    throw std::invalid_argument("environmental veil map must be HxW or HxWxC");
  if (!c.map) return;
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.map_channels;
  for (std::size_t i=0;i<n;++i)
    if (!std::isfinite(c.map[i])||c.map[i]<0.0f) throw std::invalid_argument("environmental veil map must be finite and nonnegative");
}
float map_at(const EnvironmentalVeilConfig& c,int x,int y,int ch) {
  if (!c.map) return 1.0f;
  const std::size_t p=static_cast<std::size_t>(y)*c.width+x;
  return std::max(0.0f,std::min(1.0f,c.map[c.map_channels==1?p:p*c.map_channels+ch]));
}
void apply_veil(const float* in,float* out,const EnvironmentalVeilConfig& c) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if (!std::isfinite(c.strength)||c.strength<0.0f||c.strength>1.0f||!std::isfinite(c.airlight))
    throw std::invalid_argument("invalid environmental veil strength or airlight");
  valid_map(c);
  for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) for (int ch=0;ch<c.channels;++ch) {
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
    const float a=std::max(0.0f,std::min(1.0f,c.strength*map_at(c,x,y,ch)));
    out[i]=std::max(c.clip_min,std::min(c.clip_max,(1.0f-a)*in[i]+a*c.airlight));
  }
}
void apply_gain(const float* in,float* out,const AcquisitionGainConfig& c) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if (!std::isfinite(c.gain)||c.gain<0.0f) throw std::invalid_argument("invalid acquisition gain");
  for (std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height*c.channels;i<n;++i)
    out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*c.gain));
}
}
void environmental_veil_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t) { apply_veil(i,o,c); }
void atmospheric_haze_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t) { apply_veil(i,o,c); }
void smoke_veil_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t) { apply_veil(i,o,c); }
void window_glare_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t) { apply_veil(i,o,c); }
void backlight_washout_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t) { apply_veil(i,o,c); }
void acquisition_gain_f32(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t) { apply_gain(i,o,c); }
void low_light_amplification_f32(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t) { apply_gain(i,o,c); }
void underexposure_f32(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t) { apply_gain(i,o,c); }
void overexposure_f32(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t) { apply_gain(i,o,c); }

void dirty_lens_blur_f32(const float* in,float* out,const DirtyLensBlurConfig& c,cudaStream_t) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(c.map_channels!=1&&c.map_channels!=c.channels||c.radius<0||c.radius>16||!std::isfinite(c.strength)||c.strength<0||c.strength>1)
    throw std::invalid_argument("invalid dirty-lens blur configuration");
  if(c.map){const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.map_channels;for(std::size_t i=0;i<n;++i)if(!std::isfinite(c.map[i])||c.map[i]<0)throw std::invalid_argument("dirty-lens map must be finite and nonnegative");}
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
    float sum=0.0f; int count=0;
    for(int dy=-c.radius;dy<=c.radius;++dy) for(int dx=-c.radius;dx<=c.radius;++dx){const int sx=std::max(0,std::min(c.width-1,x+dx)),sy=std::max(0,std::min(c.height-1,y+dy));sum+=in[(static_cast<std::size_t>(sy)*c.width+sx)*c.channels+ch];++count;}
    const std::size_t mp=static_cast<std::size_t>(y)*c.width+x;
    const float mask=c.map?c.map[c.map_channels==1?mp:mp*c.map_channels+ch]:1.0f;
    const float a=std::max(0.0f,std::min(1.0f,c.strength*mask));
    out[i]=std::max(c.clip_min,std::min(c.clip_max,(1.0f-a)*in[i]+a*(sum/static_cast<float>(count))));
  }
}

void sensor_temperature_drift_f32(const float* in,float* out,const SensorTemperatureDriftConfig& c,cudaStream_t) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(c.frames<=0||!std::isfinite(c.temporal_correlation)||c.temporal_correlation<-1||c.temporal_correlation>1||!std::isfinite(c.start_temperature_celsius)||!std::isfinite(c.end_temperature_celsius)||!std::isfinite(c.reference_temperature_celsius)||!std::isfinite(c.dark_current_electrons_per_second)||c.dark_current_electrons_per_second<0||!std::isfinite(c.exposure_seconds)||c.exposure_seconds<0||!std::isfinite(c.electrons_per_unit)||c.electrons_per_unit<=0||!std::isfinite(c.temperature_coefficient_per_celsius)||c.map_channels!=1&&c.map_channels!=c.channels)
    throw std::invalid_argument("invalid sensor temperature drift configuration");
  if(c.sensitivity_map){const std::size_t n=static_cast<std::size_t>(c.height)*c.width*c.map_channels;for(std::size_t i=0;i<n;++i)if(!std::isfinite(c.sensitivity_map[i])||c.sensitivity_map[i]<0)throw std::invalid_argument("temperature sensitivity map must be finite and nonnegative");}
  const std::size_t plane=static_cast<std::size_t>(c.height)*c.width*c.channels;
  for(int t=0;t<c.frames;++t){const float u=c.frames==1?0.0f:static_cast<float>(t)/static_cast<float>(c.frames-1);const float temp=c.start_temperature_celsius+(c.end_temperature_celsius-c.start_temperature_celsius)*u;const float dark=c.dark_current_electrons_per_second*c.exposure_seconds*std::exp(c.temperature_coefficient_per_celsius*(temp-c.reference_temperature_celsius))/c.electrons_per_unit;for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t p=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;const float scale=c.sensitivity_map?(c.sensitivity_map[c.map_channels==1?static_cast<std::size_t>(y)*c.width+x:p]):1.0f;out[static_cast<std::size_t>(t)*plane+p]=std::max(c.clip_min,std::min(c.clip_max,in[static_cast<std::size_t>(t)*plane+p]+dark*scale));}}
}

void electromagnetic_interference_f32(const float* in,float* out,const ElectromagneticInterferenceConfig& c,cudaStream_t) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(c.map_channels!=1&&c.map_channels!=c.channels||!std::isfinite(c.amplitude)||!std::isfinite(c.frequency_x)||!std::isfinite(c.frequency_y)||!std::isfinite(c.phase))throw std::invalid_argument("invalid electromagnetic interference configuration");
  if(c.map){const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.map_channels;for(std::size_t i=0;i<n;++i)if(!std::isfinite(c.map[i])||c.map[i]<0)throw std::invalid_argument("EMI map must be finite and nonnegative");}
  constexpr float pi=3.14159265358979323846f;for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch,mp=static_cast<std::size_t>(y)*c.width+x;const float m=c.map?c.map[c.map_channels==1?mp:mp*c.map_channels+ch]:1.0f;const float n=std::sin(2.0f*pi*(c.frequency_x*(static_cast<float>(x)/std::max(1,c.width))+c.frequency_y*(static_cast<float>(y)/std::max(1,c.height)))+c.phase);out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]+c.amplitude*m*n));}
}

void power_supply_banding_f32(const float* in,float* out,const PowerSupplyBandingConfig& c,cudaStream_t) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(c.frames<=0||!std::isfinite(c.amplitude)||!std::isfinite(c.temporal_frequency_hz)||!std::isfinite(c.frame_rate_hz)||c.frame_rate_hz<=0||!std::isfinite(c.row_frequency)||!std::isfinite(c.phase)||c.row_map_channels!=1&&c.row_map_channels!=c.channels)throw std::invalid_argument("invalid power-supply banding configuration");
  constexpr float pi=3.14159265358979323846f;const std::size_t plane=static_cast<std::size_t>(c.height)*c.width*c.channels;for(int t=0;t<c.frames;++t)for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t p=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch,i=static_cast<std::size_t>(t)*plane+p,mp=static_cast<std::size_t>(y)*c.channels+ch;const float rm=c.row_map?(c.row_map[c.row_map_channels==1?y:mp]):1.0f;const float phase=2.0f*pi*(c.temporal_frequency_hz*static_cast<float>(t)/c.frame_rate_hz+c.row_frequency*static_cast<float>(y)/std::max(1,c.height))+c.phase;out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*(1.0f+c.amplitude*rm*std::sin(phase))));}
}

void fluorescent_light_flicker_f32(const float* in,float* out,const FluorescentLightFlickerConfig& c,cudaStream_t) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(c.frames<=0||!std::isfinite(c.amplitude)||!std::isfinite(c.flicker_frequency_hz)||!std::isfinite(c.frame_rate_hz)||c.frame_rate_hz<=0||!std::isfinite(c.phase))throw std::invalid_argument("invalid fluorescent flicker configuration");
  constexpr float pi=3.14159265358979323846f;const std::size_t plane=static_cast<std::size_t>(c.height)*c.width*c.channels;for(int t=0;t<c.frames;++t){const float gain=1.0f+c.amplitude*std::sin(2.0f*pi*c.flicker_frequency_hz*static_cast<float>(t)/c.frame_rate_hz+c.phase);for(std::size_t p=0;p<plane;++p)out[static_cast<std::size_t>(t)*plane+p]=std::max(c.clip_min,std::min(c.clip_max,in[static_cast<std::size_t>(t)*plane+p]*gain));}
}

void led_rolling_band_artifacts_f32(const float* in,float* out,const LedRollingBandArtifactsConfig& c,cudaStream_t) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(c.frames<=0||!std::isfinite(c.amplitude)||!std::isfinite(c.modulation_frequency_hz)||!std::isfinite(c.frame_rate_hz)||c.frame_rate_hz<=0||!std::isfinite(c.row_cycles)||!std::isfinite(c.phase))throw std::invalid_argument("invalid LED rolling-band configuration");
  constexpr float pi=3.14159265358979323846f;const std::size_t plane=static_cast<std::size_t>(c.height)*c.width*c.channels;for(int t=0;t<c.frames;++t)for(int y=0;y<c.height;++y){const float gain=1.0f+c.amplitude*std::sin(2.0f*pi*(c.modulation_frequency_hz*static_cast<float>(t)/c.frame_rate_hz+c.row_cycles*static_cast<float>(y)/std::max(1,c.height))+c.phase);for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t p=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch,i=static_cast<std::size_t>(t)*plane+p;out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*gain));}}
}
}  // namespace augmatch
