#include "augmatch/weather/environmental.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__device__ float veil_map(const EnvironmentalVeilConfig& c,int x,int y,int ch) {
  if(!c.map) return 1.0f;
  int p=y*c.width+x;
  return fminf(1.0f,fmaxf(0.0f,c.map[c.map_channels==1?p:p*c.map_channels+ch]));
}
__device__ float clip_value(float v,float lo,float hi){return fminf(hi,fmaxf(lo,v));}
__global__ void veil_kernel(const float* in,float* out,EnvironmentalVeilConfig c){
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;
  int ch=i%c.channels,x=(i/c.channels)%c.width,y=i/(c.width*c.channels);
  float a=fminf(1.0f,fmaxf(0.0f,c.strength*veil_map(c,x,y,ch)));
  out[i]=clip_value((1.0f-a)*in[i]+a*c.airlight,c.clip_min,c.clip_max);
}
__global__ void gain_kernel(const float* in,float* out,AcquisitionGainConfig c){
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;
  out[i]=clip_value(in[i]*c.gain,c.clip_min,c.clip_max);
}
void check(cudaError_t e,const char* p){if(e!=cudaSuccess)throw std::runtime_error(std::string(p)+": "+cudaGetErrorString(e));}
void valid_image(int w,int h,int ch,const float* in,const float* out,float lo,float hi){
  if(w<=0||h<=0||ch<=0||!in||!out||!isfinite(lo)||!isfinite(hi)||hi<lo)throw std::invalid_argument("invalid environmental image configuration");
}
void valid_veil(const EnvironmentalVeilConfig& c,const float* in,const float* out){
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(c.map_channels!=1&&c.map_channels!=c.channels||!isfinite(c.strength)||c.strength<0||c.strength>1||!isfinite(c.airlight))throw std::invalid_argument("invalid environmental veil configuration");
}
void valid_gain(const AcquisitionGainConfig& c,const float* in,const float* out){
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(!isfinite(c.gain)||c.gain<0)throw std::invalid_argument("invalid acquisition gain configuration");
}
void launch_veil(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t s){valid_veil(c,i,o);int n=c.width*c.height*c.channels;veil_kernel<<<(n+255)/256,256,0,s>>>(i,o,c);cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void launch_gain(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t s){valid_gain(c,i,o);int n=c.width*c.height*c.channels;gain_kernel<<<(n+255)/256,256,0,s>>>(i,o,c);cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
void environmental_veil_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t s){launch_veil(i,o,c,s);}
void atmospheric_haze_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t s){launch_veil(i,o,c,s);}
void smoke_veil_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t s){launch_veil(i,o,c,s);}
void window_glare_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t s){launch_veil(i,o,c,s);}
void backlight_washout_f32(const float* i,float* o,const EnvironmentalVeilConfig& c,cudaStream_t s){launch_veil(i,o,c,s);}
void acquisition_gain_f32(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t s){launch_gain(i,o,c,s);}
void low_light_amplification_f32(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t s){launch_gain(i,o,c,s);}
void underexposure_f32(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t s){launch_gain(i,o,c,s);}
void overexposure_f32(const float* i,float* o,const AcquisitionGainConfig& c,cudaStream_t s){launch_gain(i,o,c,s);}

__device__ float dirty_map(const DirtyLensBlurConfig& c,int x,int y,int ch){if(!c.map)return 1.0f;const int p=y*c.width+x;return fmaxf(0.0f,c.map[c.map_channels==1?p:p*c.map_channels+ch]);}
__global__ void dirty_kernel(const float* in,float* out,DirtyLensBlurConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int ch=i%c.channels,x=(i/c.channels)%c.width,y=i/(c.width*c.channels);float sum=0.0f;int count=0;for(int dy=-c.radius;dy<=c.radius;++dy)for(int dx=-c.radius;dx<=c.radius;++dx){int sx=max(0,min(c.width-1,x+dx)),sy=max(0,min(c.height-1,y+dy));sum+=in[(sy*c.width+sx)*c.channels+ch];++count;}float a=fminf(1.0f,fmaxf(0.0f,c.strength*dirty_map(c,x,y,ch)));out[i]=clip_value((1.0f-a)*in[i]+a*sum/(float)count,c.clip_min,c.clip_max);}
__device__ float temp_map(const SensorTemperatureDriftConfig& c,int x,int y,int ch){if(!c.sensitivity_map)return 1.0f;int p=y*c.width+x;return fmaxf(0.0f,c.sensitivity_map[c.map_channels==1?p:p*c.map_channels+ch]);}
__global__ void temperature_kernel(const float* in,float* out,SensorTemperatureDriftConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.frames*c.height*c.width*c.channels;if(i>=n)return;int ch=i%c.channels,x=(i/c.channels)%c.width,y=(i/c.channels/c.width)%c.height,t=i/(c.channels*c.width*c.height);float u=c.frames==1?0.0f:(float)t/(float)(c.frames-1),temp=c.start_temperature_celsius+(c.end_temperature_celsius-c.start_temperature_celsius)*u,dark=c.dark_current_electrons_per_second*c.exposure_seconds*expf(c.temperature_coefficient_per_celsius*(temp-c.reference_temperature_celsius))/c.electrons_per_unit;out[i]=clip_value(in[i]+dark*temp_map(c,x,y,ch),c.clip_min,c.clip_max);}
__device__ float emi_map(const ElectromagneticInterferenceConfig& c,int x,int y,int ch){if(!c.map)return 1.0f;int p=y*c.width+x;return fmaxf(0.0f,c.map[c.map_channels==1?p:p*c.map_channels+ch]);}
__global__ void emi_kernel(const float* in,float* out,ElectromagneticInterferenceConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int ch=i%c.channels,x=(i/c.channels)%c.width,y=i/(c.width*c.channels);float pi=3.14159265358979323846f,phase=2.0f*pi*(c.frequency_x*(float)x/(float)max(1,c.width)+c.frequency_y*(float)y/(float)max(1,c.height))+c.phase;out[i]=clip_value(in[i]+c.amplitude*emi_map(c,x,y,ch)*sinf(phase),c.clip_min,c.clip_max);}
__device__ float row_multiplier(const PowerSupplyBandingConfig& c,int y,int ch){if(!c.row_map)return 1.0f;return fmaxf(0.0f,c.row_map[c.row_map_channels==1?y:y*c.channels+ch]);}
__global__ void banding_kernel(const float* in,float* out,PowerSupplyBandingConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.frames*c.height*c.width*c.channels;if(i>=n)return;int ch=i%c.channels,y=(i/c.channels/c.width)%c.height,t=i/(c.channels*c.width*c.height);float pi=3.14159265358979323846f,phase=2.0f*pi*(c.temporal_frequency_hz*(float)t/c.frame_rate_hz+c.row_frequency*(float)y/(float)max(1,c.height))+c.phase;out[i]=clip_value(in[i]*(1.0f+c.amplitude*row_multiplier(c,y,ch)*sinf(phase)),c.clip_min,c.clip_max);}
__global__ void fluorescent_kernel(const float* in,float* out,FluorescentLightFlickerConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.frames*c.height*c.width*c.channels;if(i>=n)return;int t=i/(c.channels*c.width*c.height);float pi=3.14159265358979323846f,g=1.0f+c.amplitude*sinf(2.0f*pi*c.flicker_frequency_hz*(float)t/c.frame_rate_hz+c.phase);out[i]=clip_value(in[i]*g,c.clip_min,c.clip_max);}
__global__ void led_kernel(const float* in,float* out,LedRollingBandArtifactsConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.frames*c.height*c.width*c.channels;if(i>=n)return;int y=(i/c.channels/c.width)%c.height,t=i/(c.channels*c.width*c.height);float pi=3.14159265358979323846f,g=1.0f+c.amplitude*sinf(2.0f*pi*(c.modulation_frequency_hz*(float)t/c.frame_rate_hz+c.row_cycles*(float)y/(float)max(1,c.height))+c.phase);out[i]=clip_value(in[i]*g,c.clip_min,c.clip_max);}
void valid_temporal(const TemporalBatchConfig& c,const float* in,const float* out){if(c.frames<=0||c.height<=0||c.width<=0||c.channels<=0||!in||!out||!isfinite(c.clip_min)||!isfinite(c.clip_max)||c.clip_max<c.clip_min)throw std::invalid_argument("invalid environmental temporal configuration");}
void valid_dirty(const DirtyLensBlurConfig& c,const float* i,const float* o){valid_image(c.width,c.height,c.channels,i,o,c.clip_min,c.clip_max);if(c.map_channels!=1&&c.map_channels!=c.channels||c.radius<0||c.radius>16||!isfinite(c.strength)||c.strength<0||c.strength>1)throw std::invalid_argument("invalid dirty-lens blur configuration");}
void valid_temperature(const SensorTemperatureDriftConfig& c,const float* i,const float* o){valid_temporal(c,i,o);if(!isfinite(c.start_temperature_celsius)||!isfinite(c.end_temperature_celsius)||!isfinite(c.reference_temperature_celsius)||!isfinite(c.dark_current_electrons_per_second)||c.dark_current_electrons_per_second<0||!isfinite(c.exposure_seconds)||c.exposure_seconds<0||!isfinite(c.electrons_per_unit)||c.electrons_per_unit<=0||!isfinite(c.temperature_coefficient_per_celsius)||c.map_channels!=1&&c.map_channels!=c.channels)throw std::invalid_argument("invalid sensor temperature drift configuration");}
void valid_emi(const ElectromagneticInterferenceConfig& c,const float* i,const float* o){valid_image(c.width,c.height,c.channels,i,o,c.clip_min,c.clip_max);if(c.map_channels!=1&&c.map_channels!=c.channels||!isfinite(c.amplitude)||!isfinite(c.frequency_x)||!isfinite(c.frequency_y)||!isfinite(c.phase))throw std::invalid_argument("invalid electromagnetic interference configuration");}
void valid_banding(const PowerSupplyBandingConfig& c,const float* i,const float* o){valid_temporal(c,i,o);if(!isfinite(c.amplitude)||!isfinite(c.temporal_frequency_hz)||!isfinite(c.frame_rate_hz)||c.frame_rate_hz<=0||!isfinite(c.row_frequency)||!isfinite(c.phase)||c.row_map_channels!=1&&c.row_map_channels!=c.channels)throw std::invalid_argument("invalid power-supply banding configuration");}
void valid_fluorescent(const FluorescentLightFlickerConfig& c,const float* i,const float* o){valid_temporal(c,i,o);if(!isfinite(c.amplitude)||!isfinite(c.flicker_frequency_hz)||!isfinite(c.frame_rate_hz)||c.frame_rate_hz<=0||!isfinite(c.phase))throw std::invalid_argument("invalid fluorescent flicker configuration");}
void valid_led(const LedRollingBandArtifactsConfig& c,const float* i,const float* o){valid_temporal(c,i,o);if(!isfinite(c.amplitude)||!isfinite(c.modulation_frequency_hz)||!isfinite(c.frame_rate_hz)||c.frame_rate_hz<=0||!isfinite(c.row_cycles)||!isfinite(c.phase))throw std::invalid_argument("invalid LED rolling-band configuration");}
void launch_dirty(const float* i,float* o,const DirtyLensBlurConfig& c,cudaStream_t s){valid_dirty(c,i,o);int n=c.width*c.height*c.channels;dirty_kernel<<<(n+255)/256,256,0,s>>>(i,o,c);check(cudaGetLastError(),"dirty_lens_blur");}
void launch_temperature(const float* i,float* o,const SensorTemperatureDriftConfig& c,cudaStream_t s){valid_temperature(c,i,o);int n=c.frames*c.height*c.width*c.channels;temperature_kernel<<<(n+255)/256,256,0,s>>>(i,o,c);check(cudaGetLastError(),"sensor_temperature_drift");}
void launch_emi(const float* i,float* o,const ElectromagneticInterferenceConfig& c,cudaStream_t s){valid_emi(c,i,o);int n=c.width*c.height*c.channels;emi_kernel<<<(n+255)/256,256,0,s>>>(i,o,c);check(cudaGetLastError(),"electromagnetic_interference");}
void launch_banding(const float* i,float* o,const PowerSupplyBandingConfig& c,cudaStream_t s){valid_banding(c,i,o);int n=c.frames*c.height*c.width*c.channels;banding_kernel<<<(n+255)/256,256,0,s>>>(i,o,c);check(cudaGetLastError(),"power_supply_banding");}
void launch_fluorescent(const float* i,float* o,const FluorescentLightFlickerConfig& c,cudaStream_t s){valid_fluorescent(c,i,o);int n=c.frames*c.height*c.width*c.channels;fluorescent_kernel<<<(n+255)/256,256,0,s>>>(i,o,c);check(cudaGetLastError(),"fluorescent_light_flicker");}
void launch_led(const float* i,float* o,const LedRollingBandArtifactsConfig& c,cudaStream_t s){valid_led(c,i,o);int n=c.frames*c.height*c.width*c.channels;led_kernel<<<(n+255)/256,256,0,s>>>(i,o,c);check(cudaGetLastError(),"led_rolling_band_artifacts");}
void dirty_lens_blur_f32(const float* i,float* o,const DirtyLensBlurConfig& c,cudaStream_t s){launch_dirty(i,o,c,s);}
void sensor_temperature_drift_f32(const float* i,float* o,const SensorTemperatureDriftConfig& c,cudaStream_t s){launch_temperature(i,o,c,s);}
void electromagnetic_interference_f32(const float* i,float* o,const ElectromagneticInterferenceConfig& c,cudaStream_t s){launch_emi(i,o,c,s);}
void power_supply_banding_f32(const float* i,float* o,const PowerSupplyBandingConfig& c,cudaStream_t s){launch_banding(i,o,c,s);}
void fluorescent_light_flicker_f32(const float* i,float* o,const FluorescentLightFlickerConfig& c,cudaStream_t s){launch_fluorescent(i,o,c,s);}
void led_rolling_band_artifacts_f32(const float* i,float* o,const LedRollingBandArtifactsConfig& c,cudaStream_t s){launch_led(i,o,c,s);}
} // namespace augmatch
