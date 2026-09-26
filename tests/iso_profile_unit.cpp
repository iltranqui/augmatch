#include "augmatch/sensor/iso_profile.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
int main(){
  const augmatch::IsoNoiseProfilePoint points[]={{100,1,1,1,1,0.02f,1.0f},{800,4,2,3,5,0.08f,0.9f},{3200,8,3,5,9,0.12f,0.75f}};
  const augmatch::IsoNoiseProfile profile{points,3};
  assert(std::abs(augmatch::iso_to_gain(450,profile)-2.5f)<1e-6f);
  assert(std::abs(augmatch::iso_shot_noise_scale(1600,profile)-2.3333333f)<1e-6f);
  assert(std::abs(augmatch::iso_read_noise_scale(1600,profile)-3.6666667f)<1e-6f);
  assert(std::abs(augmatch::iso_fpn_scale(1600,profile)-6.3333333f)<1e-6f);
  assert(std::abs(augmatch::iso_black_level(50,profile)-0.02f)<1e-6f);
  assert(std::abs(augmatch::iso_saturation_level(6400,profile)-0.75f)<1e-6f);
  augmatch::IsoNoiseApplicationConfig c{}; c.width=4;c.height=1;c.channels=1;c.iso=1600;c.profile=profile;c.shot_scale=100;c.read_noise_stddev=0;c.fpn_stddev=0;c.seed=9;
  const std::uint8_t input[]={0,64,192,255};std::uint8_t output[4]{};
  augmatch::iso_signal_levels_u8(input,output,c); assert(output[0]==24);assert(output[1]==64);assert(output[2]==192);assert(output[3]==217);
  std::vector<float> f(4,0.5f),g(4); c.read_noise_stddev=0.0f; augmatch::iso_read_noise_f32(f.data(),g.data(),c); for(float v:g)assert(v==0.5f);
  augmatch::ExposureTimeDarkCurrentConfig dark{}; dark.width=4;dark.height=1;dark.channels=1;dark.iso=800;dark.profile=profile;dark.dark_current_electrons_per_second=100.0f;dark.exposure_seconds=0.0f;dark.electrons_per_unit=1000.0f;dark.seed=42;
  const float dark_in[]={0.1f,0.2f,0.3f,0.4f};float dark_zero[4]{},dark_a[4]{},dark_b[4]{};augmatch::exposure_time_dark_current_f32(dark_in,dark_zero,dark);for(int i=0;i<4;++i)assert(dark_zero[i]==dark_in[i]);
  dark.exposure_seconds=1.0f;augmatch::exposure_time_dark_current_f32(dark_in,dark_a,dark);augmatch::exposure_time_dark_current_f32(dark_in,dark_b,dark);for(int i=0;i<4;++i){assert(dark_a[i]==dark_b[i]);assert(dark_a[i]>=dark_in[i]);assert(dark_a[i]<=1.0f);}
  augmatch::TemperatureNoiseScalingConfig thermal{};thermal.width=4;thermal.height=1;thermal.channels=1;thermal.iso=800;thermal.profile=profile;thermal.noise_stddev_electrons=20.0f;thermal.temperature_celsius=25.0f;thermal.reference_temperature_celsius=25.0f;thermal.temperature_noise_coefficient_per_celsius=std::log(2.0f)/10.0f;thermal.electrons_per_unit=1000.0f;thermal.seed=7;
  float thermal_a[4]{},thermal_b[4]{};augmatch::temperature_noise_scaling_f32(dark_in,thermal_a,thermal);augmatch::temperature_noise_scaling_f32(dark_in,thermal_b,thermal);for(int i=0;i<4;++i){assert(thermal_a[i]==thermal_b[i]);assert(thermal_a[i]>=0.0f&&thermal_a[i]<=1.0f);}
  thermal.noise_stddev_electrons=0.0f;augmatch::temperature_noise_scaling_f32(dark_in,thermal_b,thermal);for(int i=0;i<4;++i)assert(thermal_b[i]==dark_in[i]);
  return 0;
}
