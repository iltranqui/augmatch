#include "augmatch/environmental.hpp"
#include <cmath>
#include <vector>
int main() {
  const int w=3,h=3,c=1; std::vector<float> image(w*h*c,0.2f), out(image.size());
  std::vector<float> map{0,1,0, 1,1,1, 0,1,0};
  augmatch::DirtyLensBlurConfig dirty{}; dirty.width=w;dirty.height=h;dirty.channels=c;dirty.map=map.data();dirty.radius=1;dirty.strength=1;
  image[4]=1.0f; augmatch::dirty_lens_blur_f32(image.data(),out.data(),dirty);
  if (!(out[4] < 1.0f && std::abs(out[0]-.2f)<1e-6f)) return 1;
  const int t=3; const std::size_t n=static_cast<std::size_t>(t)*h*w*c; std::vector<float> batch(n,.5f), a(n), b(n);
  augmatch::SensorTemperatureDriftConfig temp{};temp.frames=t;temp.height=h;temp.width=w;temp.channels=c;temp.start_temperature_celsius=25;temp.end_temperature_celsius=35;temp.dark_current_electrons_per_second=100;temp.exposure_seconds=1;temp.electrons_per_unit=1000;temp.temperature_coefficient_per_celsius=std::log(2.0f)/10.0f;
  augmatch::sensor_temperature_drift_f32(batch.data(),a.data(),temp); if (!(a[2*w*h] > a[0])) return 2;
  augmatch::ElectromagneticInterferenceConfig emi{};emi.width=w;emi.height=h;emi.channels=c;emi.amplitude=.1f;emi.frequency_x=1;emi.frequency_y=0;emi.phase=0;augmatch::electromagnetic_interference_f32(image.data(),out.data(),emi);if(std::abs(out[0]-.2f)>1e-6f)return 3;
  augmatch::PowerSupplyBandingConfig band{};band.frames=t;band.height=h;band.width=w;band.channels=c;band.amplitude=.2f;band.temporal_frequency_hz=.5f;band.frame_rate_hz=1;band.row_frequency=1;augmatch::power_supply_banding_f32(batch.data(),a.data(),band);if(a==batch)return 4;
  augmatch::FluorescentLightFlickerConfig fl{};fl.frames=t;fl.height=h;fl.width=w;fl.channels=c;fl.amplitude=.2f;fl.flicker_frequency_hz=.25f;fl.frame_rate_hz=1;augmatch::fluorescent_light_flicker_f32(batch.data(),a.data(),fl);if(a==batch)return 5;
  augmatch::LedRollingBandArtifactsConfig led{};led.frames=t;led.height=h;led.width=w;led.channels=c;led.amplitude=.2f;led.modulation_frequency_hz=.5f;led.frame_rate_hz=1;led.row_cycles=1;augmatch::led_rolling_band_artifacts_f32(batch.data(),b.data(),led);if(b==batch)return 6;
  augmatch::sensor_temperature_drift_f32(batch.data(),a.data(),temp);augmatch::sensor_temperature_drift_f32(batch.data(),b.data(),temp);if(a!=b)return 7;
  return 0;
}
