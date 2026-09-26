#include "augmatch/sensor/iso_profile.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
int main(){
  const augmatch::CameraProfileLookupPoint p[]={{100,1,1,1,1,0,1,100,0.01f,0.02f},{400,4,3,2,5,0,1,300,0.03f,0.04f}};
  augmatch::CameraProfileLookupTable table{p,2};
  assert(std::abs(augmatch::camera_profile_gain(250,table)-2.5f)<1e-6f);
  assert(std::abs(augmatch::camera_profile_shot_scale(250,table)-200.0f)<1e-6f);
  assert(std::abs(augmatch::camera_profile_read_noise_stddev(250,table)-.02f)<1e-6f);
  augmatch::CameraProfileLookupConfig camera{};camera.width=4;camera.height=1;camera.channels=1;camera.iso=250;camera.profile=table;camera.seed=7;
  const float input[]={.1f,.2f,.3f,.4f};float a[4]{},b[4]{};augmatch::camera_profile_lookup_f32(input,a,camera);augmatch::camera_profile_lookup_f32(input,b,camera);for(int i=0;i<4;++i){assert(a[i]==b[i]);assert(a[i]>=0&&a[i]<=1);}
  const float read_map[]={0,.01f,.02f,.03f}, fpn_map[]={0,.01f,-.01f,.02f}, gain_map[]={1,1.1f,.9f,1};
  augmatch::CalibrationFrameNoiseConfig calibration{};calibration.width=4;calibration.height=1;calibration.channels=1;calibration.calibration.read_noise_map=read_map;calibration.calibration.fpn_map=fpn_map;calibration.calibration.gain_map=gain_map;calibration.calibration.map_count=4;calibration.seed=11;
  float c[4]{},d[4]{};augmatch::calibration_frame_noise_f32(input,c,calibration);augmatch::calibration_frame_noise_f32(input,d,calibration);for(int i=0;i<4;++i)assert(c[i]==d[i]);
  calibration.calibration.map_count=3;bool threw=false;try{augmatch::calibration_frame_noise_f32(input,d,calibration);}catch(const std::invalid_argument&){threw=true;}assert(threw);return 0;
}
