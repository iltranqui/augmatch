#include "augmatch/iso_profile.hpp"
#include <cassert>
#include <cmath>
#include <vector>
#include <stdexcept>
int main() {
  const augmatch::IsoNoiseProfilePoint points[]={{100,1,1,1,1,0,1},{400,4,1,1,1,0,1}};
  const augmatch::IsoNoiseProfile profile{points,2};
  std::vector<float> input(6,0.25f), a(6), b(6);
  augmatch::DualConversionGainConfig dual{}; dual.width=3;dual.height=1;dual.channels=2;dual.iso=100;dual.profile=profile;dual.low_gain_threshold=1;dual.high_gain_threshold=4;dual.low_conversion_gain=1;dual.high_conversion_gain=2;dual.low_read_noise_electrons=0;dual.high_read_noise_electrons=0;dual.seed=17;
  augmatch::dual_conversion_gain_f32(input.data(),a.data(),dual); augmatch::dual_conversion_gain_f32(input.data(),b.data(),dual); assert(a==b); for(float v:a) assert(std::abs(v-.25f)<1e-6f);
  dual.iso=400; augmatch::dual_conversion_gain_f32(input.data(),a.data(),dual); for(float v:a) assert(std::abs(v-.5f)<1e-6f);
  dual.iso=250; augmatch::dual_conversion_gain_f32(input.data(),a.data(),dual); for(float v:a) assert(v>.25f&&v<.5f);
  augmatch::GainSwitchTransitionConfig tr{};tr.width=3;tr.height=1;tr.channels=2;tr.iso=250;tr.profile=profile;tr.low_gain_threshold=1;tr.high_gain_threshold=4;tr.low_signal_gain=1;tr.high_signal_gain=2;tr.transition_width=.5f;tr.transition_strength=.1f;tr.seed=3;
  augmatch::gain_switch_transition_f32(input.data(),a.data(),tr);augmatch::gain_switch_transition_f32(input.data(),b.data(),tr);assert(a==b);for(float v:a)assert(v>=0&&v<=1);
  tr.iso=100;tr.low_gain_threshold=2;tr.high_gain_threshold=4;tr.hysteresis=2;tr.initial_high_gain=true;augmatch::gain_switch_transition_f32(input.data(),a.data(),tr);for(float v:a)assert(std::abs(v-.5f)<1e-6f);
  bool rejected=false;try{auto bad=dual;bad.high_gain_threshold=.5f;augmatch::dual_conversion_gain_f32(input.data(),a.data(),bad);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
  return 0;
}
