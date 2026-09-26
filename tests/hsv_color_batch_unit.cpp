#include "augmatch/hsv.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
int main(){
  const int w=3,h=1,c=4; const std::vector<std::uint8_t> input={255,0,0,17,0,255,0,23,0,0,255,29}; std::vector<std::uint8_t> output(input.size());
  augmatch::multiply_brightness_u8(input.data(),output.data(),{w,h,c,0.5f}); assert(output[0]==128&&output[1]==0&&output[2]==0&&output[3]==17);
  augmatch::add_to_brightness_u8(input.data(),output.data(),{w,h,c,10.0f}); assert(output[0]==255&&output[5]==255&&output[10]==255);
  augmatch::multiply_and_add_to_brightness_u8(input.data(),output.data(),{w,h,c,0.5f,10.0f}); assert(output[0]==138&&output[5]==138&&output[10]==138&&output[3]==17);
  augmatch::with_hue_and_saturation_u8(input.data(),output.data(),{w,h,c,60.0f,255.0f}); assert(output[0]==0&&output[1]==255&&output[2]==0);
  augmatch::multiply_hue_and_saturation_u8(input.data(),output.data(),{w,h,c,1.0f,0.0f}); assert(output[0]==255&&output[1]==255&&output[2]==255&&output[3]==17);
  augmatch::multiply_hue_u8(input.data(),output.data(),{w,h,c,2.0f}); assert(output[0]==255&&output[1]==0&&output[2]==0);
  augmatch::multiply_saturation_u8(input.data(),output.data(),{w,h,c,0.0f}); assert(output[0]==255&&output[1]==255&&output[2]==255);
  augmatch::remove_saturation_u8(input.data(),output.data(),{w,h,c}); assert(output[0]==255&&output[1]==255&&output[2]==255);
  augmatch::add_to_hue_and_saturation_u8(input.data(),output.data(),{w,h,c,60.0f,0.0f}); assert(output[0]==0&&output[1]==255&&output[2]==0);
  augmatch::add_to_hue_u8(input.data(),output.data(),{w,h,c,-60.0f}); assert(output[0]==0&&output[1]==0&&output[2]==255);
  augmatch::add_to_saturation_u8(input.data(),output.data(),{w,h,c,-255.0f}); assert(output[0]==255&&output[1]==255&&output[2]==255);
  augmatch::add_to_saturation_u8(input.data(),output.data(),{w,h,c,-128.0f}); assert(output[0]==255&&output[1]==128&&output[2]==128&&output[3]==17);
  return 0;
}
