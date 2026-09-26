#include "augmatch/colorspace.hpp"
#include <cstdint>
#include <vector>
int main(){
  const int w=2,h=1,c=4; const std::vector<std::uint8_t> input={10,20,30,77,200,100,50,88}; std::vector<std::uint8_t> out(input.size());
  augmatch::with_colorspace_u8(input.data(),out.data(),{w,h,c,augmatch::ColorSpace::RGB,2,2,2,1,2,3});
  if(out != std::vector<std::uint8_t>({21,42,63,77,255,202,103,88})) return 1;
  augmatch::with_brightness_channels_u8(input.data(),out.data(),{w,h,c,augmatch::ColorSpace::RGB,1,-1,-1,2,5});
  if(out != std::vector<std::uint8_t>({10,45,30,77,200,205,50,88})) return 2;
  augmatch::with_brightness_channels_u8(input.data(),out.data(),{w,h,c,augmatch::ColorSpace::HSV,2,-1,-1,.5f,10});
  if(out[3]!=77||out[7]!=88) return 3;
  augmatch::with_brightness_channels_u8(input.data(),out.data(),{w,h,c,augmatch::ColorSpace::HSV,-1,-1,-1,.5f,10});
  if(out!=input) return 4;
  augmatch::with_colorspace_u8(input.data(),out.data(),{w,h,c,augmatch::ColorSpace::LAB,1,1,1,0,0,0});
  if(out[3]!=77||out[7]!=88) return 5;
  return 0;
}
