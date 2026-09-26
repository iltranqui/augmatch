#include "augmatch/filter/filter.hpp"
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
int main(){
  const int w=9,h=7,c=3; std::vector<std::uint8_t> input(static_cast<std::size_t>(w)*h*c), output(input.size());
  for(std::size_t i=0;i<input.size();++i) input[i]=static_cast<std::uint8_t>((i*37)%256);
  augmatch::diffraction_blur_u8(input.data(),output.data(),{w,h,c,0,550,2,50}); assert(output==input);
  augmatch::bokeh_blur_u8(input.data(),output.data(),{w,h,c,0,0}); assert(output==input);
  augmatch::cat_eye_bokeh_u8(input.data(),output.data(),{w,h,c,0,.5f,.5f,.5f}); assert(output==input);
  augmatch::aperture_shape_blur_u8(input.data(),output.data(),{w,h,c,0,6,0,.0f}); assert(output==input);
  std::vector<std::uint8_t> constant(input.size(),91);
  augmatch::diffraction_blur_u8(constant.data(),output.data(),{w,h,c,4,550,2,50}); assert(output==constant);
  augmatch::bokeh_blur_u8(constant.data(),output.data(),{w,h,c,4,.3f}); assert(output==constant);
  augmatch::cat_eye_bokeh_u8(constant.data(),output.data(),{w,h,c,4,.8f,.5f,.5f}); assert(output==constant);
  augmatch::aperture_shape_blur_u8(constant.data(),output.data(),{w,h,c,4,5,17,.2f}); assert(output==constant);
  bool threw=false; try{augmatch::bokeh_blur_u8(input.data(),output.data(),{w,h,c,4,1.1f});}catch(const std::invalid_argument&){threw=true;} assert(threw);
  threw=false; try{augmatch::aperture_shape_blur_u8(input.data(),output.data(),{w,h,c,4,2,0,0});}catch(const std::invalid_argument&){threw=true;} assert(threw);
  return 0;
}
