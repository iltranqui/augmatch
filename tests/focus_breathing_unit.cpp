#include "augmatch/geometric.hpp"
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
int main(){
  const int w=9,h=7,c=1; const std::size_t n=static_cast<std::size_t>(w)*h*c;
  std::vector<std::uint8_t> input(n),output(n); for(std::size_t i=0;i<n;++i) input[i]=static_cast<std::uint8_t>((i*29u+7u)%256u);
  augmatch::focus_breathing_u8(input.data(),output.data(),{w,h,c,0.0f,0.8f,0.3f,.5f,.5f,augmatch::Interpolation::Linear,0}); assert(output==input);
  std::vector<std::uint8_t> constant(n,113); augmatch::focus_breathing_u8(constant.data(),output.data(),{w,h,c,1.0f,.2f,.1f,.5f,.5f,augmatch::Interpolation::Linear,0}); assert(output==constant);
  augmatch::focus_breathing_u8(input.data(),output.data(),{w,h,c,1.0f,.2f,.1f,.5f,.5f,augmatch::Interpolation::Nearest,9}); assert(output!=input);
  bool threw=false; try{augmatch::focus_breathing_u8(input.data(),output.data(),{w,h,c,1.0f,.2f,.1f,1.2f,.5f,augmatch::Interpolation::Linear,0});}catch(const std::invalid_argument&){threw=true;} assert(threw);
  threw=false; try{augmatch::focus_breathing_u8(input.data(),output.data(),{w,h,c,1.0f,.2f,.1f,.5f,.5f,augmatch::Interpolation::Area,0});}catch(const std::invalid_argument&){threw=true;} assert(threw);
  return 0;
}
