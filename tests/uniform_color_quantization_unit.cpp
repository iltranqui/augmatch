#include "augmatch/color/color.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>

int main(){
  const std::vector<std::uint8_t> input={0,31,32,63,64,95,127,128,191,255,77,201};
  std::vector<std::uint8_t> output(input.size());
  augmatch::uniform_color_quantization_u8(input.data(),output.data(),{4,1,3,5});
  const std::vector<std::uint8_t> expected={26,26,26,77,77,77,128,128,179,230,77,179};
  if(output!=expected)return 1;
  augmatch::uniform_color_quantization_to_n_bits_u8(input.data(),output.data(),{4,1,3,3});
  const std::vector<std::uint8_t> expected_bits={0,0,32,32,64,64,96,128,160,224,64,192};
  if(output!=expected_bits)return 2;
  bool rejected=false;
  try{augmatch::uniform_color_quantization_u8(input.data(),output.data(),{4,1,3,1});}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)return 3;
  rejected=false;
  try{augmatch::uniform_color_quantization_to_n_bits_u8(input.data(),output.data(),{4,1,3,9});}catch(const std::invalid_argument&){rejected=true;}
  return rejected?0:4;
}
