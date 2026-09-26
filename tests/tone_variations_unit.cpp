#include "augmatch/color/tone.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
int main(){
  const int w=8,h=1,c=2; std::vector<std::uint8_t> in={0,16,32,64,96,128,192,255,255,192,128,96,64,32,16,0},out(in.size()),again(in.size());
  augmatch::gamma_variation_u8(in.data(),out.data(),{w,h,c,1.0f}); assert(out==in);
  std::vector<std::uint8_t> sample={0,128,255},sample_out(3); augmatch::gamma_variation_u8(sample.data(),sample_out.data(),{3,1,1,2.0f}); assert(sample_out[0]==0&&sample_out[1]==64&&sample_out[2]==255);
  augmatch::s_curve_contrast_variation_u8(in.data(),out.data(),{w,h,c,0.0f}); assert(out==in);
  augmatch::highlight_rolloff_variation_u8(in.data(),out.data(),{w,h,c,0.8f,0.0f}); assert(out==in);
  augmatch::shadow_lift_u8(in.data(),out.data(),{w,h,c,0.0f,0.25f}); assert(out==in);
  augmatch::shadow_crush_u8(in.data(),out.data(),{w,h,c,0.0f,0.25f}); assert(out==in);
  augmatch::posterization_u8(in.data(),out.data(),{w,h,c,4}); for(auto v:out)assert(v%16==0);
  augmatch::low_bit_depth_banding_u8(in.data(),again.data(),{w,h,c,4}); assert(out==again);
  std::vector<std::uint8_t> lut(c*256); for(int ch=0;ch<c;++ch)for(int x=0;x<256;++x)lut[ch*256+x]=static_cast<std::uint8_t>(255-x+ch);
  augmatch::tone_curve_variation_u8(in.data(),out.data(),{w,h,c,lut.data()}); for(int i=0;i<w*h*c;++i)assert(out[i]==lut[(i%c)*256+in[i]]);
  return 0;
}
