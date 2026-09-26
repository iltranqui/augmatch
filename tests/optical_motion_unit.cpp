#include "augmatch/filter.hpp"
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
int main(){
  const int w=7,h=5,c=1; std::vector<std::uint8_t> in(w*h),out(w*h),ref;
  for(int y=0;y<h;++y)for(int x=0;x<w;++x)in[y*w+x]=static_cast<std::uint8_t>(y*30+x);
  std::vector<float> depth(w*h,0.5f);
  augmatch::depth_dependent_defocus_u8(in.data(),out.data(),{w,h,c,depth.data(),0.5f,10.0f,8}); assert(out==in);
  float zero[1]={0}; augmatch::camera_shake_blur_u8(in.data(),out.data(),{w,h,c,1,zero,zero}); assert(out==in);
  augmatch::linear_directional_blur_u8(in.data(),out.data(),{w,h,c,1,10.0f,0.0f}); assert(out==in);
  augmatch::rotational_motion_blur_u8(in.data(),out.data(),{w,h,c,1,3.0f,2.0f,90.0f}); assert(out==in);
  float rows[5]={0,0,0,0,0}; augmatch::rolling_shutter_motion_blur_u8(in.data(),out.data(),{w,h,c,1,rows,rows}); assert(out==in);
  bool threw=false; try { augmatch::linear_directional_blur_u8(in.data(),out.data(),{w,h,c,0,1.0f,0.0f}); } catch(const std::invalid_argument&) { threw=true; } assert(threw);
  return 0;
}
