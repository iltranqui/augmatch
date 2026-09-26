#include "augmatch/filter/filter.hpp"
#include "augmatch/color/color.hpp"
#include "augmatch/geometry/geometric.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
int main(){
  const int w=7,h=5,ch=3; std::vector<std::uint8_t> in(static_cast<std::size_t>(w)*h*ch),out(in.size());
  for(std::size_t i=0;i<in.size();++i) in[i]=static_cast<std::uint8_t>(i%251);
  augmatch::optical_defocus_u8(in.data(),out.data(),{w,h,ch,0,0}); assert(out==in);
  augmatch::optical_motion_blur_u8(in.data(),out.data(),{w,h,ch,1,0,0}); assert(out==in);
  augmatch::optical_zoom_blur_u8(in.data(),out.data(),{w,h,ch,1,1,0}); assert(out==in);
  augmatch::optical_chromatic_aberration_u8(in.data(),out.data(),{w,h,ch,0,0,0,1,1,1}); assert(out==in);
  augmatch::lateral_chromatic_aberration_u8(in.data(),out.data(),{w,h,ch,0,0,0}); assert(out==in);
  augmatch::longitudinal_chromatic_aberration_u8(in.data(),out.data(),{w,h,ch,0,0,0,1}); assert(out==in);
  augmatch::thin_prism_distortion_u8(in.data(),out.data(),{w,h,ch,0,0,0,0}); assert(out==in);
  std::vector<float> row_x(h,0),row_y(h,0);
  augmatch::rolling_shutter_geometric_distortion_u8(in.data(),out.data(),{w,h,ch,row_x.data(),row_y.data()}); assert(out==in);
  return 0;
}
