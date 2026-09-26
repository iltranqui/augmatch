#include "augmatch/blend.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
int main(){
  const int w=4,h=3,c=3; std::vector<std::uint8_t> a(w*h*c,0),b(w*h*c,200),o(a.size());
  augmatch::blend_alpha_u8(a.data(),b.data(),o.data(),{w,h,c,.5f}); for(auto x:o) assert(x==100);
  std::vector<std::uint8_t> mask(w*h,0); mask[0]=255; augmatch::blend_alpha_mask_u8(a.data(),b.data(),mask.data(),o.data(),{w,h,c});
  for(int i=0;i<c;++i)assert(o[i]==200); for(std::size_t i=c;i<o.size();++i)assert(o[i]==0);
  std::vector<std::uint8_t> o2(o.size()); augmatch::blend_alpha_simplex_noise_u8(a.data(),b.data(),o.data(),{w,h,c,0,1,2,3,9}); augmatch::blend_alpha_simplex_noise_u8(a.data(),b.data(),o2.data(),{w,h,c,0,1,2,3,9}); assert(o==o2);
  augmatch::blend_alpha_frequency_noise_u8(a.data(),b.data(),o.data(),{w,h,c,0,1,2,1,4});
  std::vector<std::uint8_t> element_mask(w*h,255); augmatch::blend_alpha_elementwise_u8(a.data(),b.data(),element_mask.data(),o.data(),{w,h,c,0,1,nullptr}); for(int i=0;i<c;++i)assert(o[i]==200);
  std::vector<std::uint8_t> colors{0,0,0}; augmatch::blend_alpha_some_colors_u8(a.data(),b.data(),o.data(),{w,h,c,1,0,colors.data(),1}); for(int i=0;i<c;++i)assert(o[i]==200);
  augmatch::blend_alpha_horizontal_linear_gradient_u8(a.data(),b.data(),o.data(),{w,h,c,0,1}); assert(o[0]==0); assert(o[(w-1)*c]==200);
  augmatch::blend_alpha_vertical_linear_gradient_u8(a.data(),b.data(),o.data(),{w,h,c,0,1}); assert(o[0]==0); assert(o[(h-1)*w*c]==200);
  augmatch::blend_alpha_regular_grid_u8(a.data(),b.data(),o.data(),{w,h,c,2,2,1,false}); assert(o[0]==200);
  augmatch::blend_alpha_checkerboard_u8(a.data(),b.data(),o.data(),{w,h,c,2,2,1,false}); assert(o[2*c]==200);
  std::vector<std::int32_t> ids{7}; std::vector<std::uint8_t> seg(w*h,0); seg[2]=7; augmatch::blend_alpha_seg_map_class_ids_u8(a.data(),b.data(),o.data(),{w,h,c,1,seg.data(),ids.data(),1}); assert(o[2*c]==200);
  augmatch::BlendAlphaBoundingBoxesConfig box{w,h,c,1,nullptr,0}; augmatch::blend_alpha_bounding_boxes_u8(a.data(),b.data(),o.data(),box); for(auto x:o)assert(x==0);
  augmatch::BoxXYXY record{1,1,3,3}; box.boxes=&record; box.box_count=1; augmatch::blend_alpha_bounding_boxes_u8(a.data(),b.data(),o.data(),box); assert(o[(w+1)*c]==200);
  return 0;
}
