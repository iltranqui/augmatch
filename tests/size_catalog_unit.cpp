#include "augmatch/size.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
int main(){
  const int w=7,h=5,c=1; std::vector<std::uint8_t> in(w*h),out(9*6),back(w*h);
  for(int i=0;i<w*h;++i) in[i]=static_cast<std::uint8_t>(i);
  augmatch::MultiplesOfConfig m{w,h,c,4,3,99,augmatch::PadBorder::Constant,augmatch::SizeAnchor::TopLeft};
  auto d=augmatch::make_pad_to_multiples_of_dimensions(m); assert(d.width==8&&d.height==6);
  out.resize(d.width*d.height); augmatch::pad_to_multiples_of_u8(in.data(),out.data(),m); assert(out[0]==0&&out[6]==6&&out[7]==99&&out[8]==7&&out[47]==99);
  m.anchor=augmatch::SizeAnchor::Center; out.assign(8*6,0); augmatch::center_pad_to_multiples_of_u8(in.data(),out.data(),m); assert(out[0]==0&&out[6]==6&&out[7]==99&&out[8]==7&&out[47]==99);
  auto cd=augmatch::make_crop_to_multiples_of_dimensions(m); assert(cd.width==4&&cd.height==3); out.assign(12,0); augmatch::center_crop_to_multiples_of_u8(in.data(),out.data(),m); assert(out[0]==8&&out[3]==11&&out[8]==22);
  augmatch::PowersOfConfig p{w,h,c,2,77,augmatch::PadBorder::Constant,augmatch::SizeAnchor::TopLeft}; assert(augmatch::make_pad_to_powers_of_dimensions(p).width==8); assert(augmatch::make_crop_to_powers_of_dimensions(p).width==4);
  augmatch::AspectRatioConfig a{w,h,c,2.0f,augmatch::SizeRounding::Nearest,55,augmatch::PadBorder::Constant,augmatch::SizeAnchor::TopLeft}; auto ad=augmatch::make_pad_to_aspect_ratio_dimensions(a); assert(ad.width==10&&ad.height==5); out.assign(50,0); augmatch::pad_to_aspect_ratio_u8(in.data(),out.data(),a); assert(out[7]==55);
  augmatch::KeepSizeByResizeConfig k{w,h,c,3,3,augmatch::Interpolation::Nearest}; augmatch::keep_size_by_resize_u8(in.data(),back.data(),k); assert(back.size()==in.size());
  return 0;
}
