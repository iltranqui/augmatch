#include "augmatch/geometry/mixing.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
int main(){
  const int w=4,h=2,c=1; std::vector<std::uint8_t>a(w*h,10),b(w*h,110),o(w*h);
  augmatch::ImageSourceU8 sa{a.data(),w,h,c}, sb{b.data(),w,h,c};
  augmatch::mixup_u8(sa,sb,o.data(),{w,h,c,.25f,0,false}); for(auto x:o) assert(x==35);
  augmatch::CutMixConfig cc{w,h,c,{1,0,3,2},0,false,.5f}; augmatch::cutmix_u8(sa,sb,o.data(),cc); assert(o[0]==10&&o[1]==110&&o[2]==110&&o[3]==10);
  std::vector<augmatch::ImageSourceU8> four{sa,sb,sa,sb}; augmatch::mosaic_u8({four.data(),four.size()},o.data(),{w,h,c,2,1,0}); assert(o[0]==10&&o[2]==110&&o[w]==10&&o[w+2]==110);
  std::vector<float> weights{1.f}; augmatch::ImageBatchU8 templ{&sb,1}; augmatch::template_transform_u8(sa,templ,o.data(),{w,h,c,1.f,weights.data(),1}); for(auto x:o) assert(x==60);
  augmatch::OverlayElementU8 element{sb,1,0,1.f,nullptr}; augmatch::overlay_elements_u8(sa,o.data(),{w,h,c,&element,1}); assert(o[0]==10&&o[1]==110&&o[2]==110&&o[3]==10);
  augmatch::BoxXYXY boxes_a[]={{0,0,1,1}},boxes_b[]={{1,1,3,2}},out_boxes[2]{}; std::int32_t labels[2]{}; augmatch::MixingTargets mt{out_boxes,labels,2,0}; augmatch::mixup_targets({boxes_a,nullptr,1},{boxes_b,nullptr,1},.5f,mt); assert(mt.count==2);
  return 0;
}
