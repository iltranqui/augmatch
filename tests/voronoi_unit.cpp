#include "augmatch/voronoi.hpp"
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  const int w=9,h=7; std::vector<std::uint8_t> in(static_cast<std::size_t>(w)*h),a(in.size()),b(in.size());
  for(std::size_t i=0;i<in.size();++i) in[i]=static_cast<std::uint8_t>(i%5);
  augmatch::VoronoiConfig v{w,h,7,123,nullptr};
  augmatch::voronoi_u8(in.data(),a.data(),v); augmatch::voronoi_u8(in.data(),b.data(),v); assert(a==b);
  for(auto x:a) assert(x<5); // target labels are sampled, never interpolated.
  std::vector<std::uint32_t> labels(static_cast<std::size_t>(w)*h);
  augmatch::voronoi_labels_u32(labels.data(),v); for(auto x:labels) assert(x<7);
  const augmatch::VoronoiPoint points[]={{0,0},{w-1,h-1}}; std::vector<std::uint8_t> explicit_out(in.size());
  augmatch::voronoi_u8(in.data(),explicit_out.data(),{w,h,2,0,points}); assert(explicit_out[0]==in[0]); assert(explicit_out.back()==in.back());
  std::vector<std::uint8_t> grid(in.size()); augmatch::regular_grid_voronoi_u8(in.data(),grid.data(),{w,h,3,2});
  assert(grid[0]==in[static_cast<std::size_t>(w)+1]); // first 3x3 cell samples its centre (1,1).
  std::vector<std::uint8_t> relative(in.size()); augmatch::relative_regular_grid_voronoi_u8(in.data(),relative.data(),{w,h,.5f,.5f});
  assert(relative.size()==in.size());
  return 0;
}
