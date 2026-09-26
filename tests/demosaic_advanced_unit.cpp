#include "augmatch/sensor/bayer.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace augmatch;
int main() {
  constexpr int w = 9, h = 8;
  std::vector<std::uint8_t> raw(w*h), a(w*h*3), b(w*h*3), wrapper(w*h*3);
  for (int i=0; i<w*h; ++i) raw[i] = static_cast<std::uint8_t>((i*37 + i/w*11) & 255);
  for (int p=0; p<4; ++p) {
    BayerConfig c{w,h,static_cast<BayerPattern>(p)};
    bayer_demosaic_malvar_he_cutler_u8(raw.data(),a.data(),c);
    bayer_demosaic_malvar_he_cutler_u8(raw.data(),b.data(),c);
    assert(a == b); // deterministic reference/property: repeated calls are exact.
    bayer_demosaic_edge_aware_u8(raw.data(),a.data(),c);
    bayer_demosaic_edge_aware_u8(raw.data(),b.data(),c);
    assert(a == b);
    const int tiles[4][2][2] = {{{0,1},{1,2}}, {{2,1},{1,0}},
                                 {{1,0},{2,1}}, {{1,2},{0,1}}};
    for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) {
      const int i=(y*w+x)*3, ch=tiles[p][y&1][x&1];
      assert(a[i+ch] == raw[y*w+x]);
    }
  }
  std::vector<std::uint8_t> constant(w*h,73), expected(w*h*3,73);
  for (int p=0; p<4; ++p) {
    BayerConfig c{w,h,static_cast<BayerPattern>(p)};
    bayer_demosaic_malvar_he_cutler_u8(constant.data(),a.data(),c);
    bayer_demosaic_edge_aware_u8(constant.data(),b.data(),c);
    assert(a == expected && b == expected);
  }
  bayer_demosaic_malvar_he_cutler_rggb_u8(raw.data(),wrapper.data(),w,h);
  bayer_demosaic_malvar_he_cutler_u8(raw.data(),a.data(),BayerConfig{w,h,BayerPattern::RGGB});
  assert(wrapper == a);
  bayer_demosaic_edge_aware_gbrg_u8(raw.data(),wrapper.data(),w,h);
  bayer_demosaic_edge_aware_u8(raw.data(),a.data(),BayerConfig{w,h,BayerPattern::GBRG});
  assert(wrapper == a);
  std::vector<std::uint8_t> one(1,255), one_out(3);
  bayer_demosaic_malvar_he_cutler_u8(one.data(),one_out.data(),BayerConfig{1,1,BayerPattern::RGGB});
  assert(one_out[0] == 255 && one_out[1] == 255 && one_out[2] == 255);
  return 0;
}
