#include "augmatch/bayer.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
}

int main() {
  constexpr int w = 8, h = 8;
  std::vector<std::uint8_t> rgb(w * h * 3), rgbw(w * h * 4), out(w * h);
  for (int p = 0; p < w * h; ++p) {
    rgb[p * 3 + 0] = 10; rgb[p * 3 + 1] = 20; rgb[p * 3 + 2] = 30;
    rgbw[p * 4 + 0] = 10; rgbw[p * 4 + 1] = 20; rgbw[p * 4 + 2] = 30; rgbw[p * 4 + 3] = 40;
  }
  for (int p = 0; p < 4; ++p) {
    const auto pattern = static_cast<augmatch::QuadBayerPattern>(p);
    augmatch::quad_bayer_sample_u8(rgb.data(), out.data(), {w, h, pattern});
    const int expected[4][4] = {{0,1,1,2}, {2,1,1,0}, {1,0,2,1}, {1,2,0,1}};
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
      require(out[y*w+x] == static_cast<std::uint8_t>(10 + 10 * expected[p][((y>>1)&1)*2 + ((x>>1)&1)]), "Quad-Bayer vector mismatch");
  }
  const int rgbw_expected[4][4] = {{0,1,3,2}, {0,3,2,1}, {1,2,3,0}, {3,1,0,2}};
  for (int p = 0; p < 4; ++p) {
    augmatch::rgbw_sample_u8(rgbw.data(), out.data(), {w, h, static_cast<augmatch::RGBWPattern>(p)});
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
      require(out[y*w+x] == static_cast<std::uint8_t>(10 * (rgbw_expected[p][(y&1)*2+(x&1)] + 1)), "RGBW vector mismatch");
  }
  const std::uint8_t mask[] = {0, 1, 2, 3, 3, 2};
  std::vector<std::uint8_t> six(6 * 6 * 4), custom(6 * 6);
  for (int p = 0; p < 36; ++p) for (int c = 0; c < 4; ++c) six[p*4+c] = static_cast<std::uint8_t>(c + 1);
  const augmatch::CustomCfaConfig config{6, 6, 4, 3, 2, mask};
  augmatch::custom_cfa_sample_u8(six.data(), custom.data(), config);
  for (int y = 0; y < 6; ++y) for (int x = 0; x < 6; ++x)
    require(custom[y*6+x] == mask[(y&1)*3+(x%3)] + 1, "custom CFA vector mismatch");
  require(augmatch::quad_bayer_channel_at(0, 0, augmatch::QuadBayerPattern::RGGB) == 0, "Quad-Bayer mapping mismatch");
  require(augmatch::rgbw_channel_at(1, 0, augmatch::RGBWPattern::RGWB) == 3, "RGBW mapping mismatch");
  bool rejected = false;
  try { const std::uint8_t bad[] = {0, 4}; augmatch::custom_cfa_sample_u8(six.data(), custom.data(), {2, 1, 4, 2, 1, bad}); } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "custom CFA must reject an out-of-range channel");
  std::cout << "PASS: CFA sampling vectors and mask validation\n";
}
