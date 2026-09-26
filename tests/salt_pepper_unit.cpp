#include "augmatch/arithmetic.hpp"
#include <cstdint>
#include <vector>

int main() {
  constexpr int w = 5, h = 4, c = 3;
  const std::vector<std::uint8_t> input(static_cast<std::size_t>(w) * h * c, 91);
  std::vector<std::uint8_t> output(input.size());
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(w) * h, 0);
  mask[1] = 1; mask[7] = 1; mask[12] = 1;
  augmatch::SaltConfig exact{}; exact.width=w; exact.height=h; exact.channels=c; exact.mask=mask.data();
  augmatch::salt_u8(input.data(), output.data(), exact);
  for (int p = 0; p < w*h; ++p) for (int ch = 0; ch < c; ++ch)
    if (output[p*c+ch] != (mask[p] ? 255 : 91)) return 1;
  augmatch::pepper_u8(input.data(), output.data(), exact);
  for (int p = 0; p < w*h; ++p) for (int ch = 0; ch < c; ++ch)
    if (output[p*c+ch] != (mask[p] ? 0 : 91)) return 2;

  mask.assign(mask.size(), 0); mask[2] = 1; mask[3] = 2;
  exact.mask = mask.data();
  augmatch::salt_and_pepper_u8(input.data(), output.data(), exact);
  for (int p = 0; p < w*h; ++p) for (int ch = 0; ch < c; ++ch) {
    const std::uint8_t expected = mask[p] == 1 ? 255 : mask[p] == 2 ? 0 : 91;
    if (output[p*c+ch] != expected) return 3;
  }

  const augmatch::SaltPepperRectangle rectangles[] = {{1, 1, 4, 3}};
  augmatch::CoarsePepperConfig coarse{}; coarse.width=w; coarse.height=h; coarse.channels=c;
  coarse.rectangles=rectangles; coarse.rectangle_count=1;
  augmatch::coarse_pepper_u8(input.data(), output.data(), coarse);
  for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) for (int ch=0; ch<c; ++ch) {
    const bool selected = x >= 1 && x < 4 && y >= 1 && y < 3;
    if (output[(y*w+x)*c+ch] != (selected ? 0 : 91)) return 4;
  }

  coarse.rectangles=nullptr; coarse.rectangle_count=0; coarse.probability=0.35f;
  coarse.block_width=2; coarse.block_height=2; coarse.seed=77;
  std::vector<std::uint8_t> a(input.size()), b(input.size());
  augmatch::coarse_salt_u8(input.data(), a.data(), coarse);
  augmatch::coarse_salt_u8(input.data(), b.data(), coarse);
  if (a != b) return 5;
  for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) for (int ch=0; ch<c; ++ch) {
    const auto value=a[(y*w+x)*c+ch];
    if (value != 91 && value != 255) return 6;
    if (value != a[(y*w+x)*c]) return 7;
  }
  return 0;
}
