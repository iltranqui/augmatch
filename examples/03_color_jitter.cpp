// 03_color_jitter — change brightness, contrast, saturation, and hue in one call.
//
// What it shows:
//   * Transforms are configured with a plain struct; only set the fields you need.
//   * The config's width/height/channels are filled in for you by easy::.
//   * easy::apply runs *any* raw `op_u8(input, output, config)` on views (here: RGB shift).
// API used: ColorJitterConfig, easy::color_jitter, RGBShiftConfig, easy::apply.
// Output: 03_color_jitter.ppm — a brighter, more saturated, hue-shifted image with a red cast.

#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"
#include "example_utils.hpp"

int main() {
  const int width = 64, height = 48, channels = 3;  // ColorJitter needs at least 3 channels
  std::vector<std::uint8_t> input = example::make_test_image(width, height);
  std::vector<std::uint8_t> jittered(input.size()), shifted(input.size());

  const auto in = augmatch::make_hwc_u8_view(input.data(), width, height, channels);
  const auto jitter_out = augmatch::make_hwc_u8_view(jittered.data(), width, height, channels);
  const auto shift_out = augmatch::make_hwc_u8_view(shifted.data(), width, height, channels);

  // Deterministic jitter: every value is explicit, so the result is reproducible.
  augmatch::ColorJitterConfig jitter;
  jitter.brightness = 0.10f;  // additive: +10% of 255
  jitter.contrast = 1.20f;    // factor: 1.0 = unchanged
  jitter.saturation = 1.50f;  // factor: 1.0 = unchanged, 0.0 = grayscale
  jitter.hue = 10.0f;         // additive, OpenCV half-degrees (180 = full turn)

  augmatch::Status status = augmatch::easy::color_jitter(in, jitter_out, jitter);
  if (!status) { std::cerr << "color_jitter: " << status.message << '\n'; return 1; }

  // Any raw op whose config starts with width/height/channels works through easy::apply.
  augmatch::RGBShiftConfig shift;
  shift.shift0 = 30;          // add 30 to red
  shift.shift2 = -30;         // subtract 30 from blue
  status = augmatch::easy::apply(augmatch::rgb_shift_u8, jitter_out.as_const(), shift_out, shift);
  if (!status) { std::cerr << "rgb_shift: " << status.message << '\n'; return 1; }

  example::write_ppm("03_color_jitter.ppm", shifted, width, height);
  std::cout << "wrote 03_color_jitter.ppm\n";
  return 0;
}
