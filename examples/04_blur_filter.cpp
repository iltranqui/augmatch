// 04_blur_filter — Gaussian and box blur.
//
// What it shows:
//   * Named convenience functions take the few parameters that matter (kernel, sigma).
//   * Invalid parameters come back as a Status, not a crash.
// API used: easy::gaussian_blur, easy::box_blur.
// Output: 04_blur_filter.ppm — the checkerboard softened by a 7x7 Gaussian.

#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"
#include "example_utils.hpp"

int main() {
  const int width = 64, height = 48, channels = 3;
  std::vector<std::uint8_t> input = example::make_test_image(width, height);
  std::vector<std::uint8_t> gaussian(input.size()), box(input.size());

  const auto in = augmatch::make_hwc_u8_view(input.data(), width, height, channels);
  const auto gaussian_out = augmatch::make_hwc_u8_view(gaussian.data(), width, height, channels);
  const auto box_out = augmatch::make_hwc_u8_view(box.data(), width, height, channels);

  // Gaussian blur: odd kernel size (3, 5, 7, ...) and the standard deviation in pixels.
  augmatch::Status status = augmatch::easy::gaussian_blur(in, gaussian_out, /*kernel_size=*/7, /*sigma=*/2.0f);
  if (!status) { std::cerr << "gaussian_blur: " << status.message << '\n'; return 1; }

  // Box blur: every pixel becomes the plain average of its 5x5 neighbourhood.
  status = augmatch::easy::box_blur(in, box_out, /*kernel_size=*/5);
  if (!status) { std::cerr << "box_blur: " << status.message << '\n'; return 1; }

  // Blurring must reduce the sharp checkerboard edges: count changed pixels as a quick check.
  std::size_t changed = 0;
  for (std::size_t i = 0; i < input.size(); ++i) changed += (input[i] != gaussian[i]);
  std::cout << "gaussian blur changed " << changed << " of " << input.size() << " samples\n";
  if (changed == 0) return 1;

  example::write_ppm("04_blur_filter.ppm", gaussian, width, height);
  std::cout << "wrote 04_blur_filter.ppm\n";
  return 0;
}
