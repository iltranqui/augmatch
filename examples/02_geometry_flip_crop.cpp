// 02_geometry_flip_crop — crop a region, then flip it vertically.
//
// What it shows:
//   * The *output view's size* decides the crop size; (x, y) is the top-left corner.
//   * Chaining two operations: the output of one step is the input of the next.
// API used: easy::crop, easy::flip_vertical.
// Output: 02_geometry_flip_crop.ppm — a 32x24 patch from the centre, upside down.

#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"
#include "example_utils.hpp"

int main() {
  const int width = 64, height = 48, channels = 3;        // source image size
  const int crop_w = 32, crop_h = 24;                     // size of the region we keep
  const int crop_x = (width - crop_w) / 2;                // centre the crop horizontally
  const int crop_y = (height - crop_h) / 2;               // ... and vertically

  std::vector<std::uint8_t> image = example::make_test_image(width, height);
  std::vector<std::uint8_t> cropped(static_cast<std::size_t>(crop_w) * crop_h * channels);
  std::vector<std::uint8_t> flipped(cropped.size());      // same size as the crop

  const auto source = augmatch::make_hwc_u8_view(image.data(), width, height, channels);
  const auto crop_view = augmatch::make_hwc_u8_view(cropped.data(), crop_w, crop_h, channels);
  const auto flip_view = augmatch::make_hwc_u8_view(flipped.data(), crop_w, crop_h, channels);

  // Step 1: crop. The rectangle must fit inside the source, otherwise Status is InvalidArgument.
  augmatch::Status status = augmatch::easy::crop(source, crop_view, crop_x, crop_y);
  if (!status) { std::cerr << "crop: " << status.message << '\n'; return 1; }

  // Step 2: flip the crop. `as_const()` turns the writable crop view into a read-only input view.
  status = augmatch::easy::flip_vertical(crop_view.as_const(), flip_view);
  if (!status) { std::cerr << "flip: " << status.message << '\n'; return 1; }

  // Sanity check: an out-of-bounds crop is rejected instead of reading past the buffer.
  const augmatch::Status bad = augmatch::easy::crop(source, crop_view, width, 0);
  if (bad) { std::cerr << "out-of-bounds crop was accepted\n"; return 1; }
  std::cout << "out-of-bounds crop rejected: " << bad.message << '\n';

  example::write_ppm("02_geometry_flip_crop.ppm", flipped, crop_w, crop_h);
  std::cout << "wrote 02_geometry_flip_crop.ppm (" << crop_w << "x" << crop_h << ")\n";
  return 0;
}
