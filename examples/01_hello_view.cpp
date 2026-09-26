// 01_hello_view — the smallest complete augmatch program.
//
// What it shows:
//   * augmatch never owns pixels: you keep a std::vector and wrap it in a *view*.
//   * An ImageView is read-only input, a MutableImageView is writable output.
//   * Every easy:: call returns a Status; check it with `if (!status)`.
// API used: make_hwc_u8_view, easy::flip_horizontal, Status.
// Output: 01_hello_view.ppm — the test image mirrored left-to-right.

#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"  // one include gives the whole library
#include "example_utils.hpp"      // make_test_image / write_ppm helpers

int main() {
  const int width = 64, height = 48, channels = 3;  // a small RGB image

  // 1. Your data: interleaved HWC bytes (R,G,B,R,G,B,...), row after row.
  const std::vector<std::uint8_t> input = example::make_test_image(width, height);  // const: read-only input
  std::vector<std::uint8_t> output(input.size());  // same size, filled by augmatch

  // 2. Wrap both buffers in views. No copy happens; the view just describes the memory.
  //    A const pointer gives an ImageView, a non-const pointer gives a MutableImageView.
  //    (easy:: functions accept either kind as input, so you rarely need to care.)
  const augmatch::ImageView in = augmatch::make_hwc_u8_view(input.data(), width, height, channels);
  const augmatch::MutableImageView out = augmatch::make_hwc_u8_view(output.data(), width, height, channels);

  // 3. Run one augmentation. The easy:: wrappers read width/height/channels from the views.
  const augmatch::Status status = augmatch::easy::flip_horizontal(in, out);
  if (!status) {                                  // Status converts to false on failure
    std::cerr << "flip failed: " << status.message << '\n';
    return 1;
  }

  // 4. Check the result: the first pixel of the output is the last pixel of row 0 in the input.
  const std::size_t last_in_row = static_cast<std::size_t>(width - 1) * channels;
  if (output[0] != input[last_in_row]) {
    std::cerr << "unexpected pixel value\n";
    return 1;
  }

  example::write_ppm("01_hello_view.ppm", output, width, height);
  std::cout << "wrote 01_hello_view.ppm (horizontally flipped)\n";
  return 0;
}
