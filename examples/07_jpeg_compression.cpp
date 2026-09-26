// 07_jpeg_compression — simulate JPEG artifacts at a low quality setting.
//
// What it shows:
//   * Compression artifacts are produced by a real libjpeg encode + decode round trip.
//   * Lower quality -> larger error; measured here as mean absolute difference.
// API used: JpegCompressionConfig, JpegSubsampling, easy::jpeg_compression.
// Build note: only built when CMake finds libjpeg (AUGMATCH_ENABLE_JPEG=ON).
// Output: 07_jpeg_compression.ppm — blocky, colour-bled version of the test image.

#include <cstdlib>
#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"
#include "example_utils.hpp"

// Mean absolute difference between two same-size byte buffers.
static double mean_abs_diff(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  double total = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) total += std::abs(int(a[i]) - int(b[i]));
  return total / static_cast<double>(a.size());
}

int main() {
  const int width = 64, height = 48, channels = 3;  // JPEG accepts 1 (gray) or 3 (RGB) channels
  std::vector<std::uint8_t> input = example::make_test_image(width, height);
  std::vector<std::uint8_t> high(input.size()), low(input.size());
  const auto in = augmatch::make_hwc_u8_view(input.data(), width, height, channels);

  augmatch::JpegCompressionConfig jpeg;
  jpeg.subsampling = augmatch::JpegSubsampling::Y420;  // halve chroma resolution, like most cameras

  jpeg.quality = 95;  // libjpeg quality scale 1..100
  augmatch::Status status = augmatch::easy::jpeg_compression(in, augmatch::make_hwc_u8_view(high.data(), width, height, channels), jpeg);
  if (!status) { std::cerr << "jpeg q95: " << status.message << '\n'; return 1; }

  jpeg.quality = 10;
  status = augmatch::easy::jpeg_compression(in, augmatch::make_hwc_u8_view(low.data(), width, height, channels), jpeg);
  if (!status) { std::cerr << "jpeg q10: " << status.message << '\n'; return 1; }

  const double error_high = mean_abs_diff(input, high), error_low = mean_abs_diff(input, low);
  std::cout << "mean abs error: quality 95 = " << error_high << ", quality 10 = " << error_low << '\n';
  if (!(error_low > error_high)) return 1;  // lower quality must lose more detail

  example::write_ppm("07_jpeg_compression.ppm", low, width, height);
  std::cout << "wrote 07_jpeg_compression.ppm\n";
  return 0;
}
