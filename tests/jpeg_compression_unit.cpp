#include "augmatch/jpeg.hpp"
#include <algorithm>
#include <cstdint>
#include <exception>
#include <cstdlib>
#include <iostream>
#include <vector>
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime.h>
#endif

int main() {
#if AUGMATCH_HAS_CUDA
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  // The CPU unit uses host pointers; CUDA pointer ownership is covered by the
  // dedicated validation executable below.
  return 77;
#endif
  constexpr int width = 32;
  constexpr int height = 24;
  std::vector<std::uint8_t> input(static_cast<std::size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    const std::size_t p = (static_cast<std::size_t>(y) * width + x) * 3;
    input[p] = static_cast<std::uint8_t>((x * 9 + y * 3) & 255);
    input[p + 1] = static_cast<std::uint8_t>((y * 11 + x) & 255);
    input[p + 2] = static_cast<std::uint8_t>((x * 5 + y * 7) & 255);
  }
  const augmatch::JpegCompressionConfig config{width, height, 3, 85, augmatch::JpegSubsampling::Y422};
  const auto encoded = augmatch::jpeg_encode_u8(input.data(), config);
  if (encoded.size() < 4 || encoded[0] != 0xff || encoded[1] != 0xd8)
    return 1;
  std::vector<std::uint8_t> decoded(input.size());
  augmatch::jpeg_decode_u8(encoded.data(), encoded.size(), decoded.data(), {width, height, 3});
  int maximum_error = 0;
  std::uint64_t total_error = 0;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const int error = std::abs(static_cast<int>(input[i]) - static_cast<int>(decoded[i]));
    maximum_error = std::max(maximum_error, error);
    total_error += static_cast<std::uint64_t>(error);
  }
  if (maximum_error > 200 || total_error > input.size() * 25) return 2;
  const auto full_chroma = augmatch::jpeg_encode_u8(input.data(), {width, height, 3, 85, augmatch::JpegSubsampling::Y444});
  if (full_chroma == encoded) return 5;
  std::vector<std::uint8_t> roundtrip(input.size());
  augmatch::image_compression_u8(input.data(), roundtrip.data(), config);
  if (roundtrip != decoded) return 3;
  bool rejected = false;
  try { augmatch::jpeg_compression_u8(input.data(), roundtrip.data(), {width, height, 3, 0, augmatch::JpegSubsampling::Y420}); }
  catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) return 4;
  std::cout << "jpeg compression unit passed (max error " << maximum_error << ")\n";
  return 0;
}
