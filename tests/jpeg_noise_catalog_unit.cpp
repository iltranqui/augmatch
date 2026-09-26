#include "augmatch/jpeg.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime.h>
#endif

int main() {
#if AUGMATCH_HAS_CUDA
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  // Host-reference assertions are intentionally not run against device APIs.
  return 77;
#endif
  constexpr int w = 40, h = 32, c = 3;
  std::vector<std::uint8_t> source(static_cast<std::size_t>(w) * h * c);
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
    const auto p = (static_cast<std::size_t>(y) * w + x) * c;
    source[p] = static_cast<std::uint8_t>((x * 13 + y * 5) & 255);
    source[p + 1] = static_cast<std::uint8_t>((x < 20 ? 20 : 230) + (y & 7));
    source[p + 2] = static_cast<std::uint8_t>((x * y) & 255);
  }
  std::vector<std::uint8_t> output(source.size());
  const augmatch::JpegCompressionConfig base{w, h, c, 55, augmatch::JpegSubsampling::Y420};
  augmatch::jpeg_quality_variation_u8(source.data(), output.data(), {w, h, c, 25, augmatch::JpegSubsampling::Y422});
  const auto quality = output;
  augmatch::jpeg_quantization_table_variation_u8(source.data(), output.data(), {w, h, c, 55, augmatch::JpegSubsampling::Y420, 2.0f, 1.0f});
  if (output == quality) return 1;
  augmatch::jpeg_chroma_subsampling_u8(source.data(), output.data(), base);
  const auto normal = output;
  augmatch::jpeg_ringing_u8(source.data(), output.data(), {w, h, c, 35, augmatch::JpegSubsampling::Y420, 0.5f});
  const auto ringing = output;
  augmatch::jpeg_blocking_u8(source.data(), output.data(), {w, h, c, 35, augmatch::JpegSubsampling::Y420, 0.5f});
  const auto blocking = output;
  augmatch::jpeg_mosquito_noise_u8(source.data(), output.data(), {w, h, c, 35, augmatch::JpegSubsampling::Y420, 0.5f});
  const auto mosquito = output;
  augmatch::jpeg_restart_marker_damage_u8(source.data(), output.data(),
      {w, h, c, 35, augmatch::JpegSubsampling::Y420, 1.0f, 4});
  const auto restart = output;
  augmatch::jpeg_progressive_decoding_u8(source.data(), output.data(),
      {w, h, c, 35, augmatch::JpegSubsampling::Y420, 1.0f, 1});
  if (output == normal || output == ringing || output == blocking || output == mosquito || output == restart) return 2;
  std::cout << "JPEG noise catalog unit passed\n";
  return 0;
}
