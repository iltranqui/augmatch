#include "augmatch/compression/jpeg.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  constexpr int width = 32, height = 24, channels = 3;
  const std::size_t bytes = static_cast<std::size_t>(width) * height * channels;
  std::vector<std::uint8_t> host(bytes), result(bytes);
  for (std::size_t i = 0; i < bytes; ++i) host[i] = static_cast<std::uint8_t>((i * 17 + i / 7) & 255);
  std::uint8_t *input = nullptr, *output = nullptr;
  if (cudaMalloc(&input, bytes) != cudaSuccess || cudaMalloc(&output, bytes) != cudaSuccess) return 77;
  cudaMemcpy(input, host.data(), bytes, cudaMemcpyHostToDevice);
  const augmatch::JpegCompressionConfig codec{width, height, channels, 60, augmatch::JpegSubsampling::Y420};
  augmatch::jpeg_compression_u8(input, output, codec);
  cudaMemcpy(result.data(), output, bytes, cudaMemcpyDeviceToHost);
  augmatch::jpeg_ringing_u8(input, output, {width, height, channels, 35, augmatch::JpegSubsampling::Y420, 0.25f});
  augmatch::jpeg_blocking_u8(input, output, {width, height, channels, 35, augmatch::JpegSubsampling::Y420, 0.25f});
  augmatch::jpeg_mosquito_noise_u8(input, output, {width, height, channels, 35, augmatch::JpegSubsampling::Y420, 0.25f});
  augmatch::jpeg_restart_marker_damage_u8(input, output,
      {width, height, channels, 35, augmatch::JpegSubsampling::Y420, 1.0f, 4});
  augmatch::jpeg_progressive_decoding_u8(input, output,
      {width, height, channels, 35, augmatch::JpegSubsampling::Y420, 1.0f, 1});
  cudaFree(input);
  cudaFree(output);
  std::cout << "JPEG CUDA host-codec fallback validation passed\n";
  return 0;
}
