#include "augmatch/core/convert.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  constexpr int width = 2, height = 2, channels = 3;
  const std::uint8_t host_input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::uint8_t* device_input = nullptr;
  float* device_output = nullptr;
  if (cudaMalloc(&device_input, sizeof(host_input)) != cudaSuccess ||
      cudaMalloc(&device_output, sizeof(float) * width * height * channels) != cudaSuccess)
    return 1;
  if (cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice) != cudaSuccess)
    return 2;
  const augmatch::ToTensorV2Config config{width, height, channels, false};
  augmatch::to_tensor_v2_u8_f32(device_input, device_output, config);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  std::vector<float> output(width * height * channels);
  if (cudaMemcpy(output.data(), device_output, sizeof(float) * output.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 4;
  const float expected[] = {
      1.0f / 255.0f, 4.0f / 255.0f, 7.0f / 255.0f, 10.0f / 255.0f,
      2.0f / 255.0f, 5.0f / 255.0f, 8.0f / 255.0f, 11.0f / 255.0f,
      3.0f / 255.0f, 6.0f / 255.0f, 9.0f / 255.0f, 12.0f / 255.0f};
  for (std::size_t i = 0; i < output.size(); ++i)
    if (std::abs(output[i] - expected[i]) > 1e-7f) return 5;

  const auto input_view = augmatch::make_hwc_u8_view(device_input, width, height, channels,
                                                       augmatch::MemorySpace::Device);
  const auto output_view = augmatch::make_chw_f32_view(device_output, width, height, channels,
                                                        augmatch::MemorySpace::Device);
  augmatch::to_tensor_v2(input_view.as_const(), output_view);
  if (cudaDeviceSynchronize() != cudaSuccess) return 6;
  cudaFree(device_output);
  cudaFree(device_input);
  std::cout << "PASS: CUDA ToTensorV2 HWC uint8 to CHW float32\n";
  return 0;
}
