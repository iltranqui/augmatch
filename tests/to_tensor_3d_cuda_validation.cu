#include "augmatch/core/convert.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  constexpr int width = 2, height = 1, depth = 2, channels = 2;
  const std::uint8_t host_input[] = {1, 2, 3, 4, 5, 6, 7, 8};
  std::uint8_t* device_input = nullptr;
  float* device_output = nullptr;
  if (cudaMalloc(&device_input, sizeof(host_input)) != cudaSuccess ||
      cudaMalloc(&device_output, sizeof(host_input) * sizeof(float)) != cudaSuccess)
    return 1;
  if (cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice) != cudaSuccess)
    return 2;
  const augmatch::ToTensor3DConfig config{width, height, depth, channels, false};
  augmatch::to_tensor_3d_u8_f32(device_input, device_output, config);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  std::vector<float> output(8);
  if (cudaMemcpy(output.data(), device_output, sizeof(float) * output.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 4;
  const int expected[] = {1, 3, 5, 7, 2, 4, 6, 8};
  for (std::size_t i = 0; i < output.size(); ++i)
    if (std::abs(output[i] - expected[i] / 255.0f) > 1e-7f) return 5;
  const augmatch::ToTensor3DConfig normalized{
      width, height, depth, channels, true, 0.1f, 0.2f, 0.3f, 0.5f, 0.25f, 0.2f};
  augmatch::to_tensor_3d_u8_f32(device_input, device_output, normalized);
  if (cudaDeviceSynchronize() != cudaSuccess) return 6;
  if (cudaMemcpy(output.data(), device_output, sizeof(float) * output.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 7;
  if (std::abs(output[0] - ((1.0f / 255.0f - 0.1f) / 0.5f)) > 1e-7f ||
      std::abs(output[4] - ((2.0f / 255.0f - 0.2f) / 0.25f)) > 1e-7f)
    return 8;
  cudaFree(device_output);
  cudaFree(device_input);
  std::cout << "PASS: CUDA ToTensor3D DHWC uint8 to CDHW float32\n";
  return 0;
}
