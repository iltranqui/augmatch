#include "augmatch/isp/isp_artifacts.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

int main() {
  int count = 32 * 24;
  std::vector<std::uint8_t> input(count), output(count), repeat(count), cpu(count);
  for (int y = 0; y < 24; ++y) for (int x = 0; x < 32; ++x)
    input[y * 32 + x] = static_cast<std::uint8_t>(x < 16 ? 45 : 210);
  std::uint8_t *device_input = nullptr, *device_output = nullptr;
  if (cudaMalloc(&device_input, count) != cudaSuccess || cudaMalloc(&device_output, count) != cudaSuccess) return 77;
  if (cudaMemcpy(device_input, input.data(), count, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
  augmatch::GradientPlusLaplacianNoiseConfig config{32, 24, 1, 4.0f, 1.0f, 1.0f, 83};
  augmatch::gradient_plus_laplacian_noise_u8(device_input, device_output, config);
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(output.data(), device_output, count, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
  augmatch::gradient_plus_laplacian_noise_u8(device_input, device_output, config);
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(repeat.data(), device_output, count, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
  if (output != repeat) return 1;
  config.sigma = 0.0f;
  config.gradient_scale = config.laplacian_scale = 0.0f;
  augmatch::gradient_plus_laplacian_noise_u8(device_input, device_output, config);
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(output.data(), device_output, count, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
  if (output != input) return 1;
  cudaFree(device_input);
  cudaFree(device_output);
  return 0;
}
