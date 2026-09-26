#include "augmatch/sensor/noise_view.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  constexpr int width = 2, height = 2, channels = 1;
  const std::uint8_t host_input[] = {0, 64, 128, 255};
  std::uint8_t* device_input = nullptr;
  float* device_output = nullptr;
  cudaStream_t stream = nullptr;
  if (cudaMalloc(&device_input, sizeof(host_input)) != cudaSuccess ||
      cudaMalloc(&device_output, sizeof(float) * width * height * channels) != cudaSuccess ||
      cudaStreamCreate(&stream) != cudaSuccess) return 77;
  cudaMemcpyAsync(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice, stream);
  auto input = augmatch::make_hwc_view(static_cast<const std::uint8_t*>(device_input), width, height,
                                       channels, augmatch::MemorySpace::Device);
  auto output = augmatch::make_hwc_view(device_output, width, height, channels,
                                        augmatch::MemorySpace::Device);
  augmatch::NoiseViewConfig config;
  config.seed = 42;
  config.mean = 0.25f;
  config.stddev = 0.0f;
  const auto status = augmatch::additive_noise_view(input, output, config, {}, stream);
  if (!status || cudaStreamSynchronize(stream) != cudaSuccess) return 77;
  std::vector<float> result(4);
  if (cudaMemcpy(result.data(), device_output, sizeof(float) * result.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 77;
  for (int i = 0; i < 4; ++i) {
    const float expected = std::min(1.0f, static_cast<float>(host_input[i]) / 255.0f + 0.25f);
    if (std::abs(result[i] - expected) > 1e-6f) return 3;
  }
  cudaStreamDestroy(stream);
  cudaFree(device_input);
  cudaFree(device_output);
  return 0;
}
