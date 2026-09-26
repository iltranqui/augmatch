#include "augmatch/arithmetic.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  constexpr int width = 3, height = 2, channels = 1;
  const std::vector<std::uint8_t> input{0, 10, 100, 200, 255, 50};
  const std::vector<float> offsets{1.5f, -20.0f, 2.5f, 100.0f, -300.0f, 0.0f};
  const std::vector<std::uint8_t> expected_add{2, 0, 103, 255, 0, 50};
  std::uint8_t *device_input = nullptr, *device_output = nullptr;
  float *device_values = nullptr;
  if (cudaMalloc(&device_input, input.size()) != cudaSuccess ||
      cudaMalloc(&device_output, input.size()) != cudaSuccess ||
      cudaMalloc(&device_values, offsets.size() * sizeof(float)) != cudaSuccess)
    return 1;
  if (cudaMemcpy(device_input, input.data(), input.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(device_values, offsets.data(), offsets.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return 2;
  augmatch::add_elementwise_u8(device_input, device_output,
                               {width, height, channels, device_values, 0.0f,
                                0.0f, 0});
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  std::vector<std::uint8_t> output(input.size());
  if (cudaMemcpy(output.data(), device_output, output.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
      output != expected_add)
    return 4;
  const std::vector<float> factors{1.5f, 0.5f, 2.5f, 0.0f, 0.25f, 3.0f};
  const std::vector<std::uint8_t> expected_multiply{0, 5, 250, 0, 64, 150};
  if (cudaMemcpy(device_values, factors.data(), factors.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 5;
  augmatch::multiply_elementwise_u8(device_input, device_output,
                                    {width, height, channels, device_values, 0.0f,
                                     0.0f, 91});
  if (cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(output.data(), device_output, output.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
      output != expected_multiply) return 6;
  std::vector<float> generated(input.size());
  augmatch::make_add_elementwise_values(generated.data(), width, height, channels,
                                        -12.0f, 18.0f, 77);
  cudaFree(device_values);
  cudaFree(device_output);
  cudaFree(device_input);
  return 0;
}
