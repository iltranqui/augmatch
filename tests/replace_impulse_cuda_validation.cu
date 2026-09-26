#include "augmatch/arithmetic.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  constexpr int width = 4, height = 2, channels = 1;
  const std::vector<std::uint8_t> input{10, 20, 30, 40, 50, 60, 70, 80};
  const std::vector<std::uint8_t> mask{0, 1, 0, 1, 1, 0, 1, 0};
  const std::vector<std::uint8_t> values{101, 102, 103, 104, 105, 106, 107, 108};
  std::uint8_t *di = nullptr, *doo = nullptr, *dm = nullptr, *dv = nullptr;
  if (cudaMalloc(&di, input.size()) != cudaSuccess || cudaMalloc(&doo, input.size()) != cudaSuccess ||
      cudaMalloc(&dm, mask.size()) != cudaSuccess || cudaMalloc(&dv, values.size()) != cudaSuccess) return 1;
  if (cudaMemcpy(di, input.data(), input.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dm, mask.data(), mask.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dv, values.data(), values.size(), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
  augmatch::replace_elementwise_u8(di, doo,
                                   {width, height, channels, dm, dv, 0, 0.0f, 0});
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  std::vector<std::uint8_t> output(input.size());
  if (cudaMemcpy(output.data(), doo, output.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
      output != std::vector<std::uint8_t>({10, 102, 30, 104, 105, 60, 107, 80})) return 4;
  augmatch::impulse_noise_u8(di, doo,
                             {width, height, channels, dm, dv, 0.0f, 0.5f, 0});
  if (cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(output.data(), doo, output.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
      output != std::vector<std::uint8_t>({10, 102, 30, 104, 105, 60, 107, 80})) return 5;
  augmatch::impulse_noise_u8(di, doo,
                             {width, height, channels, nullptr, nullptr, 1.0f, 1.0f, 77});
  if (cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(output.data(), doo, output.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
      output != std::vector<std::uint8_t>(8, 255)) return 6;
  cudaFree(dv);
  cudaFree(dm);
  cudaFree(doo);
  cudaFree(di);
  return 0;
}
