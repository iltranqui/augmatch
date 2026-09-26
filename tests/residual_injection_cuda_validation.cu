#include "augmatch/isp_artifacts.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;

  const int w = 8, h = 5, c = 3;
  const std::size_t n = static_cast<std::size_t>(w) * h * c;
  std::vector<std::uint8_t> input(n), gpu_output(n);
  for (std::size_t i = 0; i < n; ++i) input[i] = static_cast<std::uint8_t>((i * 19) % 256);

  std::uint8_t *device_input = nullptr, *device_output = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&device_input), n), "cudaMalloc(device_input)");
  check(cudaMalloc(reinterpret_cast<void**>(&device_output), n), "cudaMalloc(device_output)");
  try {
    check(cudaMemcpy(device_input, input.data(), n, cudaMemcpyHostToDevice), "copy input to CUDA");

    const augmatch::LaplacianResidualInjectionConfig lap{w, h, c, nullptr, 1, .25f};
    augmatch::laplacian_residual_injection_u8(device_input, device_output, lap);
    check(cudaDeviceSynchronize(), "synchronize Laplacian residual injection");
    check(cudaMemcpy(gpu_output.data(), device_output, n, cudaMemcpyDeviceToHost), "copy Laplacian output from CUDA");

    const augmatch::SobelResidualInjectionConfig sobel{w, h, c, nullptr, 1, .25f, 1};
    augmatch::sobel_residual_injection_u8(device_input, device_output, sobel);
    check(cudaDeviceSynchronize(), "synchronize Sobel residual injection");
    check(cudaMemcpy(gpu_output.data(), device_output, n, cudaMemcpyDeviceToHost), "copy Sobel output from CUDA");

    const augmatch::HighPassResidualInjectionConfig highpass{w, h, c, nullptr, 1, .25f};
    augmatch::high_pass_residual_injection_u8(device_input, device_output, highpass);
    check(cudaDeviceSynchronize(), "synchronize high-pass residual injection");
    check(cudaMemcpy(gpu_output.data(), device_output, n, cudaMemcpyDeviceToHost), "copy high-pass output from CUDA");
  } catch (...) {
    cudaFree(device_input);
    cudaFree(device_output);
    throw;
  }
  check(cudaFree(device_input), "cudaFree(device_input)");
  check(cudaFree(device_output), "cudaFree(device_output)");
  std::cout << "Residual injection CUDA validation passed\n";
}
