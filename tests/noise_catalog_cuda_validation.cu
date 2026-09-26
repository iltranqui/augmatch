#include "augmatch/noise.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  constexpr int width = 256, height = 256, channels = 1;
  constexpr std::size_t count = static_cast<std::size_t>(width) * height * channels;
  std::vector<std::uint8_t> host_input(count, 128), first(count), second(count), changed(count);
  std::uint8_t* device_input = nullptr;
  std::uint8_t* device_output = nullptr;
  if (cudaMalloc(&device_input, count) != cudaSuccess || cudaMalloc(&device_output, count) != cudaSuccess) return 77;
  const auto h2d_start = std::chrono::steady_clock::now();
  if (cudaMemcpy(device_input, host_input.data(), count, cudaMemcpyHostToDevice) != cudaSuccess) return 77;
  const double h2d_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h2d_start).count();
  augmatch::ShotNoiseConfig config;
  config.width = width; config.height = height; config.channels = channels;
  config.scale = 64.0f; config.gain = 1.0f; config.seed = 0x9911;
  double d2h_ms = 0.0;
  try {
    augmatch::shot_noise_u8(device_input, device_output, config);
    if (cudaDeviceSynchronize() != cudaSuccess) return 77;
    const auto d2h_start = std::chrono::steady_clock::now();
    if (cudaMemcpy(first.data(), device_output, count, cudaMemcpyDeviceToHost) != cudaSuccess) return 77;
    d2h_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d2h_start).count();
    augmatch::shot_noise_u8(device_input, device_output, config);
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(second.data(), device_output, count, cudaMemcpyDeviceToHost) != cudaSuccess) return 77;
    config.seed++;
    augmatch::shot_noise_u8(device_input, device_output, config);
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(changed.data(), device_output, count, cudaMemcpyDeviceToHost) != cudaSuccess) return 77;
  } catch (const std::exception& error) {
    std::cerr << "CUDA runtime blocker: " << error.what() << '\n';
    cudaFree(device_input);
    cudaFree(device_output);
    return 77;
  }

  double mean = 0.0;
  for (std::uint8_t value : first) mean += static_cast<double>(value) / 255.0;
  mean /= static_cast<double>(count);
  double variance = 0.0;
  for (std::uint8_t value : first) {
    const double d = static_cast<double>(value) / 255.0 - mean;
    variance += d * d;
  }
  variance /= static_cast<double>(count - 1);
  const double expected_mean = 128.0 / 255.0;
  const double expected_variance = (128.0 / 255.0) / 64.0;
  std::size_t changed_count = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (first[i] != second[i]) return 3;
    if (first[i] != changed[i]) ++changed_count;
  }
  cudaFree(device_input);
  cudaFree(device_output);
  if (std::abs(mean - expected_mean) > 0.015 || std::abs(variance - expected_variance) > 0.003 || changed_count == 0) return 4;
  std::cout << "CUDA_TRANSFER bytes=" << count << " h2d_ms=" << h2d_ms << " d2h_ms=" << d2h_ms << '\n';
  std::cout << "PASS: CUDA shot-noise statistics, fixed-seed determinism, and independent seeds\n";
  return 0;
}
