#include "augmatch/color/tone.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
std::uint8_t quantize(float value) {
  return static_cast<std::uint8_t>(std::floor(std::max(0.0f, std::min(255.0f, value)) + 0.5f));
}
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    const int width = 23, height = 5, channels = 4;
    const std::size_t count = static_cast<std::size_t>(width) * height * channels;
    std::vector<std::uint8_t> input(count), actual(count), expected(count);
    for (std::size_t i = 0; i < count; ++i) input[i] = static_cast<std::uint8_t>((91 * i + 7) & 255);
    std::uint8_t* device_input = nullptr;
    std::uint8_t* device_output = nullptr;
    check(cudaMalloc(&device_input, count), "input allocation");
    check(cudaMalloc(&device_output, count), "output allocation");
    check(cudaMemcpy(device_input, input.data(), count, cudaMemcpyHostToDevice), "input upload");

    const augmatch::SigmoidContrastConfig sigmoid{width, height, channels, 0.41f, 6.75f};
    augmatch::sigmoid_contrast_u8(device_input, device_output, sigmoid);
    check(cudaMemcpy(actual.data(), device_output, count, cudaMemcpyDeviceToHost), "sigmoid download");
    for (std::size_t i = 0; i < count; ++i) {
      const float x = static_cast<float>(input[i]) / 255.0f;
      expected[i] = quantize((1.0f / (1.0f + expf(sigmoid.gain * (sigmoid.cutoff - x)))) * 255.0f);
    }
    if (actual != expected) return 1;

    const augmatch::LogContrastConfig logarithm{width, height, channels, 1.35f, 2.75f};
    augmatch::log_contrast_u8(device_input, device_output, logarithm);
    check(cudaMemcpy(actual.data(), device_output, count, cudaMemcpyDeviceToHost), "log download");
    const float scale = powf(logarithm.base, logarithm.gain) - 1.0f;
    const float denominator = logf(logarithm.base);
    for (std::size_t i = 0; i < count; ++i) {
      const float x = static_cast<float>(input[i]) / 255.0f;
      expected[i] = quantize((log1pf(x * scale) / denominator) * 255.0f);
    }
    if (actual != expected) return 2;
    check(cudaFree(device_input), "input release");
    check(cudaFree(device_output), "output release");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 77;
  }
}
