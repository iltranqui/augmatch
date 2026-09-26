#include "augmatch/tone.hpp"
#include "clahe_impl.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct Case {
  int width;
  int height;
  int channels;
  float clip_limit;
  int tiles_x;
  int tiles_y;
};
}  // namespace

int main() {
  try {
    const Case cases[] = {
        {9, 7, 3, 3.5f, 3, 2},  // non-divisible dimensions and interleaved channels
        {16, 12, 1, 40.0f, 4, 3},  // divisible dimensions
        {17, 13, 4, 1.0f, 2, 5},  // extra channels and a small clip limit
        {1, 1, 1, 2.0f, 8, 8},  // reflected padding beyond a one-pixel image
    };
    for (const Case& test : cases) {
      const std::size_t elements = static_cast<std::size_t>(test.width) * test.height * test.channels;
      std::vector<std::uint8_t> input(elements), expected(elements), actual(elements);
      for (int y = 0; y < test.height; ++y) {
        for (int x = 0; x < test.width; ++x) {
          for (int channel = 0; channel < test.channels; ++channel) {
            input[(static_cast<std::size_t>(y) * test.width + x) * test.channels + channel] =
                static_cast<std::uint8_t>((29 * x + 17 * y + 41 * channel + x * y) & 255);
          }
        }
      }
      augmatch::detail::clahe_u8_host(input.data(), expected.data(), test.width, test.height,
                                      test.channels, test.clip_limit, test.tiles_x, test.tiles_y);
      std::uint8_t* device_input = nullptr;
      std::uint8_t* device_output = nullptr;
      check(cudaMalloc(reinterpret_cast<void**>(&device_input), elements), "input allocation");
      check(cudaMalloc(reinterpret_cast<void**>(&device_output), elements), "output allocation");
      check(cudaMemcpy(device_input, input.data(), elements, cudaMemcpyHostToDevice), "input upload");
      augmatch::clahe_u8(device_input, device_output,
                         {test.width, test.height, test.channels, test.clip_limit, test.tiles_x, test.tiles_y});
      check(cudaMemcpy(actual.data(), device_output, elements, cudaMemcpyDeviceToHost), "output download");
      check(cudaFree(device_input), "input release");
      check(cudaFree(device_output), "output release");
      int maximum_difference = 0;
      for (std::size_t index = 0; index < elements; ++index) {
        const int difference = std::abs(static_cast<int>(actual[index]) - static_cast<int>(expected[index]));
        maximum_difference = std::max(maximum_difference, difference);
      }
      if (maximum_difference > 1) {
        std::cerr << "CLAHE CUDA mismatch for " << test.width << 'x' << test.height
                  << "x" << test.channels << " tiles " << test.tiles_x << 'x' << test.tiles_y
                  << " (maximum difference " << maximum_difference << ")\n";
        return 1;
      }
    }
    std::cout << "CLAHE CUDA validation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CLAHE CUDA validation failed: " << error.what() << '\n';
    return 2;
  }
}
