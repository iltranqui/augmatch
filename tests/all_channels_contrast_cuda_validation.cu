#include "augmatch/color/tone.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
}

int main() {
  try {
    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "device query");
    if (device_count == 0) return 77;
    const int width = 13, height = 9, channels = 4;
    const std::size_t elements = static_cast<std::size_t>(width) * height * channels;
    std::vector<std::uint8_t> input(elements), clahe_alias(elements), clahe_native(elements);
    std::vector<std::uint8_t> equalize_alias(elements), equalize_native(elements);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        for (int channel = 0; channel < channels; ++channel)
          input[(static_cast<std::size_t>(y) * width + x) * channels + channel] =
              static_cast<std::uint8_t>((19 * x + 31 * y + 53 * channel + x * y) & 255);
    std::uint8_t *device_input = nullptr, *device_output = nullptr, *device_reference = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_input), elements), "input allocation");
    check(cudaMalloc(reinterpret_cast<void**>(&device_output), elements), "output allocation");
    check(cudaMalloc(reinterpret_cast<void**>(&device_reference), elements), "reference allocation");
    check(cudaMemcpy(device_input, input.data(), elements, cudaMemcpyHostToDevice), "input upload");
    const augmatch::AllChannelsCLAHEConfig clahe_config{width, height, channels, 3.5f, 4, 3};
    augmatch::all_channels_clahe_u8(device_input, device_output, clahe_config);
    augmatch::clahe_u8(device_input, device_reference,
                       {width, height, channels, clahe_config.clip_limit, clahe_config.tiles_x, clahe_config.tiles_y});
    check(cudaMemcpy(clahe_alias.data(), device_output, elements, cudaMemcpyDeviceToHost), "CLAHE alias download");
    check(cudaMemcpy(clahe_native.data(), device_reference, elements, cudaMemcpyDeviceToHost), "CLAHE reference download");
    if (clahe_alias != clahe_native) throw std::runtime_error("CUDA AllChannelsCLAHE differs from native CLAHE");

    const augmatch::AllChannelsHistogramEqualizationConfig equalize_config{width, height, channels};
    augmatch::all_channels_histogram_equalization_u8(device_input, device_output, equalize_config);
    augmatch::equalize_u8(device_input, device_reference, {width, height, channels});
    check(cudaMemcpy(equalize_alias.data(), device_output, elements, cudaMemcpyDeviceToHost), "equalize alias download");
    check(cudaMemcpy(equalize_native.data(), device_reference, elements, cudaMemcpyDeviceToHost), "equalize reference download");
    if (equalize_alias != equalize_native)
      throw std::runtime_error("CUDA AllChannelsHistogramEqualization differs from native equalize");
    check(cudaFree(device_input), "input release");
    check(cudaFree(device_output), "output release");
    check(cudaFree(device_reference), "reference release");
    std::cout << "all-channel contrast CUDA validation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "all-channel contrast CUDA validation failed: " << error.what() << '\n';
    return 1;
  }
}
