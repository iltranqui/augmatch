#include "augmatch/sensor/bayer.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t e, const char* where) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
}

int main() {
  try {
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) return 77;
    constexpr int w = 11, h = 9;
    const std::size_t raw_bytes = static_cast<std::size_t>(w) * h;
    const std::size_t rgb_bytes = raw_bytes * 3;
    std::vector<std::uint8_t> raw(raw_bytes), constant_raw(raw_bytes, 128), gpu(rgb_bytes), gpu_again(rgb_bytes);
    for (std::size_t i = 0; i < raw.size(); ++i)
      raw[i] = static_cast<std::uint8_t>((i * 29 + (i / w) * 17) & 255);
    const augmatch::DemosaicingNoiseAmplificationConfig zero_noise{w, h, augmatch::BayerPattern::RGGB, 0.0f, 1.5f, 123};
    const augmatch::DemosaicingNoiseAmplificationConfig config{w, h, augmatch::BayerPattern::RGGB, 7.0f, 1.5f, 123};

    std::uint8_t* device_input = nullptr;
    std::uint8_t* device_output = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_input), raw_bytes), "cudaMalloc input");
    check(cudaMalloc(reinterpret_cast<void**>(&device_output), rgb_bytes), "cudaMalloc output");

    // A constant Bayer plane is an independent oracle: every demosaiced channel must remain 128.
    check(cudaMemcpy(device_input, constant_raw.data(), raw_bytes, cudaMemcpyHostToDevice), "copy constant input");
    augmatch::bayer_demosaicing_noise_amplification_u8(device_input, device_output, zero_noise);
    check(cudaGetLastError(), "constant launch");
    check(cudaDeviceSynchronize(), "constant synchronize");
    check(cudaMemcpy(gpu.data(), device_output, rgb_bytes, cudaMemcpyDeviceToHost), "copy constant output");
    for (std::size_t i = 0; i < rgb_bytes; ++i) {
      if (gpu[i] != 128) {
        std::fprintf(stderr, "constant-input oracle mismatch at %zu: gpu=%u\n", i, gpu[i]);
        return 1;
      }
    }

    // The stochastic path must be repeatable for the explicit seed and remain uint8-bounded.
    check(cudaMemcpy(device_input, raw.data(), raw_bytes, cudaMemcpyHostToDevice), "copy input");
    augmatch::bayer_demosaicing_noise_amplification_u8(device_input, device_output, config);
    check(cudaGetLastError(), "noise amplification launch");
    check(cudaDeviceSynchronize(), "noise amplification synchronize");
    check(cudaMemcpy(gpu.data(), device_output, rgb_bytes, cudaMemcpyDeviceToHost), "copy output");
    augmatch::bayer_demosaicing_noise_amplification_u8(device_input, device_output, config);
    check(cudaGetLastError(), "repeat noise amplification launch");
    check(cudaDeviceSynchronize(), "repeat noise amplification synchronize");
    check(cudaMemcpy(gpu_again.data(), device_output, rgb_bytes, cudaMemcpyDeviceToHost), "copy repeat output");
    check(cudaFree(device_input), "free input");
    check(cudaFree(device_output), "free output");
    for (std::size_t i = 0; i < rgb_bytes; ++i) {
      if (gpu[i] != gpu_again[i]) {
        std::fprintf(stderr, "demosaic artifact determinism failure at %zu: first=%u second=%u\n", i, gpu[i], gpu_again[i]);
        return 1;
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "demosaic artifact CUDA error: %s\n", e.what());
    return 1;
  }
}
