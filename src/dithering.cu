#include "augmatch/dithering.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {
__device__ constexpr float kBayer4x4[16] = {
    0.0f, 8.0f, 2.0f, 10.0f,
    12.0f, 4.0f, 14.0f, 6.0f,
    3.0f, 11.0f, 1.0f, 9.0f,
    15.0f, 7.0f, 13.0f, 5.0f};

__device__ int quantized_byte(int level, int levels) {
  const int value = static_cast<int>(floorf(static_cast<float>(level * 255) /
                                             static_cast<float>(levels) + 0.5f));
  return max(0, min(255, value));
}

__global__ void ordered_kernel(const unsigned char* in, unsigned char* out, DitheringConfig c, int levels) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(c.width) * c.height * c.channels;
  if (index >= count) return;
  const std::size_t pixel = index / c.channels;
  const int x = static_cast<int>(pixel % c.width);
  const int y = static_cast<int>(pixel / c.width);
  const float threshold = (kBayer4x4[(y & 3) * 4 + (x & 3)] + 0.5f) / 16.0f;
  const float value = static_cast<float>(in[index]) / 255.0f;
  const int level = max(0, min(levels, static_cast<int>(floorf(value * levels + threshold))));
  out[index] = static_cast<unsigned char>(quantized_byte(level, levels));
}

__global__ void error_diffusion_kernel(const unsigned char* in, unsigned char* out,
                                       float* errors, DitheringConfig c, int levels) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= c.channels) return;
  const std::size_t pixels = static_cast<std::size_t>(c.width) * c.height;
  float* channel_errors = errors + static_cast<std::size_t>(channel) * pixels;
  for (int y = 0; y < c.height; ++y) {
    for (int x = 0; x < c.width; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * c.width + x;
      const std::size_t index = pixel * c.channels + channel;
      const float working = fmaxf(0.0f, fminf(1.0f,
          static_cast<float>(in[index]) / 255.0f + channel_errors[pixel]));
      const int level = max(0, min(levels, static_cast<int>(floorf(working * levels + 0.5f))));
      const float quantized = static_cast<float>(level) / static_cast<float>(levels);
      out[index] = static_cast<unsigned char>(quantized_byte(level, levels));
      const float error = working - quantized;
      if (x + 1 < c.width) channel_errors[pixel + 1] += error * (7.0f / 16.0f);
      if (y + 1 < c.height) {
        channel_errors[pixel + c.width] += error * (5.0f / 16.0f);
        if (x > 0) channel_errors[pixel + c.width - 1] += error * (3.0f / 16.0f);
        if (x + 1 < c.width) channel_errors[pixel + c.width + 1] += error * (1.0f / 16.0f);
      }
    }
  }
}

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
void validate(const DitheringConfig& c, const std::uint8_t* in, std::uint8_t* out) {
  if (!in || !out || c.width <= 0 || c.height <= 0 || c.channels <= 0)
    throw std::invalid_argument("invalid dithering dimensions or buffer");
  if (c.bit_depth < 1 || c.bit_depth > 8)
    throw std::invalid_argument("dithering bit depth must be in [1,8]");
  if (c.mode != DitheringMode::OrderedBayer4x4 &&
      c.mode != DitheringMode::ErrorDiffusionFloydSteinberg)
    throw std::invalid_argument("unsupported dithering mode");
}
}  // namespace

void dithering_u8(const std::uint8_t* in, std::uint8_t* out, const DitheringConfig& c, cudaStream_t stream) {
  validate(c, in, out);
  const int levels = (1 << c.bit_depth) - 1;
  const std::size_t count = static_cast<std::size_t>(c.width) * c.height * c.channels;
  if (c.mode == DitheringMode::OrderedBayer4x4) {
    const std::size_t blocks = (count + 255) / 256;
    ordered_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(in, out, c, levels);
    check(cudaGetLastError(), "ordered_dithering_kernel");
    return;
  }
  float* errors = nullptr;
  const std::size_t error_count = static_cast<std::size_t>(c.width) * c.height * c.channels;
  check(cudaMalloc(reinterpret_cast<void**>(&errors), error_count * sizeof(float)), "dithering error allocation");
  try {
    check(cudaMemsetAsync(errors, 0, error_count * sizeof(float), stream), "dithering error initialization");
    const unsigned blocks = static_cast<unsigned>((c.channels + 255) / 256);
    error_diffusion_kernel<<<blocks, 256, 0, stream>>>(in, out, errors, c, levels);
    check(cudaGetLastError(), "error_diffusion_dithering_kernel");
    check(cudaFree(errors), "dithering error release");
    errors = nullptr;
  } catch (...) {
    if (errors) cudaFree(errors);
    throw;
  }
}

void dither_u8(const std::uint8_t* in, std::uint8_t* out, const DitheringConfig& c, cudaStream_t stream) {
  dithering_u8(in, out, c, stream);
}
}  // namespace augmatch
