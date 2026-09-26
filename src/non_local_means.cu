#include "augmatch/filter/filter.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {
__device__ int reflect101_nlm(int value, int size) {
  if (size == 1) return 0;
  while (value < 0 || value >= size) value = value < 0 ? -value : 2 * size - value - 2;
  return value;
}

__global__ void non_local_means_kernel(const unsigned char* in, unsigned char* out,
                                       NonLocalMeansDenoisingConfig c) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = c.width * c.height * c.channels;
  if (index >= count) return;
  const int channel = index % c.channels;
  const int x = (index / c.channels) % c.width;
  const int y = index / (c.width * c.channels);
  const int patch_side = 2 * c.patch_radius + 1;
  const float inv_h2 = 1.0f / (c.h * c.h);
  const float normalizer = static_cast<float>(patch_side * patch_side * c.channels);
  float weighted_sum = 0.0f;
  float weight_sum = 0.0f;
  for (int sy = -c.search_radius; sy <= c.search_radius; ++sy) {
    for (int sx = -c.search_radius; sx <= c.search_radius; ++sx) {
      float squared_distance = 0.0f;
      for (int py = -c.patch_radius; py <= c.patch_radius; ++py) {
        for (int px = -c.patch_radius; px <= c.patch_radius; ++px) {
          const int ax = reflect101_nlm(x + px, c.width);
          const int ay = reflect101_nlm(y + py, c.height);
          const int bx = reflect101_nlm(x + sx + px, c.width);
          const int by = reflect101_nlm(y + sy + py, c.height);
          const int a = (ay * c.width + ax) * c.channels;
          const int b = (by * c.width + bx) * c.channels;
          for (int patch_channel = 0; patch_channel < c.channels; ++patch_channel) {
            const float difference = static_cast<float>(in[a + patch_channel]) -
                                     static_cast<float>(in[b + patch_channel]);
            squared_distance += difference * difference;
          }
        }
      }
      const float weight = expf(-(squared_distance / normalizer) * inv_h2);
      const int sample_x = reflect101_nlm(x + sx, c.width);
      const int sample_y = reflect101_nlm(y + sy, c.height);
      weighted_sum += weight * in[(sample_y * c.width + sample_x) * c.channels + channel];
      weight_sum += weight;
    }
  }
  const float value = fminf(255.0f, fmaxf(0.0f, weighted_sum / weight_sum));
  out[index] = static_cast<unsigned char>(floorf(value + 0.5f));
}

void validate_nlm_cuda(const std::uint8_t* in, const std::uint8_t* out,
                       const NonLocalMeansDenoisingConfig& c) {
  if (!in || !out || c.width <= 0 || c.height <= 0 || c.channels <= 0 ||
      c.patch_radius < 0 || c.patch_radius > 4 || c.search_radius < 0 ||
      c.search_radius > 8 || !std::isfinite(c.h) || c.h <= 0.0f) {
    throw std::invalid_argument("invalid non-local means denoising configuration");
  }
}

void check_nlm(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
} // namespace

void non_local_means_denoising_u8(const std::uint8_t* in, std::uint8_t* out,
                                  const NonLocalMeansDenoisingConfig& c,
                                  cudaStream_t stream) {
  validate_nlm_cuda(in, out, c);
  const int count = c.width * c.height * c.channels;
  non_local_means_kernel<<<(count + 255) / 256, 256, 0, stream>>>(in, out, c);
  check_nlm(cudaGetLastError(), "non_local_means_kernel");
}
} // namespace augmatch
