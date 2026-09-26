#include "augmatch/voronoi.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
std::uint64_t mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

augmatch::VoronoiPoint reference_site(int w, int h, int count,
                                       std::uint64_t seed, int i) {
  const int gx = static_cast<int>(ceil(sqrt(static_cast<double>(count))));
  const int gy = (count + gx - 1) / gx;
  const int col = i % gx, row = i / gx;
  const std::uint64_t r = mix(seed + static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ULL);
  const std::uint64_t q = mix(r + 0x632be59bd9b4e019ULL);
  const int x = static_cast<int>((static_cast<std::uint64_t>(col) * w + r % static_cast<std::uint64_t>(w / gx + 1)) / gx);
  const int y = static_cast<int>((static_cast<std::uint64_t>(row) * h + q % static_cast<std::uint64_t>(h / gy + 1)) / gy);
  return {std::min(w - 1, std::max(0, x)), std::min(h - 1, std::max(0, y))};
}

void host_reference(const std::vector<unsigned char>& in,
                    std::vector<unsigned char>& mask,
                    std::vector<std::uint32_t>& labels,
                    int w, int h, int count, std::uint64_t seed) {
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
    std::uint64_t best = UINT64_MAX;
    int winner = 0;
    for (int i = 0; i < count; ++i) {
      const auto p = reference_site(w, h, count, seed, i);
      const std::int64_t dx = x - p.x, dy = y - p.y;
      const auto distance = static_cast<std::uint64_t>(dx * dx + dy * dy);
      if (distance < best) { best = distance; winner = i; }
    }
    const auto site = reference_site(w, h, count, seed, winner);
    const std::size_t at = static_cast<std::size_t>(y) * w + x;
    labels[at] = static_cast<std::uint32_t>(winner);
    mask[at] = in[static_cast<std::size_t>(site.y) * w + site.x];
  }
}

bool check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  constexpr int w = 17, h = 13, count = 9;
  constexpr std::uint64_t seed = 42;
  const std::size_t n = static_cast<std::size_t>(w) * h;
  std::vector<unsigned char> in(n), actual_mask(n), expected_mask(n);
  std::vector<std::uint32_t> actual_labels(n), expected_labels(n);
  for (std::size_t i = 0; i < n; ++i)
    in[i] = static_cast<unsigned char>((i * 11 + 3) % 251);
  host_reference(in, expected_mask, expected_labels, w, h, count, seed);

  unsigned char *device_in = nullptr, *device_mask = nullptr;
  std::uint32_t* device_labels = nullptr;
  if (!check(cudaMalloc(&device_in, n * sizeof(*device_in)), "cudaMalloc input") ||
      !check(cudaMalloc(&device_mask, n * sizeof(*device_mask)), "cudaMalloc mask") ||
      !check(cudaMalloc(&device_labels, n * sizeof(*device_labels)), "cudaMalloc labels")) {
    if (device_in) cudaFree(device_in);
    if (device_mask) cudaFree(device_mask);
    if (device_labels) cudaFree(device_labels);
    return 1;
  }
  bool ok = check(cudaMemcpy(device_in, in.data(), n * sizeof(*device_in), cudaMemcpyHostToDevice), "copy input to device");
  augmatch::UniformVoronoiConfig config{w, h, count, seed};
  if (ok) {
    augmatch::uniform_voronoi_u8(device_in, device_mask, config);
    augmatch::uniform_voronoi_labels_u32(device_labels, config);
    ok = check(cudaDeviceSynchronize(), "synchronize Voronoi kernels");
  }
  if (ok) ok = check(cudaMemcpy(actual_mask.data(), device_mask, n * sizeof(*device_mask), cudaMemcpyDeviceToHost), "copy mask from device");
  if (ok) ok = check(cudaMemcpy(actual_labels.data(), device_labels, n * sizeof(*device_labels), cudaMemcpyDeviceToHost), "copy labels from device");
  cudaFree(device_in);
  cudaFree(device_mask);
  cudaFree(device_labels);
  if (!ok) return 1;
  if (actual_mask != expected_mask) {
    std::fprintf(stderr, "CUDA Voronoi mask differs from independent host reference\n");
    return 1;
  }
  if (actual_labels != expected_labels) {
    std::fprintf(stderr, "CUDA Voronoi labels differ from independent host reference\n");
    return 1;
  }
  return 0;
}
