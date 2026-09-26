#include "augmatch/sensor/bayer.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int patterns[4][2][2] = {
    {{0,1},{1,2}}, {{2,1},{1,0}}, {{1,0},{2,1}}, {{1,2},{0,1}}};
int channel(int pattern, int x, int y) { return patterns[pattern][y & 1][x & 1]; }
int clamp(int value, int limit) { return std::max(0, std::min(limit - 1, value)); }
float raw_at(const std::vector<unsigned char>& raw, int w, int h, int x, int y) {
  return raw[clamp(y, h) * w + clamp(x, w)];
}
unsigned char rounded(float value) {
  return static_cast<unsigned char>(std::floor(std::max(0.0f, std::min(255.0f, value)) + .5f));
}
float mhc(const std::vector<unsigned char>& raw, int w, int h, int x, int y,
          int target, int pattern) {
  const int center = channel(pattern, x, y);
  if (center == target) return raw_at(raw, w, h, x, y);
  int kind = 0;
  bool horizontal = true;
  if (target != 1) {
    if (center == 1) {
      horizontal = channel(pattern, clamp(x - 1, w), y) == target ||
                   channel(pattern, clamp(x + 1, w), y) == target;
      kind = 1;
    } else kind = 2;
  }
  static const float green[5][5] = {
      {0,0,-1,0,0}, {0,0,2,0,0}, {-1,2,4,2,-1},
      {0,0,2,0,0}, {0,0,-1,0,0}};
  static const float same[5][5] = {
      {0,0,.5f,0,0}, {0,-1,0,-1,0}, {-1,4,5,4,-1},
      {0,-1,0,-1,0}, {0,0,.5f,0,0}};
  static const float diagonal[5][5] = {
      {0,0,-1.5f,0,0}, {0,2,0,2,0}, {-1.5f,0,6,0,-1.5f},
      {0,2,0,2,0}, {0,0,-1.5f,0,0}};
  float sum = 0;
  for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
    const float coefficient = kind == 0 ? green[dy + 2][dx + 2] :
        kind == 2 ? diagonal[dy + 2][dx + 2] :
        horizontal ? same[dy + 2][dx + 2] : same[dx + 2][dy + 2];
    sum += coefficient * raw_at(raw, w, h, x + dx, y + dy);
  }
  return sum / 8.0f;
}
float edge_direction(const std::vector<unsigned char>& raw, int w, int h, int x,
                     int y, int target, int pattern, int dx, int dy) {
  float sum = 0; int count = 0;
  for (int sign : {-1, 1}) for (int distance = 1; distance <= 2; ++distance) {
    const int xx = clamp(x + sign * distance * dx, w);
    const int yy = clamp(y + sign * distance * dy, h);
    if (channel(pattern, xx, yy) == target) {
      sum += raw_at(raw, w, h, xx, yy); ++count; break;
    }
  }
  return count ? sum / count : raw_at(raw, w, h, x, y);
}
float edge(const std::vector<unsigned char>& raw, int w, int h, int x, int y,
           int target, int pattern) {
  const int center = channel(pattern, x, y);
  if (center == target) return raw_at(raw, w, h, x, y);
  if (center == 1) {
    const float gh = std::fabs(raw_at(raw,w,h,x-1,y)-raw_at(raw,w,h,x+1,y));
    const float gv = std::fabs(raw_at(raw,w,h,x,y-1)-raw_at(raw,w,h,x,y+1));
    const float horizontal = edge_direction(raw,w,h,x,y,target,pattern,1,0);
    const float vertical = edge_direction(raw,w,h,x,y,target,pattern,0,1);
    return gh < gv ? horizontal : gh > gv ? vertical : (horizontal + vertical) * .5f;
  }
  const float gd1 = std::fabs(raw_at(raw,w,h,x-1,y-1)-raw_at(raw,w,h,x+1,y+1));
  const float gd2 = std::fabs(raw_at(raw,w,h,x-1,y+1)-raw_at(raw,w,h,x+1,y-1));
  const float d1 = edge_direction(raw,w,h,x,y,target,pattern,1,1);
  const float d2 = edge_direction(raw,w,h,x,y,target,pattern,1,-1);
  return gd1 < gd2 ? d1 : gd1 > gd2 ? d2 : (d1 + d2) * .5f;
}
std::vector<unsigned char> reference(const std::vector<unsigned char>& raw, int w,
                                     int h, int pattern, bool edge_aware) {
  std::vector<unsigned char> result(w * h * 3);
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
    for (int c = 0; c < 3; ++c)
      result[(y * w + x) * 3 + c] = rounded(edge_aware ?
          edge(raw,w,h,x,y,c,pattern) : mhc(raw,w,h,x,y,c,pattern));
  return result;
}
}

int main() {
  int devices = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&devices);
  if (device_status == cudaErrorNoDevice || (device_status == cudaSuccess && devices == 0)) return 77;
  if (device_status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(device_status));
    return 1;
  }
  const int w = 8, h = 8;
  std::vector<unsigned char> host(w * h), gpu(w * h * 3);
  for (int i = 0; i < w * h; ++i) host[i] = static_cast<unsigned char>((i * 29 + 7) & 255);
  unsigned char *di = nullptr, *do_ = nullptr;
  cudaError_t status = cudaMalloc(&di, w * h);
  if (status == cudaSuccess) status = cudaMalloc(&do_, w * h * 3);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA allocation failed: %s\n", cudaGetErrorString(status));
    if (di) cudaFree(di);
    if (do_) cudaFree(do_);
    return 1;
  }
  if (cudaMemcpy(di, host.data(), w * h, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
  for (int p = 0; p < 4; ++p) {
    augmatch::BayerConfig config{w, h, static_cast<augmatch::BayerPattern>(p)};
    for (int op = 0; op < 2; ++op) {
      const auto expected = reference(host, w, h, p, op != 0);
      if (op == 0) augmatch::bayer_demosaic_malvar_he_cutler_u8(di, do_, config);
      else augmatch::bayer_demosaic_edge_aware_u8(di, do_, config);
      if (cudaDeviceSynchronize() != cudaSuccess ||
          cudaMemcpy(gpu.data(), do_, w * h * 3, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "CUDA execution failed pattern %d operation %d: %s\n",
                     p, op, cudaGetErrorString(cudaGetLastError()));
        cudaFree(di); cudaFree(do_); return 1;
      }
      if (expected != gpu) {
        std::fprintf(stderr, "%s CUDA mismatch pattern %d\n",
                     op ? "edge-aware" : "Malvar", p);
        cudaFree(di); cudaFree(do_); return 1;
      }
    }
  }
  cudaFree(di); cudaFree(do_);
  return 0;
}
