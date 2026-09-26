#include "augmatch/bayer.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  constexpr int w = 8, h = 8;
  const std::size_t pixels = w * h;
  std::vector<std::uint8_t> rgb(pixels * 3), rgbw(pixels * 4), mask{0, 1, 2, 3};
  for (std::size_t i = 0; i < pixels; ++i) {
    rgb[i*3+0] = 10; rgb[i*3+1] = 20; rgb[i*3+2] = 30;
    rgbw[i*4+0] = 10; rgbw[i*4+1] = 20; rgbw[i*4+2] = 30; rgbw[i*4+3] = 40;
  }
  std::uint8_t *di=nullptr,*do_=nullptr,*dw=nullptr,*dm=nullptr;
  if (cudaMalloc(&di, rgb.size()) != cudaSuccess || cudaMalloc(&do_, pixels) != cudaSuccess ||
      cudaMalloc(&dw, rgbw.size()) != cudaSuccess || cudaMalloc(&dm, mask.size()) != cudaSuccess) return 77;
  cudaMemcpy(di, rgb.data(), rgb.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, rgbw.data(), rgbw.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dm, mask.data(), mask.size(), cudaMemcpyHostToDevice);
  try {
    augmatch::quad_bayer_sample_u8(di, do_, {w, h, augmatch::QuadBayerPattern::RGGB});
    augmatch::rgbw_sample_u8(dw, do_, {w, h, augmatch::RGBWPattern::RGWB});
    augmatch::custom_cfa_sample_u8(di, do_, {w, h, 3, 2, 2, dm});
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
  } catch (const std::exception& error) {
    const std::string message = error.what();
    if (message.find("unsupported toolchain") != std::string::npos ||
        message.find("no kernel image") != std::string::npos) return 77;
    return 1;
  }
  cudaFree(di); cudaFree(do_); cudaFree(dw); cudaFree(dm);
  std::puts("PASS: CUDA CFA kernels");
  return 0;
}
