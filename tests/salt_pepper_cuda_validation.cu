#include "augmatch/color/arithmetic.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  int device = 0;
  if (cudaGetDeviceCount(&device) != cudaSuccess || device == 0) return 77;
  constexpr int w=4,h=3,c=2;
  const std::size_t pixels=static_cast<std::size_t>(w)*h, n=pixels*c;
  std::vector<std::uint8_t> input(n, 37), mask(pixels, 0), output(n);
  mask[1]=1; mask[5]=1;
  std::uint8_t *di=nullptr,*do_=nullptr,*dm=nullptr;
  if(cudaMalloc(&di,n)||cudaMalloc(&do_,n)||cudaMalloc(&dm,pixels)) return 1;
  if(cudaMemcpy(di,input.data(),n,cudaMemcpyHostToDevice)||cudaMemcpy(dm,mask.data(),pixels,cudaMemcpyHostToDevice)) return 2;
  augmatch::SaltConfig cfg{}; cfg.width=w;cfg.height=h;cfg.channels=c;cfg.mask=dm;
  augmatch::salt_u8(di,do_,cfg); if(cudaDeviceSynchronize()!=cudaSuccess) return 3;
  if(cudaMemcpy(output.data(),do_,n,cudaMemcpyDeviceToHost)!=cudaSuccess) return 4;
  for(std::size_t p=0;p<pixels;++p) for(int ch=0;ch<c;++ch) if(output[p*c+ch]!=(mask[p]?255:37)) return 5;
  cudaFree(dm);cudaFree(do_);cudaFree(di);
  std::cout << "PASS: CUDA Salt mask" << std::endl;
  return 0;
}
