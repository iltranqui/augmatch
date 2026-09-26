#include "augmatch/catalog/imgcorruptlike.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  int devices=0;
  if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
  const int w=8,h=4,c=3; const std::size_t n=static_cast<std::size_t>(w)*h*c;
  std::vector<std::uint8_t> host(n,100), out(n);
  std::uint8_t *di=nullptr,*do_=nullptr;
  if(cudaMalloc(&di,n)!=cudaSuccess || cudaMalloc(&do_,n)!=cudaSuccess) return 77;
  cudaMemcpy(di,host.data(),n,cudaMemcpyHostToDevice);
  augmatch::SpeckleNoiseConfig speckle{w,h,c,0.0f,0.1f,9};
  augmatch::speckle_noise_u8(di,do_,speckle);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  assert(cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost)==cudaSuccess);
  augmatch::PixelateConfig pixel{w,h,c,2};
  augmatch::pixelate_u8(di,do_,pixel);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  cudaFree(di); cudaFree(do_);
  return 0;
}
