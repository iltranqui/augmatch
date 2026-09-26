#include "augmatch/hsv.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
int main(){
  int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
  const int w=3,h=1,c=4; const std::vector<std::uint8_t> input={255,0,0,17,0,255,0,23,0,0,255,29}; std::vector<std::uint8_t> expected(input.size()),got(input.size());
  expected={138,0,0,17,0,138,0,23,0,0,138,29};
  std::uint8_t *di=nullptr,*do_=nullptr; if(cudaMalloc(&di,input.size())!=cudaSuccess||cudaMalloc(&do_,input.size())!=cudaSuccess)return 1;
  if(cudaMemcpy(di,input.data(),input.size(),cudaMemcpyHostToDevice)!=cudaSuccess)return 1;
  augmatch::multiply_and_add_to_brightness_u8(di,do_,{w,h,c,0.5f,10.0f}); if(cudaDeviceSynchronize()!=cudaSuccess)return 1;
  if(cudaMemcpy(got.data(),do_,got.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
  if(got!=expected)return 2;
  expected={255,128,128,17,128,255,128,23,128,128,255,29};
  augmatch::add_to_saturation_u8(di,do_,{w,h,c,-128.0f}); if(cudaDeviceSynchronize()!=cudaSuccess)return 1;
  if(cudaMemcpy(got.data(),do_,got.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
  cudaFree(di);cudaFree(do_); return got==expected?0:3;
}
