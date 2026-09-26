#include "augmatch/color.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

int main(){
  int devices=0;
  if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
  const int w=4,h=1,c=4; const std::vector<std::uint8_t> input={0,31,32,7,63,64,95,8,127,128,191,9,255,77,201,10};
  std::vector<std::uint8_t> output(input.size()); std::uint8_t *di=nullptr,*doo=nullptr;
  if(cudaMalloc(&di,input.size())!=cudaSuccess||cudaMalloc(&doo,input.size())!=cudaSuccess)return 77;
  if(cudaMemcpy(di,input.data(),input.size(),cudaMemcpyHostToDevice)!=cudaSuccess)return 77;
  augmatch::uniform_color_quantization_to_n_bits_u8(di,doo,{w,h,c,3});
  if(cudaDeviceSynchronize()!=cudaSuccess)return 77;
  if(cudaMemcpy(output.data(),doo,output.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)return 77;
  const std::vector<std::uint8_t> expected_bits={0,0,32,7,32,64,64,8,96,128,160,9,224,64,192,10};
  if(output!=expected_bits)return 1;
  augmatch::uniform_color_quantization_u8(di,doo,{w,h,c,4});
  if(cudaDeviceSynchronize()!=cudaSuccess)return 77;
  if(cudaMemcpy(output.data(),doo,output.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)return 77;
  const std::vector<std::uint8_t> expected_levels={32,32,32,7,32,96,96,8,96,160,160,9,224,96,224,10};
  cudaFree(di);cudaFree(doo);
  return output==expected_levels?0:2;
}
