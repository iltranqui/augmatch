#include "augmatch/color/colorspace.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
int main(){
  int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
  const int w=2,h=1,c=4;const std::vector<std::uint8_t> in={10,20,30,77,200,100,50,88};std::vector<std::uint8_t> out(in.size());std::uint8_t *di=nullptr,*do_=nullptr;
  if(cudaMalloc(&di,in.size())!=cudaSuccess||cudaMalloc(&do_,in.size())!=cudaSuccess)return 1;if(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice)!=cudaSuccess)return 1;
  augmatch::with_colorspace_u8(di,do_,{w,h,c,augmatch::ColorSpace::RGB,2,2,2,1,2,3});if(cudaDeviceSynchronize()!=cudaSuccess)return 1;if(cudaMemcpy(out.data(),do_,out.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
  if(out!=std::vector<std::uint8_t>({21,42,63,77,255,202,103,88}))return 2;
  augmatch::with_brightness_channels_u8(di,do_,{w,h,c,augmatch::ColorSpace::RGB,1,-1,-1,2,5});if(cudaDeviceSynchronize()!=cudaSuccess)return 1;if(cudaMemcpy(out.data(),do_,out.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
  cudaFree(di);cudaFree(do_);return out==std::vector<std::uint8_t>({10,45,30,77,200,205,50,88})?0:3;
}
