#include "augmatch/color.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int main(){
  int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
  const int w=2,h=1,c=4; const std::uint8_t host[]={10,20,30,7,200,100,50,8}; std::uint8_t *di=nullptr,*do_=nullptr;
  if(cudaMalloc(&di,8)||cudaMalloc(&do_,8)||cudaMemcpy(di,host,8,cudaMemcpyHostToDevice))return 1;
  augmatch::ColorMatrixPerturbationConfig cfg{w,h,c}; augmatch::color_matrix_perturbation_u8(di,do_,cfg);
  augmatch::per_channel_gain_noise_u8(di,do_,{w,h,c,{1,1,1},0.0f,7});
  augmatch::color_temperature_error_u8(di,do_,{w,h,c,6500.0f,0.0f});
  if(cudaDeviceSynchronize()!=cudaSuccess)return 1;
  std::uint8_t out[8]{};if(cudaMemcpy(out,do_,8,cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;cudaFree(di);cudaFree(do_);
  for(int i=0;i<8;++i)if(out[i]!=host[i])return 1; return 0;
}
