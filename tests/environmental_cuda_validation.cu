#include "augmatch/environmental.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
int main(){
  int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
  const int w=3,h=2,c=1; std::vector<float> host(w*h*c,.2f), result(host.size());
  float *in=nullptr,*out=nullptr; if(cudaMalloc(&in,host.size()*sizeof(float))!=cudaSuccess||cudaMalloc(&out,host.size()*sizeof(float))!=cudaSuccess)return 77;
  cudaMemcpy(in,host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice);
  augmatch::AcquisitionGainConfig cfg{w,h,c,2.0f,0,1}; augmatch::overexposure_f32(in,out,cfg); cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess)return 77;
  cudaMemcpy(result.data(),out,result.size()*sizeof(float),cudaMemcpyDeviceToHost); cudaFree(in);cudaFree(out);
  for(float x:result)if(std::abs(x-.4f)>1e-5f)return 1; return 0;
}
