#include "augmatch/tone.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int main(){
 try {
  int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
  const int w=16,h=1,c=1,n=w*h*c; std::vector<unsigned char> host(n),back(n); for(int i=0;i<n;++i)host[i]=static_cast<unsigned char>(i*17);
  unsigned char *di=nullptr,*do_=nullptr; if(cudaMalloc(&di,n)||cudaMalloc(&do_,n))return 77; cudaMemcpy(di,host.data(),n,cudaMemcpyHostToDevice);
  augmatch::GammaVariationConfig cfg{w,h,c,1.0f}; augmatch::gamma_variation_u8(di,do_,cfg); if(cudaDeviceSynchronize()!=cudaSuccess)return 1; cudaMemcpy(back.data(),do_,n,cudaMemcpyDeviceToHost); if(back!=host)return 1;
  augmatch::PosterizationConfig q{w,h,c,4}; augmatch::posterization_u8(di,do_,q); if(cudaDeviceSynchronize()!=cudaSuccess)return 1; cudaMemcpy(back.data(),do_,n,cudaMemcpyDeviceToHost); for(auto x:back)if(x%16)return 1;
  cudaFree(di);cudaFree(do_);std::puts("tone variation CUDA validation passed");return 0;
 } catch (...) { return 77; }
}
