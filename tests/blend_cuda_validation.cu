#include "augmatch/blend.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int main(){
  const int n=12; std::vector<unsigned char> a(n,0),b(n,200),o(n);
  unsigned char *da=nullptr,*db=nullptr,*do_=nullptr;
  if(cudaMalloc(&da,n)||cudaMalloc(&db,n)||cudaMalloc(&do_,n)) return 77;
  if(cudaMemcpy(da,a.data(),n,cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(db,b.data(),n,cudaMemcpyHostToDevice)!=cudaSuccess)return 77;
  augmatch::blend_alpha_u8(da,db,do_,{2,2,3,.5f});
  if(cudaDeviceSynchronize()!=cudaSuccess)return 77;
  if(cudaMemcpy(o.data(),do_,n,cudaMemcpyDeviceToHost)!=cudaSuccess)return 77;
  cudaFree(da);cudaFree(db);cudaFree(do_);
  for(auto v:o)if(v!=100)return 1;
  return 0;
}
