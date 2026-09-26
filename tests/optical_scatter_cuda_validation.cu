#include "augmatch/noise.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
int main() {
 try {
  int w=5,h=3,c=1; std::vector<float> host(w*h,0.1f), result(host.size());
  float *di=nullptr,*do_=nullptr; if(cudaMalloc(&di,host.size()*sizeof(float))!=cudaSuccess)return 77;
  if(cudaMalloc(&do_,host.size()*sizeof(float))!=cudaSuccess)return 77;
  cudaMemcpy(di,host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice);
  augmatch::FlareSource source{2,1,.1f,0,-1}; augmatch::FlareHalo halo{2,1,2,.5f,-1};
  augmatch::FlareSource *ds=nullptr; augmatch::FlareHalo *dh=nullptr;
  cudaMalloc(&ds,sizeof(source)); cudaMalloc(&dh,sizeof(halo)); cudaMemcpy(ds,&source,sizeof(source),cudaMemcpyHostToDevice); cudaMemcpy(dh,&halo,sizeof(halo),cudaMemcpyHostToDevice);
  augmatch::FlareConfig fc{w,h,c,ds,1,dh,1,nullptr,1,1,0,1}; augmatch::flare_f32(di,do_,fc);
  if(cudaDeviceSynchronize()!=cudaSuccess)return 77; cudaMemcpy(result.data(),do_,result.size()*sizeof(float),cudaMemcpyDeviceToHost);
  if(std::abs(result[7]-.55f)>1e-5f)return 1;
  augmatch::VeilingGlareConfig vg{w,h,c,nullptr,1,.5f,0,1}; augmatch::veiling_glare_f32(di,do_,vg);
  if(cudaDeviceSynchronize()!=cudaSuccess)return 77; cudaMemcpy(result.data(),do_,result.size()*sizeof(float),cudaMemcpyDeviceToHost);
  if(std::abs(result[0]-.15f)>1e-5f)return 2;
  augmatch::BloomConfig bc{w,h,c,1,.05f,1,nullptr,1,0,1}; augmatch::bloom_f32(di,do_,bc);
  if(cudaDeviceSynchronize()!=cudaSuccess)return 77;
  cudaFree(dh);cudaFree(ds);cudaFree(do_);cudaFree(di);return 0;
 } catch (...) { return 77; }
}
