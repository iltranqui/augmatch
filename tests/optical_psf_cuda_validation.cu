#include "augmatch/filter.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
int main(){
  int device=0; if(cudaGetDeviceCount(&device)!=cudaSuccess||device==0)return 77;
  const int w=9,h=7,c=1; const std::size_t n=static_cast<std::size_t>(w)*h*c; std::vector<std::uint8_t> host(n,83),result(n);
  std::uint8_t *in=nullptr,*out=nullptr; if(cudaMalloc(&in,n)!=cudaSuccess||cudaMalloc(&out,n)!=cudaSuccess)return 77; if(cudaMemcpy(in,host.data(),n,cudaMemcpyHostToDevice)!=cudaSuccess)return 77;
  augmatch::diffraction_blur_u8(in,out,{w,h,c,3,550,2,50}); augmatch::bokeh_blur_u8(in,out,{w,h,c,3,.2f}); augmatch::cat_eye_bokeh_u8(in,out,{w,h,c,3,.7f,.5f,.5f}); augmatch::aperture_shape_blur_u8(in,out,{w,h,c,3,6,11,.2f});
  if(cudaDeviceSynchronize()!=cudaSuccess)return 1; if(cudaMemcpy(result.data(),out,n,cudaMemcpyDeviceToHost)!=cudaSuccess)return 1; for(auto v:result)if(v!=83)return 2; cudaFree(in);cudaFree(out);return 0;
}
