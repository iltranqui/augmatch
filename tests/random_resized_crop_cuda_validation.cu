#include "augmatch/geometry/size.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <exception>
#include <cstring>
#include <string>
#include <vector>
int main(){
  int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0) return 77;
  const int w=7,h=6,ch=2,ow=4,oh=3; const std::size_t ni=std::size_t(w)*h*ch,no=std::size_t(ow)*oh*ch;
  std::vector<unsigned char> in(ni),out(no),expected(no); for(std::size_t i=0;i<ni;++i) in[i]=static_cast<unsigned char>((i*19+5)&255);
  augmatch::RandomResizedCropConfig cfg{w,h,ow,oh,ch,2,1,4,3,augmatch::Interpolation::Nearest,17};
  for(int y=0;y<oh;++y) for(int x=0;x<ow;++x) for(int c=0;c<ch;++c) expected[(y*ow+x)*ch+c]=in[((y+1)*w+x+2)*ch+c];
  unsigned char *di=nullptr,*doo=nullptr; if(cudaMalloc(&di,ni)||cudaMalloc(&doo,no)) return 1;
  if(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice)!=cudaSuccess) return 1;
  try { augmatch::random_resized_crop_u8(di,doo,cfg); const cudaError_t sync=cudaDeviceSynchronize(); if(sync!=cudaSuccess) { const char* msg=cudaGetErrorString(sync); return std::strstr(msg,"unsupported")?77:3; } const cudaError_t copy=cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost); if(copy!=cudaSuccess) { const char* msg=cudaGetErrorString(copy); return std::strstr(msg,"unsupported")?77:1; } }
  catch(const std::exception& error) { cudaFree(di); cudaFree(doo); const std::string message=error.what(); return (message.find("unsupported")!=std::string::npos||message.find("no kernel image")!=std::string::npos)?77:1; }
  cudaFree(di); cudaFree(doo);
  return out==expected?0:1;
}
