#include "augmatch/filter/canny.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>
int main(){
  int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
  const int w=32,h=24,ch=3; const std::size_t n=static_cast<std::size_t>(w)*h*ch;
  std::vector<std::uint8_t> input(n,0),output(n),repeat(n);
  for(int y=6;y<18;++y)for(int x=8;x<24;++x)for(int c=0;c<ch;++c)input[(static_cast<std::size_t>(y)*w+x)*ch+c]=static_cast<std::uint8_t>(c==0?220:100);
  std::uint8_t *di=nullptr,*do1=nullptr,*do2=nullptr;
  if(cudaMalloc(&di,n)!=cudaSuccess||cudaMalloc(&do1,n)!=cudaSuccess||cudaMalloc(&do2,n)!=cudaSuccess)return 1;
  if(cudaMemcpy(di,input.data(),n,cudaMemcpyHostToDevice)!=cudaSuccess)return 2;
  const augmatch::CannyConfig cfg{w,h,ch,20.0f,80.0f,3,augmatch::BorderPolicy::Clamp,0};
  try { augmatch::canny_u8(di,do1,cfg); augmatch::canny_u8(di,do2,cfg); }
  catch(const std::runtime_error& error){ if(std::string(error.what()).find("unsupported toolchain")!=std::string::npos)return 77; return 9; }
  if(cudaDeviceSynchronize()!=cudaSuccess)return 3;
  if(cudaMemcpy(output.data(),do1,n,cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(repeat.data(),do2,n,cudaMemcpyDeviceToHost)!=cudaSuccess)return 4;
  cudaFree(di);cudaFree(do1);cudaFree(do2);
  bool edge=false;for(std::size_t i=0;i<n;i+=ch){if(output[i]!=0&&output[i]!=255)return 5;if(output[i])edge=true;for(int c=1;c<ch;++c)if(output[i+c]!=output[i])return 6;if(output[i]!=repeat[i])return 7;}
  return edge?0:8;
}
