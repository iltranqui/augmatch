#include "augmatch/geometry/geometric.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
int main(){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;const int w=11,h=8,c=1;const std::size_t n=static_cast<std::size_t>(w)*h;std::vector<std::uint8_t> host(n,91),result(n);std::uint8_t *in=nullptr,*out=nullptr;if(cudaMalloc(&in,n)!=cudaSuccess||cudaMalloc(&out,n)!=cudaSuccess)return 77;if(cudaMemcpy(in,host.data(),n,cudaMemcpyHostToDevice)!=cudaSuccess)return 77;augmatch::focus_breathing_u8(in,out,{w,h,c,1.0f,.2f,.1f,.5f,.5f,augmatch::Interpolation::Linear,0});if(cudaDeviceSynchronize()!=cudaSuccess)return 1;if(cudaMemcpy(result.data(),out,n,cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;for(auto v:result)if(v!=91)return 2;cudaFree(in);cudaFree(out);return 0;}
