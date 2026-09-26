#include "augmatch/compression/transport.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
int main() {
  int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0) return 77;
  std::vector<std::uint8_t> in{0,1,2,3,4,5}, out(6); std::uint8_t *di=nullptr,*do_=nullptr;
  if(cudaMalloc(&di,6)||cudaMalloc(&do_,6)) return 77; if(cudaMemcpy(di,in.data(),6,cudaMemcpyHostToDevice)) return 77;
  augmatch::truncated_frame_u8(di,do_,{6,3,255}); if(cudaDeviceSynchronize()) return 77; if(cudaMemcpy(out.data(),do_,6,cudaMemcpyDeviceToHost)) return 77;
  assert(out[0]==0&&out[2]==2&&out[3]==255&&out[5]==255); cudaFree(di); cudaFree(do_); std::cout<<"transport CUDA validation passed\n"; return 0;
}
