#include "augmatch/noise.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

int main() {
  int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
  constexpr int w=3,h=4,c=1; const std::vector<float> input{0,1,0,0,0,0,0,0,0,0,0,0};
  augmatch::BrightPixel bright{1,0,1.0f,-1}; std::vector<std::uint8_t> mask{0,1,0,0,1,0,0,0,0,1,1,1};
  float *di=nullptr,*doo=nullptr; augmatch::BrightPixel *dp=nullptr; std::uint8_t *dm=nullptr;
  if(cudaMalloc(&di,input.size()*sizeof(float))!=cudaSuccess||cudaMalloc(&doo,input.size()*sizeof(float))!=cudaSuccess||cudaMalloc(&dp,sizeof(bright))!=cudaSuccess||cudaMalloc(&dm,mask.size())!=cudaSuccess)return 1;
  cudaMemcpy(di,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(dp,&bright,sizeof(bright),cudaMemcpyHostToDevice);cudaMemcpy(dm,mask.data(),mask.size(),cudaMemcpyHostToDevice);
  augmatch::BloomingVerticalSmearConfig bloom{w,h,c,dp,1,nullptr,0.5f,0.5f,0.5f,0.0f,1.0f,9}; augmatch::blooming_vertical_smear_f32(di,doo,bloom);
  if(cudaDeviceSynchronize()!=cudaSuccess)return 2; std::vector<float> result(input.size());cudaMemcpy(result.data(),doo,result.size()*sizeof(float),cudaMemcpyDeviceToHost);
  const float expected[]{0,1,0,0,0.125f,0,0,0.0625f,0,0,0.03125f,0};
  for(std::size_t i=0;i<result.size();++i)if(std::abs(result[i]-expected[i])>1e-6f)return 3;
  augmatch::SensorDustOpaqueMaskConfig dust{w,h,c,dm,0,0.75f};augmatch::sensor_dust_opaque_mask_f32(di,doo,dust);if(cudaDeviceSynchronize()!=cudaSuccess)return 4;cudaMemcpy(result.data(),doo,result.size()*sizeof(float),cudaMemcpyDeviceToHost);if(std::abs(result[1]-0.75f)>1e-6f||result[0]!=0.0f)return 5;
  cudaFree(dm);cudaFree(dp);cudaFree(doo);cudaFree(di);return 0;
}
