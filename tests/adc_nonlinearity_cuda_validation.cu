#include "augmatch/noise.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  constexpr int w=4,h=1,c=1,levels=4;
  const std::vector<float> input{0.0f,0.34f,0.67f,1.0f};
  const std::vector<float> dnl_lut(levels,1.0f), inl_lut{0.0f,0.25f,-0.25f,0.0f}, capacity{0.2f,0.4f,0.8f,1.2f};
  float *di=nullptr,*do_=nullptr,*dl=nullptr,*il=nullptr,*cm=nullptr;
  if (cudaMalloc(&di,input.size()*sizeof(float))!=cudaSuccess || cudaMalloc(&do_,input.size()*sizeof(float))!=cudaSuccess || cudaMalloc(&dl,dnl_lut.size()*sizeof(float))!=cudaSuccess || cudaMalloc(&il,inl_lut.size()*sizeof(float))!=cudaSuccess || cudaMalloc(&cm,capacity.size()*sizeof(float))!=cudaSuccess) return 77;
  cudaMemcpy(di,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dl,dnl_lut.data(),dnl_lut.size()*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(il,inl_lut.data(),inl_lut.size()*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(cm,capacity.data(),capacity.size()*sizeof(float),cudaMemcpyHostToDevice);
  augmatch::adc_differential_non_linearity_f32(di,do_,{w,h,c,levels,dl,0.0f,7});
  cudaDeviceSynchronize(); std::vector<float> result(input.size()); cudaMemcpy(result.data(),do_,result.size()*sizeof(float),cudaMemcpyDeviceToHost);
  for (std::size_t i=0;i<result.size();++i) if (std::abs(result[i]-static_cast<float>(i)/3.0f)>1e-6f) return 1;
  augmatch::adc_integral_non_linearity_f32(di,do_,{w,h,c,levels,il,0.0f,7});
  cudaDeviceSynchronize(); cudaMemcpy(result.data(),do_,result.size()*sizeof(float),cudaMemcpyDeviceToHost);
  if (std::abs(result[1]-0.41666666f)>1e-5f || std::abs(result[2]-0.58333333f)>1e-5f) return 2;
  augmatch::sensor_well_capacity_variation_f32(di,do_,{w,h,c,cm,0.0f,7});
  cudaDeviceSynchronize(); cudaMemcpy(result.data(),do_,result.size()*sizeof(float),cudaMemcpyDeviceToHost);
  const float expected[]{0.0f,0.34f,0.67f,1.0f};
  for (std::size_t i=0;i<result.size();++i) if (std::abs(result[i]-expected[i])>1e-6f) return 3;
  cudaFree(cm); cudaFree(il); cudaFree(dl); cudaFree(do_); cudaFree(di);
  return 0;
}
