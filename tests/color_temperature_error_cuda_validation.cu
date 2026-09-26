#include "augmatch/color/color.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
namespace {
struct Rgb { float r,g,b; };
Rgb blackbody(float kelvin) {
  const float t=kelvin/100.0f; Rgb rgb{};
  if(t<=66.0f) { rgb.r=255.0f; rgb.g=99.4708025861f*std::log(t)-161.1195681661f; rgb.b=t<=19.0f?0.0f:138.5177312231f*std::log(t-10.0f)-305.0447927307f; }
  else { rgb.r=329.698727446f*std::pow(t-60.0f,-0.1332047592f); rgb.g=288.1221695283f*std::pow(t-60.0f,-0.0755148492f); rgb.b=255.0f; }
  rgb.r=std::fmin(255.0f,std::fmax(0.0f,rgb.r)); rgb.g=std::fmin(255.0f,std::fmax(0.0f,rgb.g)); rgb.b=std::fmin(255.0f,std::fmax(0.0f,rgb.b)); return rgb;
}
int rounded(float value) { return std::max(0,std::min(255,static_cast<int>(std::floor(value+0.5f)))); }
}
int main() {
  int count=0;
  if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) return 77;
  constexpr int w=17,h=9,ch=4;
  std::vector<std::uint8_t> input(w*h*ch), output(input.size()), expected(input.size());
  for (std::size_t i=0;i<input.size();++i) input[i]=static_cast<std::uint8_t>((i*31+71)%256);
  try {
    augmatch::ColorTemperatureErrorConfig config{w,h,ch,3200.0f,0.63f};
    const Rgb reference=blackbody(6500.0f), color=blackbody(config.temperature_kelvin);
    const float gain[3]={1+config.strength*(color.r/reference.r-1),1+config.strength*(color.g/reference.g-1),1+config.strength*(color.b/reference.b-1)};
    for(std::size_t p=0;p<input.size()/ch;++p) {
      for(int k=0;k<3;++k) expected[p*ch+k]=static_cast<std::uint8_t>(rounded(input[p*ch+k]*gain[k]));
      expected[p*ch+3]=input[p*ch+3];
    }
    std::uint8_t *device_in=nullptr,*device_out=nullptr;
    if (cudaMalloc(&device_in,input.size())!=cudaSuccess || cudaMalloc(&device_out,input.size())!=cudaSuccess) return 77;
    cudaMemcpy(device_in,input.data(),input.size(),cudaMemcpyHostToDevice);
    augmatch::color_temperature_error_u8(device_in,device_out,config);
    cudaError_t status=cudaDeviceSynchronize();
    if (status==cudaSuccess) status=cudaMemcpy(output.data(),device_out,output.size(),cudaMemcpyDeviceToHost);
    cudaFree(device_in); cudaFree(device_out);
    if (status!=cudaSuccess) return 1;
    for (std::size_t i=0;i<output.size();++i) {
      // CPU libm and CUDA libm can differ at a half-integer rounding boundary.
      if (i%ch==3) { if (output[i]!=input[i]) return 2; }
      else if (std::abs(static_cast<int>(output[i])-static_cast<int>(expected[i]))>1) return 3;
    }
  } catch (...) { return 4; }
  return 0;
}
