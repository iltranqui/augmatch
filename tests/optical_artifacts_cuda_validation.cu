#include "augmatch/sensor/noise.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
int main(){
  const int w=3,h=2,c=2;std::vector<float> host(w*h*c,0.5f),result(host.size());float *di=nullptr,*do_=nullptr,*dm=nullptr;cudaMalloc(&di,host.size()*sizeof(float));cudaMalloc(&do_,host.size()*sizeof(float));std::vector<float> map(w*h,1.0f);map[1]=0.5f;cudaMalloc(&dm,map.size()*sizeof(float));cudaMemcpy(di,host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice);
  augmatch::LensVignettingConfig v;v.width=w;v.height=h;v.channels=c;v.map=dm;augmatch::lens_vignetting_f32(di,do_,v);if(cudaDeviceSynchronize()!=cudaSuccess)return 77;cudaMemcpy(result.data(),do_,result.size()*sizeof(float),cudaMemcpyDeviceToHost);if(std::abs(result[3]-.25f)>1e-5f)return 1;
  augmatch::SensorLensDustShadowsConfig d{w,h,c,dm,1,0,1};augmatch::sensor_lens_dust_shadows_f32(di,do_,d);if(cudaDeviceSynchronize()!=cudaSuccess)return 77;cudaMemcpy(result.data(),do_,result.size()*sizeof(float),cudaMemcpyDeviceToHost);if(std::abs(result[3]-.25f)>1e-5f)return 2;cudaFree(dm);cudaFree(do_);cudaFree(di);return 0;
}
