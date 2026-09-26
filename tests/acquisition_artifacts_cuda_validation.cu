#include "augmatch/weather/environmental.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
int main(){
  int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
  const int w=3,h=3,c=1,t=3;const std::size_t image_n=static_cast<std::size_t>(w)*h*c,batch_n=static_cast<std::size_t>(t)*image_n;std::vector<float> image(image_n,.2f),batch(batch_n,.5f),out(batch_n),single(image_n),mask(image_n,1.0f);
  float *di=nullptr,*do_=nullptr,*dm=nullptr; if(cudaMalloc(&di,batch_n*sizeof(float))!=cudaSuccess||cudaMalloc(&do_,batch_n*sizeof(float))!=cudaSuccess)return 77;cudaMemcpy(di,batch.data(),batch_n*sizeof(float),cudaMemcpyHostToDevice);
  augmatch::FluorescentLightFlickerConfig f{};f.frames=t;f.height=h;f.width=w;f.channels=c;f.amplitude=.2f;f.flicker_frequency_hz=.25f;f.frame_rate_hz=1;augmatch::fluorescent_light_flicker_f32(di,do_,f);if(cudaDeviceSynchronize()!=cudaSuccess)return 77;cudaMemcpy(out.data(),do_,batch_n*sizeof(float),cudaMemcpyDeviceToHost);if(out==batch)return 1;
  augmatch::SensorTemperatureDriftConfig temp{};temp.frames=t;temp.height=h;temp.width=w;temp.channels=c;temp.start_temperature_celsius=25;temp.end_temperature_celsius=35;temp.dark_current_electrons_per_second=100;temp.electrons_per_unit=1000;temp.temperature_coefficient_per_celsius=std::log(2.0f)/10.0f;augmatch::sensor_temperature_drift_f32(di,do_,temp);if(cudaDeviceSynchronize()!=cudaSuccess)return 77;
  float *si=nullptr,*so=nullptr;cudaMalloc(&si,image_n*sizeof(float));cudaMalloc(&so,image_n*sizeof(float));cudaMemcpy(si,image.data(),image_n*sizeof(float),cudaMemcpyHostToDevice);cudaMalloc(&dm,image_n*sizeof(float));cudaMemcpy(dm,mask.data(),image_n*sizeof(float),cudaMemcpyHostToDevice);augmatch::DirtyLensBlurConfig d{};d.width=w;d.height=h;d.channels=c;d.map=dm;d.radius=1;d.strength=1;augmatch::dirty_lens_blur_f32(si,so,d);if(cudaDeviceSynchronize()!=cudaSuccess)return 77;cudaMemcpy(single.data(),so,image_n*sizeof(float),cudaMemcpyDeviceToHost);for(float v:single)if(v<0||v>1)return 2;
  cudaFree(dm);cudaFree(si);cudaFree(so);cudaFree(di);cudaFree(do_);return 0;
}
