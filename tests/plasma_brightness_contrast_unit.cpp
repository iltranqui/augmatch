#include "augmatch/color/color.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime.h>
#endif

int main(){
  const int w=4,h=1,c=2;
  const std::uint8_t input[]={0,255,64,128,128,64,255,0};
  const float field[]={0.0f,0.5f,1.0f,0.25f};
  std::uint8_t output[8]{};
#if AUGMATCH_HAS_CUDA
  auto run=[&](const augmatch::PlasmaBrightnessContrastConfig& host_config,const std::uint8_t* host_input,std::uint8_t* host_output){
    const std::size_t bytes=sizeof(input); std::uint8_t *device_input=nullptr,*device_output=nullptr; float *device_field=nullptr;
    if(cudaMalloc(&device_input,bytes)!=cudaSuccess||cudaMalloc(&device_output,bytes)!=cudaSuccess)return false;
    cudaMemcpy(device_input,host_input,bytes,cudaMemcpyHostToDevice);
    augmatch::PlasmaBrightnessContrastConfig device_config=host_config;
    if(host_config.field){if(cudaMalloc(&device_field,sizeof(field))!=cudaSuccess)return false;cudaMemcpy(device_field,host_config.field,sizeof(field),cudaMemcpyHostToDevice);device_config.field=device_field;}
    try{augmatch::plasma_brightness_contrast_u8(device_input,device_output,device_config);cudaDeviceSynchronize();cudaMemcpy(host_output,device_output,bytes,cudaMemcpyDeviceToHost);}catch(...){cudaFree(device_field);cudaFree(device_input);cudaFree(device_output);throw;}
    cudaFree(device_field);cudaFree(device_input);cudaFree(device_output); return true;
  };
#else
  auto run=[&](const augmatch::PlasmaBrightnessContrastConfig& config,const std::uint8_t* host_input,std::uint8_t* host_output){augmatch::plasma_brightness_contrast_u8(host_input,host_output,config);return true;};
#endif
#if AUGMATCH_HAS_CUDA
  auto run_contrast=[&](const augmatch::PlasmaContrastConfig& host_config,const std::uint8_t* host_input,std::uint8_t* host_output){
    const std::size_t bytes=sizeof(input); std::uint8_t *device_input=nullptr,*device_output=nullptr; float *device_field=nullptr;
    if(cudaMalloc(&device_input,bytes)!=cudaSuccess||cudaMalloc(&device_output,bytes)!=cudaSuccess)return false;
    cudaMemcpy(device_input,host_input,bytes,cudaMemcpyHostToDevice); augmatch::PlasmaContrastConfig device_config=host_config;
    if(host_config.field){if(cudaMalloc(&device_field,sizeof(field))!=cudaSuccess)return false;cudaMemcpy(device_field,host_config.field,sizeof(field),cudaMemcpyHostToDevice);device_config.field=device_field;}
    try{augmatch::plasma_contrast_u8(device_input,device_output,device_config);cudaDeviceSynchronize();cudaMemcpy(host_output,device_output,bytes,cudaMemcpyDeviceToHost);}catch(...){cudaFree(device_field);cudaFree(device_input);cudaFree(device_output);throw;}
    cudaFree(device_field);cudaFree(device_input);cudaFree(device_output); return true;
  };
#else
  auto run_contrast=[&](const augmatch::PlasmaContrastConfig& config,const std::uint8_t* host_input,std::uint8_t* host_output){augmatch::plasma_contrast_u8(host_input,host_output,config);return true;};
#endif
  if(!run({w,h,c,0.1f,1.0f,1.0f,field,0},input,output))return 5;
  const std::uint8_t expected[]={153,153,90,154,154,26,217,89};
  for(int i=0;i<8;++i)if(output[i]!=expected[i])return 1;
  std::vector<float> generated(static_cast<std::size_t>(w)*h); augmatch::make_plasma_field(generated.data(),w,h,77);
  std::vector<std::uint8_t> seeded(8),repeat(8),explicit_output(8),contrast(8),same_as_contrast(8);
  const augmatch::PlasmaBrightnessContrastConfig seeded_config{w,h,c,-0.07f,1.4f,0.8f,nullptr,77};
  if(!run(seeded_config,input,seeded.data())||!run(seeded_config,input,repeat.data()))return 6;
  if(!run({w,h,c,-0.07f,1.4f,0.8f,generated.data(),0},input,explicit_output.data()))return 7;
  if(seeded!=repeat||seeded!=explicit_output)return 2;
  if(!run({w,h,c,0.0f,1.4f,0.8f,generated.data(),0},input,same_as_contrast.data()))return 8;
  if(!run_contrast({w,h,c,1.4f,0.8f,generated.data(),0},input,contrast.data()))return 9;
  if(same_as_contrast!=contrast)return 3;
#if !AUGMATCH_HAS_CUDA
  bool rejected=false; const float invalid[]={1.01f};
  try{augmatch::plasma_brightness_contrast_u8(input,output,{1,1,1,0.0f,1.0f,1.0f,invalid,0});}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)return 4;
#endif
  return 0;
}
