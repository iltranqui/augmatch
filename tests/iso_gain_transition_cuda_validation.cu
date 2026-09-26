#include "augmatch/sensor/iso_profile.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
static void ck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
int main(){int devices=0;ck(cudaGetDeviceCount(&devices));if(!devices)return 77;
  const augmatch::IsoNoiseProfilePoint hp[]={{100,1,1,1,1,0,1},{400,4,1,1,1,0,1}};augmatch::IsoNoiseProfilePoint* dp=nullptr;ck(cudaMalloc(&dp,sizeof(hp)));ck(cudaMemcpy(dp,hp,sizeof(hp),cudaMemcpyHostToDevice));
  float hi[4]={.1f,.25f,.5f,.9f},ho[4]={};float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,sizeof(hi)));ck(cudaMalloc(&doo,sizeof(ho)));ck(cudaMemcpy(di,hi,sizeof(hi),cudaMemcpyHostToDevice));
  augmatch::DualConversionGainConfig c{};c.width=4;c.height=1;c.channels=1;c.iso=250;c.profile={dp,2};c.low_gain_threshold=1;c.high_gain_threshold=4;c.low_conversion_gain=1;c.high_conversion_gain=2;c.seed=8;augmatch::dual_conversion_gain_f32(di,doo,c);ck(cudaDeviceSynchronize());ck(cudaMemcpy(ho,doo,sizeof(ho),cudaMemcpyDeviceToHost));for(int i=0;i<4;++i)if(!(ho[i]>=hi[i]&&ho[i]<=1))return 1;
  augmatch::GainSwitchTransitionConfig t{};t.width=4;t.height=1;t.channels=1;t.iso=250;t.profile={dp,2};t.low_gain_threshold=1;t.high_gain_threshold=4;t.low_signal_gain=1;t.high_signal_gain=2;t.transition_width=.5f;t.transition_strength=.1f;augmatch::gain_switch_transition_f32(di,doo,t);ck(cudaDeviceSynchronize());ck(cudaMemcpy(ho,doo,sizeof(ho),cudaMemcpyDeviceToHost));for(float v:ho)if(v<0||v>1)return 1;
  ck(cudaFree(di));ck(cudaFree(doo));ck(cudaFree(dp));return 0;}
