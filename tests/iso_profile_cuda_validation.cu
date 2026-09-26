#include "augmatch/sensor/iso_profile.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
int main(){int devices=0;check(cudaGetDeviceCount(&devices));if(devices==0)return 77;
  const augmatch::IsoNoiseProfilePoint host[]={{100,1,1,1,1,0,1},{800,4,2,3,5,0.05f,0.95f}};augmatch::IsoNoiseProfilePoint* points=nullptr;check(cudaMalloc(&points,sizeof(host)));check(cudaMemcpy(points,host,sizeof(host),cudaMemcpyHostToDevice));
  std::uint8_t in[4]={0,64,192,255},out[4]={};std::uint8_t *di=nullptr,*doo=nullptr;check(cudaMalloc(&di,4));check(cudaMalloc(&doo,4));check(cudaMemcpy(di,in,4,cudaMemcpyHostToDevice));augmatch::IsoNoiseApplicationConfig c{};c.width=4;c.height=1;c.channels=1;c.iso=450;c.profile={points,2};c.shot_scale=100;c.read_noise_stddev=0;c.fpn_stddev=0;c.seed=4;augmatch::iso_signal_levels_u8(di,doo,c);check(cudaDeviceSynchronize());check(cudaMemcpy(out,doo,4,cudaMemcpyDeviceToHost));if(out[0]!=6||out[3]!=249){std::fprintf(stderr,"ISO level mismatch: got %u and %u, expected 6 and 249\n",static_cast<unsigned>(out[0]),static_cast<unsigned>(out[3]));return 1;}
  float hinput[4]={0.1f,0.2f,0.3f,0.4f},houtput[4]={};float *dfi=nullptr,*dfo=nullptr;check(cudaMalloc(&dfi,sizeof(hinput)));check(cudaMalloc(&dfo,sizeof(houtput)));check(cudaMemcpy(dfi,hinput,sizeof(hinput),cudaMemcpyHostToDevice));
  augmatch::ExposureTimeDarkCurrentConfig dark{};dark.width=4;dark.height=1;dark.channels=1;dark.iso=450;dark.profile={points,2};dark.dark_current_electrons_per_second=100.0f;dark.exposure_seconds=1.0f;dark.electrons_per_unit=1000.0f;dark.seed=5;augmatch::exposure_time_dark_current_f32(dfi,dfo,dark);check(cudaDeviceSynchronize());check(cudaMemcpy(houtput,dfo,sizeof(houtput),cudaMemcpyDeviceToHost));for(float v:houtput)if(v<0.1f||v>1.0f)return 1;
  augmatch::TemperatureNoiseScalingConfig thermal{};thermal.width=4;thermal.height=1;thermal.channels=1;thermal.iso=450;thermal.profile={points,2};thermal.noise_stddev_electrons=10.0f;thermal.electrons_per_unit=1000.0f;thermal.seed=6;augmatch::temperature_noise_scaling_f32(dfi,dfo,thermal);check(cudaDeviceSynchronize());check(cudaMemcpy(houtput,dfo,sizeof(houtput),cudaMemcpyDeviceToHost));for(float v:houtput)if(v<0.0f||v>1.0f)return 1;
  check(cudaFree(dfi));check(cudaFree(dfo));check(cudaFree(di));check(cudaFree(doo));check(cudaFree(points));return 0;}
