#include "augmatch/sensor/iso_profile.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
static void ck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
int main(){int devices=0;ck(cudaGetDeviceCount(&devices));if(!devices)return 77;
  const augmatch::CameraProfileLookupPoint hp[]={{100,1,1,1,1,0,1,100,0,0},{400,2,1,1,1,0,1,100,0,0}};augmatch::CameraProfileLookupPoint* dp=nullptr;ck(cudaMalloc(&dp,sizeof(hp)));ck(cudaMemcpy(dp,hp,sizeof(hp),cudaMemcpyHostToDevice));
  float hi[4]={.1f,.2f,.3f,.4f},ho[4]={};float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,sizeof(hi)));ck(cudaMalloc(&doo,sizeof(ho)));ck(cudaMemcpy(di,hi,sizeof(hi),cudaMemcpyHostToDevice));augmatch::CameraProfileLookupConfig camera{};camera.width=4;camera.height=1;camera.channels=1;camera.iso=250;camera.profile={dp,2};camera.seed=7;augmatch::camera_profile_lookup_f32(di,doo,camera);ck(cudaDeviceSynchronize());ck(cudaMemcpy(ho,doo,sizeof(ho),cudaMemcpyDeviceToHost));for(float v:ho)if(v<0||v>1)return 1;
  augmatch::CalibrationFrameNoiseConfig cal{};cal.width=4;cal.height=1;cal.channels=1;cal.seed=3;float rm[4]={0,.01f,.02f,.03f},fm[4]={0,.01f,-.01f,.02f},gm[4]={1,1,1,1};float *dr=nullptr,*df=nullptr,*dg=nullptr;ck(cudaMalloc(&dr,sizeof(rm)));ck(cudaMalloc(&df,sizeof(fm)));ck(cudaMalloc(&dg,sizeof(gm)));ck(cudaMemcpy(dr,rm,sizeof(rm),cudaMemcpyHostToDevice));ck(cudaMemcpy(df,fm,sizeof(fm),cudaMemcpyHostToDevice));ck(cudaMemcpy(dg,gm,sizeof(gm),cudaMemcpyHostToDevice));cal.calibration={0,0,0,dr,df,dg,4};augmatch::calibration_frame_noise_f32(di,doo,cal);ck(cudaDeviceSynchronize());ck(cudaMemcpy(ho,doo,sizeof(ho),cudaMemcpyDeviceToHost));for(float v:ho)if(v<0||v>1)return 1;ck(cudaFree(dr));ck(cudaFree(df));ck(cudaFree(dg));ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));return 0;}
