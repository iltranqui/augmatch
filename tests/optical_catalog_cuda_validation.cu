#include "augmatch/filter/filter.hpp"
#include "augmatch/color/color.hpp"
#include "augmatch/geometry/geometric.hpp"
#include <cuda_runtime.h>
#include <vector>
int main(){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;const int w=7,h=5,ch=3;const std::size_t n=static_cast<std::size_t>(w)*h*ch;std::vector<unsigned char> host(n,90),result(n);std::vector<float> rows(h,0);unsigned char *di=nullptr,*do_=nullptr;float *dx=nullptr,*dy=nullptr;cudaMalloc(&di,n);cudaMalloc(&do_,n);cudaMalloc(&dx,h*sizeof(float));cudaMalloc(&dy,h*sizeof(float));cudaMemcpy(di,host.data(),n,cudaMemcpyHostToDevice);cudaMemcpy(dx,rows.data(),h*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(dy,rows.data(),h*sizeof(float),cudaMemcpyHostToDevice);
  auto check=[&](auto fn,auto cfg){fn(di,do_,cfg,nullptr);if(cudaDeviceSynchronize()!=cudaSuccess)return false;cudaMemcpy(result.data(),do_,n,cudaMemcpyDeviceToHost);return result==host;};
  if(!check(augmatch::optical_defocus_u8,augmatch::OpticalDefocusConfig{w,h,ch,0,0}))return 1;
  if(!check(augmatch::optical_motion_blur_u8,augmatch::OpticalMotionBlurConfig{w,h,ch,1,0,0}))return 2;
  if(!check(augmatch::optical_zoom_blur_u8,augmatch::OpticalZoomBlurConfig{w,h,ch,1,1,0}))return 3;
  if(!check(augmatch::optical_chromatic_aberration_u8,augmatch::OpticalChromaticAberrationConfig{w,h,ch,0,0,0,1,1,1}))return 4;
  if(!check(augmatch::lateral_chromatic_aberration_u8,augmatch::LateralChromaticAberrationConfig{w,h,ch,0,0,0}))return 5;
  if(!check(augmatch::longitudinal_chromatic_aberration_u8,augmatch::LongitudinalChromaticAberrationConfig{w,h,ch,0,0,0,1}))return 6;
  if(!check(augmatch::thin_prism_distortion_u8,augmatch::ThinPrismDistortionConfig{w,h,ch,0,0,0,0}))return 7;
  if(!check(augmatch::rolling_shutter_geometric_distortion_u8,augmatch::RollingShutterGeometricDistortionConfig{w,h,ch,dx,dy}))return 8;
  cudaFree(di);cudaFree(do_);cudaFree(dx);cudaFree(dy);return 0;}
