#include "augmatch/core/pixel.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__device__ unsigned char sat(float x){return (unsigned char)fminf(255.f,fmaxf(0.f,rintf(x)));}
__global__ void bc_kernel(const unsigned char* in,unsigned char* out,BrightnessContrastConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=sat(c.contrast*in[i]+c.brightness*255.f);}
__global__ void gamma_kernel(const unsigned char* in,unsigned char* out,GammaConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=sat(powf(in[i]/255.f,c.gamma)*255.f);}
__global__ void gray_kernel(const unsigned char* in,unsigned char* out,GrayscaleConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const unsigned char* p=in+i*c.channels;unsigned char y=sat(.299f*p[0]+.587f*p[min(1,c.channels-1)]+.114f*p[min(2,c.channels-1)]);for(int ch=0;ch<c.channels;++ch)out[i*c.channels+ch]=y;}
void check(cudaError_t e,const char* p){if(e!=cudaSuccess)throw std::runtime_error(std::string(p)+": "+cudaGetErrorString(e));}
void checkbase(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid pixel dimensions");}
}
void brightness_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const BrightnessContrastConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out||!isfinite(c.brightness)||!isfinite(c.contrast)||c.contrast<0)throw std::invalid_argument("invalid brightness/contrast configuration");int n=c.width*c.height*c.channels;bc_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"bc_kernel");}
void gamma_u8(const std::uint8_t* in,std::uint8_t* out,const GammaConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out||!isfinite(c.gamma)||c.gamma<=0)throw std::invalid_argument("invalid gamma configuration");int n=c.width*c.height*c.channels;gamma_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"gamma_kernel");}
void grayscale_u8(const std::uint8_t* in,std::uint8_t* out,const GrayscaleConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null grayscale buffer");int n=c.width*c.height;gray_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"gray_kernel");}
}
