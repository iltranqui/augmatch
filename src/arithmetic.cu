#include "augmatch/color/arithmetic.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__device__ unsigned char sat(float x){return (unsigned char)fminf(255.f,fmaxf(0.f,floorf(x+0.5f)));}
__device__ unsigned long long mix(unsigned long long x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
__device__ float uni(unsigned long long x){return (float)((mix(x)>>11)*(1.0/9007199254740992.0));}
__device__ float normal(unsigned long long x){float a=fmaxf(uni(x),1e-7f);return sqrtf(-2.f*logf(a))*cosf(6.28318530718f*uni(x^0xd1b54a32d192ed03ULL));}
__global__ void add_kernel(const unsigned char* in,unsigned char* out,ScalarArithmeticConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=sat(in[i]+c.value);}
__global__ void mul_kernel(const unsigned char* in,unsigned char* out,ScalarArithmeticConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=(unsigned char)fminf(255.f,fmaxf(0.f,in[i]*c.value));}
__device__ float elementwise_value(const float* values,int i,float min_value,float max_value,unsigned long long seed){
  if(values)return values[i];
  const float unit=uni(seed+(unsigned long long)i*0x9e3779b97f4a7c15ULL);
  return min_value+(max_value-min_value)*unit;
}
__global__ void add_elementwise_kernel(const unsigned char* in,unsigned char* out,AddElementwiseConfig c){
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;
  if(i<n)out[i]=sat(in[i]+elementwise_value(c.values,i,c.min_value,c.max_value,c.seed));
}
__global__ void multiply_elementwise_kernel(const unsigned char* in,unsigned char* out,MultiplyElementwiseConfig c){
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;
  if(i<n)out[i]=sat(in[i]*elementwise_value(c.values,i,c.min_value,c.max_value,c.seed));
}
__global__ void poster_kernel(const unsigned char* in,unsigned char* out,PosterizeConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=(unsigned char)(in[i]&(255<<(8-c.bits)));}
__global__ void solar_kernel(const unsigned char* in,unsigned char* out,SolarizeConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=in[i]>c.threshold?(unsigned char)(255-in[i]):in[i];}
__global__ void inv_kernel(const unsigned char* in,unsigned char* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=255-in[i];}
__global__ void gauss_kernel(const unsigned char* in,unsigned char* out,GaussianNoiseConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=sat(in[i]+c.stddev*normal(c.seed+(unsigned long long)i*0x9e3779b97f4a7c15ULL));}
__global__ void sp_kernel(const unsigned char* in,unsigned char* out,SaltPepperConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n){unsigned long long k=c.seed+(unsigned long long)i*0x9e3779b97f4a7c15ULL;float r=uni(k);out[i]=r<c.probability?(uni(c.seed^0xabcdefULL+(unsigned long long)i)<c.salt_probability?255:0):in[i];}}
__global__ void replace_elementwise_kernel(const unsigned char* in,unsigned char* out,ReplaceElementwiseConfig c){
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;
  if(i<n){const bool selected=c.mask?c.mask[i]!=0:uni(c.seed+(unsigned long long)i*0x9e3779b97f4a7c15ULL)<c.probability;out[i]=selected?(c.values?c.values[i]:c.replacement_value):in[i];}
}
__global__ void impulse_noise_kernel(const unsigned char* in,unsigned char* out,ImpulseNoiseConfig c){
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;
  if(i<n){const bool selected=c.mask?c.mask[i]!=0:uni(c.seed+(unsigned long long)i*0x9e3779b97f4a7c15ULL)<c.probability;const unsigned char impulse=c.values?c.values[i]:(uni(c.seed^0xabcdefULL+(unsigned long long)i)<c.salt_probability?255:0);out[i]=selected?impulse:in[i];}
}
__global__ void salt_kernel(const unsigned char* in,unsigned char* out,SaltPepperConfig c,int mode,bool coarse){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;
  const int pixel=i/c.channels,x=pixel%c.width,y=pixel/c.width;
  bool selected=false,salt=true;
  if(c.mask){selected=c.mask[pixel]!=0;if(mode==2&&c.mask[pixel]==2)salt=false;}
  for(int k=0;k<c.rectangle_count;++k){const SaltPepperConfig::Rectangle r=c.rectangles[k];if(x>=r.x0&&x<r.x1&&y>=r.y0&&y<r.y1){selected=true;salt=c.salt_probability>=0.5f;}}
  if(!c.mask&&c.rectangle_count==0){const int bw=coarse?c.block_width:1,bh=coarse?c.block_height:1,blocks_x=(c.width+bw-1)/bw;const unsigned long long block=(unsigned long long)(y/bh*blocks_x+x/bw),key=c.seed+block*0x9e3779b97f4a7c15ULL;selected=uni(key)<c.probability;salt=uni(key^0xd1b54a32d192ed03ULL)<c.salt_probability;}
  const unsigned char value=mode==0?255:mode==1?0:(salt?255:0);out[i]=selected?value:in[i];
}
void check(cudaError_t e,const char* p){if(e!=cudaSuccess)throw std::runtime_error(std::string(p)+": "+cudaGetErrorString(e));}
void checkbase(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid arithmetic dimensions");}
void checkrange(float min_value,float max_value,bool multiply){if(!isfinite(min_value)||!isfinite(max_value)||min_value>max_value||(multiply&&(min_value<0.0f||max_value<0.0f)))throw std::invalid_argument("invalid elementwise arithmetic range");}
unsigned long long host_mix(unsigned long long x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
float host_elementwise_value(unsigned long long seed,std::size_t i,float min_value,float max_value){const float unit=(float)((host_mix(seed+(unsigned long long)i*0x9e3779b97f4a7c15ULL)>>11)*(1.0/9007199254740992.0));return min_value+(max_value-min_value)*unit;}
}
void add_u8(const std::uint8_t* in,std::uint8_t* out,const ScalarArithmeticConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out||!isfinite(c.value))throw std::invalid_argument("invalid add configuration");int n=c.width*c.height*c.channels;add_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"add_kernel");}
void multiply_u8(const std::uint8_t* in,std::uint8_t* out,const ScalarArithmeticConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out||!isfinite(c.value)||c.value<0)throw std::invalid_argument("invalid multiply configuration");int n=c.width*c.height*c.channels;mul_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"mul_kernel");}
void make_add_elementwise_values(float* values,int width,int height,int channels,float min_value,float max_value,std::uint64_t seed){
  checkbase(width,height,channels);if(!values)throw std::invalid_argument("null add elementwise values");checkrange(min_value,max_value,false);const std::size_t n=(std::size_t)width*(std::size_t)height*(std::size_t)channels;for(std::size_t i=0;i<n;++i)values[i]=host_elementwise_value(seed,i,min_value,max_value);
}
void add_elementwise_u8(const std::uint8_t* in,std::uint8_t* out,const AddElementwiseConfig& c,cudaStream_t s){
  checkbase(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null elementwise arithmetic buffer");if(!c.values)checkrange(c.min_value,c.max_value,false);const int n=c.width*c.height*c.channels;add_elementwise_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"add_elementwise_kernel");
}
void make_multiply_elementwise_values(float* values,int width,int height,int channels,float min_value,float max_value,std::uint64_t seed){
  checkbase(width,height,channels);if(!values)throw std::invalid_argument("null multiply elementwise values");checkrange(min_value,max_value,true);const std::size_t n=(std::size_t)width*(std::size_t)height*(std::size_t)channels;for(std::size_t i=0;i<n;++i)values[i]=host_elementwise_value(seed,i,min_value,max_value);
}
void multiply_elementwise_u8(const std::uint8_t* in,std::uint8_t* out,const MultiplyElementwiseConfig& c,cudaStream_t s){
  checkbase(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null elementwise arithmetic buffer");if(!c.values)checkrange(c.min_value,c.max_value,true);const int n=c.width*c.height*c.channels;multiply_elementwise_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"multiply_elementwise_kernel");
}
void posterize_u8(const std::uint8_t* in,std::uint8_t* out,const PosterizeConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out||c.bits<1||c.bits>8)throw std::invalid_argument("posterize bits must be in [1,8]");int n=c.width*c.height*c.channels;poster_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"poster_kernel");}
void solarize_u8(const std::uint8_t* in,std::uint8_t* out,const SolarizeConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out||c.threshold<0||c.threshold>255)throw std::invalid_argument("solarize threshold must be in [0,255]");int n=c.width*c.height*c.channels;solar_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"solar_kernel");}
void invert_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,cudaStream_t s){checkbase(w,h,c);if(!in||!out)throw std::invalid_argument("null invert buffer");int n=w*h*c;inv_kernel<<<(n+255)/256,256,0,s>>>(in,out,n);check(cudaGetLastError(),"invert_kernel");}
void gaussian_noise_u8(const std::uint8_t* in,std::uint8_t* out,const GaussianNoiseConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out||c.stddev<0||!isfinite(c.stddev))throw std::invalid_argument("invalid Gaussian noise configuration");int n=c.width*c.height*c.channels;gauss_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"gauss_kernel");}
void salt_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const SaltPepperConfig& c,cudaStream_t s){checkbase(c.width,c.height,c.channels);if(!in||!out||c.probability<0||c.probability>1||c.salt_probability<0||c.salt_probability>1)throw std::invalid_argument("invalid salt and pepper configuration");int n=c.width*c.height*c.channels;sp_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"sp_kernel");}
void make_replace_elementwise_mask(std::uint8_t* mask,int width,int height,int channels,float probability,std::uint64_t seed){
  checkbase(width,height,channels);if(!mask||!isfinite(probability)||probability<0.0f||probability>1.0f)throw std::invalid_argument("invalid replacement mask configuration");const std::size_t n=(std::size_t)width*(std::size_t)height*(std::size_t)channels;for(std::size_t i=0;i<n;++i)mask[i]=host_elementwise_value(seed,i,0.0f,1.0f)<probability?1:0;
}
void replace_elementwise_u8(const std::uint8_t* in,std::uint8_t* out,const ReplaceElementwiseConfig& c,cudaStream_t s){
  checkbase(c.width,c.height,c.channels);if(!in||!out||(c.mask==nullptr&&(!isfinite(c.probability)||c.probability<0.0f||c.probability>1.0f))||(c.mask==nullptr&&c.values!=nullptr))throw std::invalid_argument("invalid ReplaceElementwise configuration");const int n=c.width*c.height*c.channels;replace_elementwise_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"replace_elementwise_kernel");
}
void make_impulse_noise_mask(std::uint8_t* mask,int width,int height,int channels,float probability,std::uint64_t seed){make_replace_elementwise_mask(mask,width,height,channels,probability,seed);}
void impulse_noise_u8(const std::uint8_t* in,std::uint8_t* out,const ImpulseNoiseConfig& c,cudaStream_t s){
  checkbase(c.width,c.height,c.channels);if(!in||!out||(c.mask==nullptr&&(!isfinite(c.probability)||c.probability<0.0f||c.probability>1.0f))||(c.mask==nullptr&&c.values!=nullptr)||(!c.values&&(!isfinite(c.salt_probability)||c.salt_probability<0.0f||c.salt_probability>1.0f)))throw std::invalid_argument("invalid ImpulseNoise configuration");const int n=c.width*c.height*c.channels;impulse_noise_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"impulse_noise_kernel");
}
void salt_validate(const std::uint8_t* in,const std::uint8_t* out,const SaltPepperConfig& c){
  checkbase(c.width,c.height,c.channels);if(!in||!out||c.rectangle_count<0||(c.rectangle_count>0&&!c.rectangles)||c.block_width<=0||c.block_height<=0||!isfinite(c.probability)||c.probability<0.0f||c.probability>1.0f||!isfinite(c.salt_probability)||c.salt_probability<0.0f||c.salt_probability>1.0f)throw std::invalid_argument("invalid salt and pepper configuration");
}
void salt_call(const std::uint8_t* in,std::uint8_t* out,const SaltPepperConfig& c,int mode,bool coarse,cudaStream_t s){salt_validate(in,out,c);const int n=c.width*c.height*c.channels;salt_kernel<<<(n+255)/256,256,0,s>>>(in,out,c,mode,coarse);check(cudaGetLastError(),"salt_kernel");}
void salt_u8(const std::uint8_t* in,std::uint8_t* out,const SaltConfig& c,cudaStream_t s){salt_call(in,out,c,0,false,s);}
void pepper_u8(const std::uint8_t* in,std::uint8_t* out,const PepperConfig& c,cudaStream_t s){salt_call(in,out,c,1,false,s);}
void salt_and_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const SaltAndPepperConfig& c,cudaStream_t s){salt_call(in,out,c,2,false,s);}
void coarse_salt_u8(const std::uint8_t* in,std::uint8_t* out,const CoarseSaltConfig& c,cudaStream_t s){salt_call(in,out,c,0,true,s);}
void coarse_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const CoarsePepperConfig& c,cudaStream_t s){salt_call(in,out,c,1,true,s);}
void coarse_salt_and_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const CoarseSaltAndPepperConfig& c,cudaStream_t s){salt_call(in,out,c,2,true,s);}
void coarse_salt_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const CoarseSaltPepperConfig& c,cudaStream_t s){salt_call(in,out,c,2,true,s);}
}
