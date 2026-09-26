#include "augmatch/core/convert.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__device__ unsigned char sat(float x){return (unsigned char)fminf(255.f,fmaxf(0.f,rintf(x)));}
__global__ void to_float_kernel(const unsigned char* in,float* out,ConvertConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=in[i]/255.f;}
__global__ void from_float_kernel(const float* in,unsigned char* out,ConvertConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=sat(in[i]*255.f);}
__global__ void norm_kernel(const unsigned char* in,float* out,NormalizeConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n){int ch=i%c.channels;float m=ch==0?c.mean0:ch==1?c.mean1:c.mean2,s=ch==0?c.std0:ch==1?c.std1:c.std2;out[i]=(in[i]/255.f-m)/s;}}
void check(cudaError_t e,const char* p){if(e!=cudaSuccess)throw std::runtime_error(std::string(p)+": "+cudaGetErrorString(e));}
void valid(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid conversion dimensions");}
}
void to_float_u8(const std::uint8_t* in,float* out,const ConvertConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null to-float buffer");int n=c.width*c.height*c.channels;to_float_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"to_float_kernel");}
void from_float_u8(const float* in,std::uint8_t* out,const ConvertConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null from-float buffer");int n=c.width*c.height*c.channels;from_float_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"from_float_kernel");}
void normalize_u8_f32(const std::uint8_t* in,float* out,const NormalizeConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.std0==0||c.std1==0||c.std2==0)throw std::invalid_argument("invalid normalize configuration");int n=c.width*c.height*c.channels;norm_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"norm_kernel");}
__device__ float tensor_value(unsigned char value, int channel, ToTensorV2Config c){
  const float normalized=value/255.0f;
  if(!c.normalize)return normalized;
  const float mean=channel==0?c.mean0:channel==1?c.mean1:c.mean2;
  const float st=channel==0?c.std0:channel==1?c.std1:c.std2;
  return (normalized-mean)/st;
}
__global__ void tensor_kernel(const unsigned char* in,float* out,ToTensorV2Config c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int n=c.width*c.height*c.channels;
  if(i<n){
    const int channel=i%c.channels;
    const int pixel=i/c.channels;
    const int x=pixel%c.width;
    const int y=pixel/c.width;
    const int output=(channel*c.height+y)*c.width+x;
    out[output]=tensor_value(in[i],channel,c);
  }
}
void validate_tensor_config(const ToTensorV2Config& c){
  valid(c.width,c.height,c.channels);
  if(c.normalize&&(!isfinite(c.mean0)||!isfinite(c.mean1)||!isfinite(c.mean2)||!isfinite(c.std0)||!isfinite(c.std1)||!isfinite(c.std2)||c.std0==0.f||c.std1==0.f||c.std2==0.f))
    throw std::invalid_argument("invalid ToTensorV2 normalization configuration");
}
void validate_tensor_views(const ImageView& input,const MutableTensorViewF32& output){
  if(!input.valid()||input.type!=DataType::UInt8||input.layout!=Layout::HWC||!input.contiguous())throw std::invalid_argument("ToTensorV2 requires contiguous HWC uint8 input");
  if(!output.valid()||!output.contiguous())throw std::invalid_argument("ToTensorV2 requires contiguous CHW float32 output");
  if(input.width!=output.width||input.height!=output.height||input.channels!=output.channels)throw std::invalid_argument("ToTensorV2 input and output shapes differ");
  if(input.memory!=MemorySpace::Device||output.memory!=MemorySpace::Device)throw std::invalid_argument("CUDA ToTensorV2 requires device views");
}
void to_tensor_v2_u8_f32(const std::uint8_t* in,float* out,const ToTensorV2Config& c,cudaStream_t s){
  validate_tensor_config(c);if(!in||!out)throw std::invalid_argument("null ToTensorV2 buffer");
  const int n=c.width*c.height*c.channels;tensor_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"tensor_kernel");
}
void to_tensor_v2(const std::uint8_t* in,float* out,const ToTensorV2Config& c,cudaStream_t s){to_tensor_v2_u8_f32(in,out,c,s);}
void to_tensor_chw_f32(const std::uint8_t* in,float* out,const ToTensorV2Config& c,cudaStream_t s){to_tensor_v2_u8_f32(in,out,c,s);}
void to_tensor_v2(const ImageView& input,const MutableTensorViewF32& output,const TensorNormalizeConfig& n,cudaStream_t s){
  validate_tensor_views(input,output);
  const ToTensorV2Config c{input.width,input.height,input.channels,n.enabled,n.mean0,n.mean1,n.mean2,n.std0,n.std1,n.std2};
  to_tensor_v2_u8_f32(static_cast<const std::uint8_t*>(input.data),output.data,c,s);
}
__device__ float tensor_3d_value(unsigned char value, int channel, ToTensor3DConfig c){
  const float normalized=value/255.0f;
  if(!c.normalize)return normalized;
  const float mean=channel==0?c.mean0:channel==1?c.mean1:c.mean2;
  const float st=channel==0?c.std0:channel==1?c.std1:c.std2;
  return (normalized-mean)/st;
}
__global__ void tensor_3d_kernel(const unsigned char* in,float* out,ToTensor3DConfig c){
  const std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.depth*c.channels;
  if(i<n){
    const int channel=static_cast<int>(i%c.channels);
    const std::size_t voxel=i/c.channels;
    const int x=static_cast<int>(voxel%c.width);
    const std::size_t yz=voxel/c.width;
    const int y=static_cast<int>(yz%c.height);
    const int depth=static_cast<int>(yz/c.height);
    const std::size_t output=(static_cast<std::size_t>(channel)*c.depth+depth)*c.height*c.width+
                             static_cast<std::size_t>(y)*c.width+x;
    out[output]=tensor_3d_value(in[i],channel,c);
  }
}
void validate_tensor_3d_config(const ToTensor3DConfig& c){
  valid(c.width,c.height,c.channels);
  if(c.depth<=0)throw std::invalid_argument("invalid ToTensor3D dimensions");
  if(c.normalize&&(!isfinite(c.mean0)||!isfinite(c.mean1)||!isfinite(c.mean2)||!isfinite(c.std0)||!isfinite(c.std1)||!isfinite(c.std2)||c.std0==0.f||c.std1==0.f||c.std2==0.f))
    throw std::invalid_argument("invalid ToTensor3D normalization configuration");
}
void to_tensor_3d_u8_f32(const std::uint8_t* in,float* out,const ToTensor3DConfig& c,cudaStream_t s){
  validate_tensor_3d_config(c);if(!in||!out)throw std::invalid_argument("null ToTensor3D buffer");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.depth*c.channels;
  tensor_3d_kernel<<<static_cast<unsigned int>((n+255)/256),256,0,s>>>(in,out,c);check(cudaGetLastError(),"tensor_3d_kernel");
}
void to_tensor_3d(const std::uint8_t* in,float* out,const ToTensor3DConfig& c,cudaStream_t s){to_tensor_3d_u8_f32(in,out,c,s);}
void to_tensor_cdhw_f32(const std::uint8_t* in,float* out,const ToTensor3DConfig& c,cudaStream_t s){to_tensor_3d_u8_f32(in,out,c,s);}
}
