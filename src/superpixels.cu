#include "augmatch/superpixels.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace augmatch { namespace {
__global__ void superpixels_kernel(const std::uint8_t* in,std::uint8_t* out,SuperpixelConfig c){
  const std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;
  if(i>=n)return;
  const int ch=static_cast<int>(i%static_cast<std::size_t>(c.channels));
  const int x=static_cast<int>((i/c.channels)%c.width);
  const int y=static_cast<int>(i/(static_cast<std::size_t>(c.width)*c.channels));
  const int x0=(x/c.cell_width)*c.cell_width, y0=(y/c.cell_height)*c.cell_height;
  const int x1=(c.cell_width>c.width-x0)?c.width:x0+c.cell_width;
  const int y1=(c.cell_height>c.height-y0)?c.height:y0+c.cell_height;
  const unsigned long long count=static_cast<unsigned long long>(x1-x0)*static_cast<unsigned long long>(y1-y0);
  unsigned long long sum=0;
  for(int py=y0;py<y1;++py) for(int px=x0;px<x1;++px)
    sum+=in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch];
  out[i]=static_cast<std::uint8_t>((sum+count/2u)/count);
}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(std::string("superpixels CUDA: ")+cudaGetErrorString(error));}
}
void superpixels_u8(const std::uint8_t* in,std::uint8_t* out,const SuperpixelConfig& c,cudaStream_t stream){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.cell_width<=0||c.cell_height<=0)
    throw std::invalid_argument("invalid superpixel configuration");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;
  superpixels_kernel<<<static_cast<unsigned int>((n+255u)/256u),256,0,stream>>>(in,out,c);
  check(cudaGetLastError());
}
}
