#include "augmatch/isp_artifacts.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
int main(){
  int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
  const int w=9,h=7,c=1; const std::size_t n=static_cast<std::size_t>(w)*h*c; std::vector<std::uint8_t> host(n,83),result(n);
  std::uint8_t *in=nullptr,*out=nullptr; if(cudaMalloc(&in,n)!=cudaSuccess||cudaMalloc(&out,n)!=cudaSuccess)return 77; if(cudaMemcpy(in,host.data(),n,cudaMemcpyHostToDevice)!=cudaSuccess)return 77;
  augmatch::edge_oversharpening_u8(in,out,{w,h,c,.25f});
  augmatch::unsharp_mask_halos_u8(in,out,{w,h,c,1,1.0f,.5f});
  augmatch::laplacian_halos_u8(in,out,{w,h,c,.25f});
  augmatch::ringing_near_strong_edges_u8(in,out,{w,h,c,.5f,1.0f,16.0f});
  augmatch::local_contrast_enhancement_artifacts_u8(in,out,{w,h,c,1,.5f,1.0f});
  augmatch::haloing_from_tone_mapping_u8(in,out,{w,h,c,2,1.5f,.5f,1.0f});
  augmatch::local_sharpening_noise_amplification_u8(in,out,{w,h,c,1,.5f,2.0f});
  augmatch::high_frequency_attenuation_u8(in,out,{w,h,c,1,1.0f});
  augmatch::detail_smearing_u8(in,out,{w,h,c,1,1.0f,.5f,true});
  augmatch::overshoot_and_undershoot_u8(in,out,{w,h,c,1,1.0f,1.0f,8.0f,8.0f});
  augmatch::gradient_plus_laplacian_noise_u8(in,out,{w,h,c,0.0f,0.0f,0.0f,7});
  augmatch::deblocking_halos_u8(in,out,{w,h,c,8,1,0.0f});
  augmatch::demosaicing_edge_artifacts_u8(in,out,{w,h,c,0.0f,1.0f});
  augmatch::edge_dependent_quantization_u8(in,out,{w,h,c,1,0.0f});
  augmatch::edge_dependent_compression_error_u8(in,out,{w,h,c,8,0.0f,0.0f,7});
  augmatch::gradient_reversal_u8(in,out,{w,h,c,0.0f,1.0f});
  augmatch::clipped_edge_ringing_u8(in,out,{w,h,c,0.0f,1.0f,16.0f});
  if(cudaDeviceSynchronize()!=cudaSuccess)return 1; if(cudaMemcpy(result.data(),out,n,cudaMemcpyDeviceToHost)!=cudaSuccess)return 1; for(auto v:result)if(v!=83)return 2; cudaFree(in);cudaFree(out);return 0;
}
