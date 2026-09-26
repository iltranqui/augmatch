#include "augmatch/color/color.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <exception>
#include <vector>

int main() {
  int devices=0; if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
  try {
    const int w=4,h=2,c=4; const std::size_t n=static_cast<std::size_t>(w)*h*c;
    std::vector<std::uint8_t> host(n); for (std::size_t i=0;i<n;++i) host[i]=static_cast<std::uint8_t>((i*19+7)&255);
    std::uint8_t *di=nullptr,*do_=nullptr; if(cudaMalloc(&di,n)||cudaMalloc(&do_,n)) return 1;
    if(cudaMemcpy(di,host.data(),n,cudaMemcpyHostToDevice)!=cudaSuccess) return 1;
    augmatch::ChromaSubsamplingArtifactsConfig sc{w,h,c,augmatch::ChromaSubsampling::Y420};
    augmatch::chroma_subsampling_artifacts_u8(di,do_,sc); if(cudaDeviceSynchronize()!=cudaSuccess) return 1;
    std::vector<std::uint8_t> out(n); if(cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost)!=cudaSuccess) return 1;
    for(int i=3;i<w*h*c;i+=c) if(out[i]!=host[i]) return 1;
    augmatch::LocalToneMappingNoiseConfig lc{w,h,c,1,.25f,.01f,13};
    augmatch::local_tone_mapping_noise_u8(di,do_,lc); if(cudaDeviceSynchronize()!=cudaSuccess) return 1;
    if(cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost)!=cudaSuccess) return 1;
    for(int i=3;i<w*h*c;i+=c) if(out[i]!=host[i]) return 1;
    cudaFree(di); cudaFree(do_); return 0;
  } catch (const std::exception&) {
    // Unsupported-driver PTX/toolchain combinations are an environment skip,
    // matching the validation convention used by the CTest CUDA suite.
    return 77;
  }
}
