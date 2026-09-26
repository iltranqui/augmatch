#include "augmatch/bayer.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
int main(){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;const int w=8,h=8;std::vector<float> host(w*h,0.25f),result(w*h);float *di=nullptr,*do_=nullptr;if(cudaMalloc(&di,host.size()*sizeof(float))!=cudaSuccess||cudaMalloc(&do_,host.size()*sizeof(float))!=cudaSuccess)return 77;cudaMemcpy(di,host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice);try{auto map=augmatch::bayer_plane_map(augmatch::BayerPattern::RGGB);augmatch::BayerPlaneGainConfig c{w,h,map,{1,2,3,4},0,1};augmatch::bayer_plane_gain_f32(di,do_,c);if(cudaDeviceSynchronize()!=cudaSuccess)return 77;cudaMemcpy(result.data(),do_,result.size()*sizeof(float),cudaMemcpyDeviceToHost);for(int y=0;y<h;++y)for(int x=0;x<w;++x){const int p=map.plane[((y&1)<<1)|(x&1)];if(std::fabs(result[y*w+x]-.25f*(p+1))>1e-5f)return 1;}}catch(const std::exception& e){const std::string s=e.what();if(s.find("unsupported toolchain")!=std::string::npos||s.find("no kernel image")!=std::string::npos)return 77;return 1;}cudaFree(di);cudaFree(do_);std::puts("PASS: CUDA CFA effects");return 0;}
