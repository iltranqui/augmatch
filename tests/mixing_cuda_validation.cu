#include "augmatch/mixing.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <exception>
#include <vector>
#include <string>
int main(){try {int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;const int n=4;std::vector<unsigned char>a(n,10),b(n,110),out(n);unsigned char *da=nullptr,*db=nullptr,*do_=nullptr;cudaMalloc(&da,n);cudaMalloc(&db,n);cudaMalloc(&do_,n);cudaMemcpy(da,a.data(),n,cudaMemcpyHostToDevice);cudaMemcpy(db,b.data(),n,cudaMemcpyHostToDevice);augmatch::ImageSourceU8 sa{da,2,2,1},sb{db,2,2,1};augmatch::mixup_u8(sa,sb,do_,{2,2,1,.25f,0,false});if(cudaDeviceSynchronize()!=cudaSuccess)return 77;cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost);for(auto x:out)assert(x==35);cudaFree(da);cudaFree(db);cudaFree(do_);return 0;} catch(const std::exception& e){return std::string(e.what()).find("unsupported toolchain")!=std::string::npos?77:1;}}
