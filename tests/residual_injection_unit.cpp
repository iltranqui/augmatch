#include "augmatch/isp/isp_artifacts.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  const int w=3,h=2,c=2;
  std::vector<std::uint8_t> in{10,20,30,40,50,60,70,80,90,100,110,120};
  std::vector<std::uint8_t> out(in.size());
  std::vector<float> map{10.0f,-20.0f,30.0f,-40.0f,50.0f,-60.0f};
  augmatch::LaplacianResidualInjectionConfig lc{w,h,c,map.data(),1,1.0f};
  augmatch::laplacian_residual_injection_u8(in.data(),out.data(),lc);
  for (std::size_t i=0;i<in.size();++i) assert(out[i]==static_cast<std::uint8_t>(std::lround(std::max(0.0f,std::min(255.0f,float(in[i])+map[i/2])))));
  augmatch::SobelResidualInjectionConfig sc{w,h,c,map.data(),1,0.5f,0};
  augmatch::sobel_residual_injection_u8(in.data(),out.data(),sc);
  for (std::size_t i=0;i<in.size();++i) assert(out[i]<=255);
  augmatch::HighPassResidualInjectionConfig hc{w,h,c,nullptr,1,0.0f};
  augmatch::high_pass_residual_injection_u8(in.data(),out.data(),hc); assert(out==in);
  bool threw=false; try { auto bad=lc; bad.map_channels=3; augmatch::laplacian_residual_injection_u8(in.data(),out.data(),bad); } catch(const std::invalid_argument&) { threw=true; } assert(threw);
  std::cout << "Residual injection contracts passed\n";
}
