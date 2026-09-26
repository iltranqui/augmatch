#include "augmatch/environmental.hpp"
#include <cmath>
#include <vector>
int main(){
  const int w=3,h=2,c=2; std::vector<float> in(w*h*c,0.2f),out(in.size());
  std::vector<float> mask{0,1,0,1,0,1};
  augmatch::EnvironmentalVeilConfig v{w,h,c,mask.data(),1,0.5f,1.0f,0,1};
  augmatch::atmospheric_haze_f32(in.data(),out.data(),v);
  if(std::abs(out[0]-.2f)>1e-6f||std::abs(out[2]-.6f)>1e-6f)return 1;
  v.map=nullptr; v.strength=0; augmatch::smoke_veil_f32(in.data(),out.data(),v); if(out!=in)return 2;
  augmatch::AcquisitionGainConfig g{w,h,c,3,0,1}; augmatch::overexposure_f32(in.data(),out.data(),g); if(std::abs(out[0]-.6f)>1e-6f)return 3;
  g.gain=0; augmatch::low_light_amplification_f32(in.data(),out.data(),g); for(float x:out)if(x!=0)return 4;
  return 0;
}
