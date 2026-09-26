#include "augmatch/sensor/noise.hpp"
#include <cmath>
#include <vector>
int main() {
  const int w=5,h=3,c=1; std::vector<float> in(w*h*c,0.1f), out(in.size());
  augmatch::FlareSource source{2.0f,1.0f,0.1f,0.0f,-1};
  augmatch::FlareHalo halo{2.0f,1.0f,2.0f,0.5f,-1};
  augmatch::FlareConfig fc{w,h,c,&source,1,&halo,1,nullptr,1,1.0f,0.0f,1.0f};
  augmatch::flare_f32(in.data(),out.data(),fc);
  if (std::abs(out[7]-0.55f)>1e-6f || out[0]!=in[0]) return 1;
  std::vector<augmatch::Ghost> ghosts{{1.0f,0.0f,1.0f,1.0f}};
  in[0]=0.8f; augmatch::GhostingConfig gc{w,h,c,ghosts.data(),1,nullptr,1,1.0f,0.0f,1.0f};
  augmatch::ghosting_f32(in.data(),out.data(),gc);
  if (std::abs(out[1]-0.8f)>1e-6f) return 2;
  augmatch::VeilingGlareConfig vg{w,h,c,nullptr,1,0.5f,0.0f,1.0f};
  augmatch::veiling_glare_f32(in.data(),out.data(),vg);
  if (std::abs(out[0]-1.0f)>1e-6f) return 3;
  in.assign(w*h*c,0.0f); in[7]=1.0f;
  augmatch::BloomConfig bc{w,h,c,1,0.5f,1.0f,nullptr,1,0.0f,1.0f};
  augmatch::bloom_f32(in.data(),out.data(),bc);
  if (!(out[6]>0.0f && out[7]>=1.0f)) return 4;
  bc.strength=0.0f; augmatch::bloom_f32(in.data(),out.data(),bc);
  if (out!=in) return 5;
  return 0;
}
