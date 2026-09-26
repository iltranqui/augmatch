#include "augmatch/isp_artifacts.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
#include <iostream>
int main(){
  const int w=5,h=4,c=1; std::vector<std::uint8_t> in(w*h),out(w*h),again(w*h);
  for(int y=0;y<h;++y)for(int x=0;x<w;++x)in[y*w+x]=static_cast<std::uint8_t>(x<2?32:220);
  augmatch::EdgeOversharpeningConfig edge{w,h,c,0.25f}; augmatch::edge_oversharpening_u8(in.data(),out.data(),edge); for(auto v:out)assert(v<=255);
  augmatch::UnsharpMaskHalosConfig unsharp{w,h,c,1,1.0f,0.5f}; augmatch::unsharp_mask_halos_u8(in.data(),out.data(),unsharp);
  augmatch::LaplacianHalosConfig lap{w,h,c,0.5f}; augmatch::laplacian_halos_u8(in.data(),out.data(),lap);
  augmatch::RingingNearStrongEdgesConfig ring{w,h,c,0.5f,1.0f,20.0f}; augmatch::ringing_near_strong_edges_u8(in.data(),out.data(),ring);
  augmatch::LocalContrastEnhancementArtifactsConfig contrast{w,h,c,1,0.5f,1.0f}; augmatch::local_contrast_enhancement_artifacts_u8(in.data(),out.data(),contrast);
  augmatch::HaloingFromToneMappingConfig tone{w,h,c,2,1.5f,0.5f,1.0f}; augmatch::haloing_from_tone_mapping_u8(in.data(),out.data(),tone);
  augmatch::LocalSharpeningNoiseAmplificationConfig noise{w,h,c,1,0.5f,2.0f}; augmatch::local_sharpening_noise_amplification_u8(in.data(),out.data(),noise);
  augmatch::HighFrequencyAttenuationConfig attenuation{w,h,c,1,1.0f}; augmatch::high_frequency_attenuation_u8(in.data(),out.data(),attenuation);
  augmatch::DetailSmearingConfig smear{w,h,c,1,1.0f,1.0f,false}; augmatch::detail_smearing_u8(in.data(),out.data(),smear);
  augmatch::OvershootAndUndershootConfig bounds{w,h,c,1,1.0f,1.0f,8.0f,8.0f}; augmatch::overshoot_and_undershoot_u8(in.data(),out.data(),bounds);
  for(auto v:out)assert(v<=255);
  augmatch::EdgeOversharpeningConfig identity{w,h,c,0.0f}; augmatch::edge_oversharpening_u8(in.data(),out.data(),identity); assert(out==in);
  augmatch::DetailSmearingConfig deterministic{w,h,c,1,1.0f,0.5f,true}; augmatch::detail_smearing_u8(in.data(),out.data(),deterministic); augmatch::detail_smearing_u8(in.data(),again.data(),deterministic); assert(out==again);
  std::cout<<"ISP artifact contracts passed\n"; return 0;
}
