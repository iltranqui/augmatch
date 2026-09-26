#include "augmatch/bayer.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
using namespace augmatch;
int main() {
  constexpr int w=11,h=9; std::vector<std::uint8_t> raw(w*h), base(w*h*3), out(w*h*3), again(w*h*3);
  for (int i=0;i<w*h;++i) raw[i]=static_cast<std::uint8_t>((i*29+i/w*17)&255);
  BayerConfig bc{w,h,BayerPattern::RGGB}; bayer_demosaic_edge_aware_u8(raw.data(),base.data(),bc);
  DirectionalDemosaicingArtifactsConfig directional{w,h,BayerPattern::RGGB,0,0.0f}; bayer_directional_demosaicing_artifacts_u8(raw.data(),out.data(),directional); assert(out==base);
  FalseColorZipperArtifactsConfig zipper{w,h,BayerPattern::RGGB,0.0f}; bayer_false_color_zipper_artifacts_u8(raw.data(),out.data(),zipper); assert(out==base);
  DemosaicingAliasingConfig aliasing{w,h,BayerPattern::RGGB,3,0.25f,0.0f}; bayer_demosaicing_aliasing_u8(raw.data(),out.data(),aliasing); assert(out==base);
  DemosaicingRingingConfig ringing{w,h,BayerPattern::RGGB,0.0f}; bayer_demosaicing_ringing_u8(raw.data(),out.data(),ringing); assert(out==base);
  DemosaicingNoiseAmplificationConfig noise{w,h,BayerPattern::RGGB,0.0f,2.0f,123}; bayer_demosaicing_noise_amplification_u8(raw.data(),out.data(),noise); assert(out==base);
  directional.strength=2.0f; bayer_directional_demosaicing_artifacts_u8(raw.data(),out.data(),directional); bayer_directional_demosaicing_artifacts_u8(raw.data(),again.data(),directional); assert(out==again); assert(std::all_of(out.begin(),out.end(),[](std::uint8_t v){return v<=255;}));
  noise.noise_stddev=12.0f; bayer_demosaicing_noise_amplification_u8(raw.data(),out.data(),noise); bayer_demosaicing_noise_amplification_u8(raw.data(),again.data(),noise); assert(out==again);
  return 0;
}
