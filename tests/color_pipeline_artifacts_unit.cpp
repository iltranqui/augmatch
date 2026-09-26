#include "augmatch/color/color.hpp"
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  const int w=4,h=2,c=4;
  std::vector<std::uint8_t> input(static_cast<std::size_t>(w)*h*c), output(input.size());
  for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
    const std::size_t p=(static_cast<std::size_t>(y)*w+x)*c;
    input[p]=static_cast<std::uint8_t>(x*60); input[p+1]=static_cast<std::uint8_t>(y*100+20);
    input[p+2]=static_cast<std::uint8_t>(255-x*40); input[p+3]=77;
  }
  augmatch::chroma_subsampling_artifacts_u8(input.data(),output.data(),{w,h,c,augmatch::ChromaSubsampling::Y444});
  assert(output==input);
  augmatch::chroma_subsampling_artifacts_u8(input.data(),output.data(),{w,h,c,augmatch::ChromaSubsampling::Y420});
  for (int y=0;y<h;++y) for (int x=0;x<w;++x) assert(output[(static_cast<std::size_t>(y)*w+x)*c+3]==77);
  std::vector<std::uint8_t> local(output.size());
  augmatch::LocalToneMappingNoiseConfig cfg{w,h,c,1,0.0f,0.0f,9};
  augmatch::local_tone_mapping_noise_u8(input.data(),local.data(),cfg);
  // A zero-strength, zero-noise local operation is a YCbCr round trip.
  for (std::size_t i=0;i<local.size();i+=c) { assert(local[i+3]==77); }
  cfg.tone_strength=1.0f; cfg.noise_stddev=0.0f;
  augmatch::local_tone_mapping_noise_u8(input.data(),local.data(),cfg);
  for (int i=0;i<w*h;++i) assert(local[static_cast<std::size_t>(i)*c+3]==77);
  std::vector<std::uint8_t> repeat(local.size());
  augmatch::local_tone_mapping_noise_u8(input.data(),repeat.data(),{w,h,c,1,0.25f,0.02f,42});
  augmatch::local_tone_mapping_noise_u8(input.data(),local.data(),{w,h,c,1,0.25f,0.02f,42});
  assert(local==repeat);
  return 0;
}
