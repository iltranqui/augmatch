#include "augmatch/imgcorruptlike.hpp"
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  const int w=4,h=3,c=3;
  std::vector<std::uint8_t> input(static_cast<std::size_t>(w)*h*c,100), a(input.size()), b(input.size());
  augmatch::SpeckleNoiseConfig noise{w,h,c,0.0f,0.2f,42};
  augmatch::speckle_noise_u8(input.data(),a.data(),noise);
  augmatch::speckle_noise_u8(input.data(),b.data(),noise);
  assert(a==b);
  augmatch::FogConfig fog{w,h,c,0.5f,1.0f,nullptr,7};
  augmatch::fog_u8(input.data(),a.data(),fog);
  assert(a!=input);
  augmatch::FogConfig identity{w,h,c,0.0f,1.0f,nullptr,7};
  augmatch::fog_u8(input.data(),a.data(),identity);
  assert(a==input);
  augmatch::ContrastConfig contrast{w,h,c,1.0f};
  augmatch::contrast_u8(input.data(),a.data(),contrast);
  assert(a==input);
  augmatch::BrightnessConfig dark{w,h,c,0.0f};
  augmatch::brightness_u8(input.data(),a.data(),dark);
  for(auto x:a) assert(x==0);
  augmatch::SaturateConfig sat{w,h,c,0.0f};
  augmatch::saturate_u8(input.data(),a.data(),sat);
  assert(a[0]==a[1] && a[1]==a[2]);
  augmatch::PixelateConfig pixel{w,h,c,2};
  augmatch::pixelate_u8(input.data(),a.data(),pixel);
  assert(a==input);
  return 0;
}
