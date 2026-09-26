#include "augmatch/augmatch.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
struct Rgb { float r, g, b; };
Rgb blackbody(float kelvin) {
  const float t = kelvin / 100.0f;
  Rgb rgb{};
  if (t <= 66.0f) {
    rgb.r = 255.0f;
    rgb.g = 99.4708025861f * std::log(t) - 161.1195681661f;
    rgb.b = t <= 19.0f ? 0.0f : 138.5177312231f * std::log(t - 10.0f) - 305.0447927307f;
  } else {
    rgb.r = 329.698727446f * std::pow(t - 60.0f, -0.1332047592f);
    rgb.g = 288.1221695283f * std::pow(t - 60.0f, -0.0755148492f);
    rgb.b = 255.0f;
  }
  rgb.r = std::fmin(255.0f, std::fmax(0.0f, rgb.r));
  rgb.g = std::fmin(255.0f, std::fmax(0.0f, rgb.g));
  rgb.b = std::fmin(255.0f, std::fmax(0.0f, rgb.b));
  return rgb;
}
int rounded(float x) { return std::max(0, std::min(255, static_cast<int>(std::floor(x + 0.5f)))); }
void rejects(augmatch::ColorTemperatureErrorConfig c) {
  std::uint8_t in[3]{1, 2, 3}, out[3]{};
  bool did_throw = false;
  try { augmatch::color_temperature_error_u8(in, out, c); }
  catch (const std::invalid_argument&) { did_throw = true; }
  assert(did_throw);
}
}
int main() {
  constexpr int w=3, h=2, ch=4;
  std::vector<std::uint8_t> in(w*h*ch), out(in.size());
  for (std::size_t i=0; i<in.size(); ++i) in[i]=static_cast<std::uint8_t>((i*47+13)%256);
  // Neutral strength and the 6500 K reference preserve every byte.
  augmatch::color_temperature_error_u8(in.data(),out.data(),{w,h,ch,3000.0f,0.0f});
  assert(out==in);
  augmatch::color_temperature_error_u8(in.data(),out.data(),{w,h,ch,6500.0f,1.0f});
  assert(out==in);
  // Check the exact documented gain interpolation, rounding, clipping, and alpha.
  for (float temperature : {1000.0f, 3000.0f, 10000.0f, 40000.0f}) {
    for (float strength : {0.0f, 0.25f, 0.5f, 1.0f}) {
      augmatch::color_temperature_error_u8(in.data(),out.data(),{w,h,ch,temperature,strength});
      const Rgb reference=blackbody(6500.0f), color=blackbody(temperature);
      const float gain[3]={1+strength*(color.r/reference.r-1),1+strength*(color.g/reference.g-1),1+strength*(color.b/reference.b-1)};
      for (std::size_t p=0; p<in.size()/ch; ++p) {
        for (int k=0; k<3; ++k) assert(out[p*ch+k]==rounded(in[p*ch+k]*gain[k]));
        assert(out[p*ch+3]==in[p*ch+3]);
      }
    }
  }
  auto alias=in;
  std::vector<std::uint8_t> expected(in.size());
  augmatch::color_temperature_error_u8(in.data(),expected.data(),{w,h,ch,3000.0f,0.5f});
  augmatch::color_temperature_error_u8(alias.data(),alias.data(),{w,h,ch,3000.0f,0.5f});
  assert(alias==expected);
  rejects({1,1,3,999.0f,0.5f}); rejects({1,1,3,40001.0f,0.5f});
  rejects({1,1,3,6500.0f,-0.01f}); rejects({1,1,3,6500.0f,1.01f});
  rejects({1,1,2,6500.0f,0.5f});
}
