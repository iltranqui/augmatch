#include "augmatch/sensor/noise.hpp"
#include <cmath>
#include <vector>

int main() {
  constexpr int w=3,h=4,c=1;
  std::vector<float> input(w*h*c,0.0f), output(input.size()), repeat(input.size());
  input[1]=1.0f;
  augmatch::BrightPixel bright{1,0,1.0f,-1};
  augmatch::BloomingVerticalSmearConfig bloom{w,h,c,&bright,1,nullptr,0.5f,0.5f,0.5f,0.0f,1.0f,9};
  augmatch::blooming_vertical_smear_f32(input.data(),output.data(),bloom);
  const float expected[]{1.0f,0.125f,0.0625f,0.03125f};
  for(int y=0;y<h;++y) if(std::abs(output[y*w+1]-expected[y])>1e-6f) return 1;
  augmatch::blooming_vertical_smear_f32(input.data(),repeat.data(),bloom);
  if(output!=repeat) return 2;
  bloom.bright_pixels=nullptr; bloom.bright_pixel_count=0; bloom.smear_decay=1.0f;
  augmatch::blooming_vertical_smear_f32(input.data(),output.data(),bloom);
  for(int y=0;y<h;++y) if(output[y*w+1]<0.24f || output[y*w+1]>1.0f) return 3;
  const std::vector<float> column(w,0.2f); bloom.column_smear_map=column.data(); bloom.smear_strength=1.0f;
  augmatch::blooming_vertical_smear_f32(std::vector<float>(w*h,0.0f).data(),output.data(),bloom);
  for(int y=0;y<h;++y) for(int x=0;x<w;++x) if(std::abs(output[y*w+x]-0.2f)>1e-6f) return 4;

  std::vector<std::uint8_t> mask{0,1,0, 0,1,0, 0,0,0, 1,1,1};
  std::vector<float> dust_input(w*h*c,0.0f), dust_output(dust_input.size());
  for(int i=0;i<w*h;++i) dust_input[i]=static_cast<float>(i)/11.0f;
  augmatch::SensorDustOpaqueMaskConfig dust{w,h,c,mask.data(),0,0.75f};
  augmatch::sensor_dust_opaque_mask_f32(dust_input.data(),dust_output.data(),dust);
  if(std::abs(dust_output[1]-0.75f)>1e-6f || std::abs(dust_output[0]-dust_input[0])>1e-6f) return 5;
  dust.blur_radius=1;
  augmatch::sensor_dust_opaque_mask_f32(dust_input.data(),dust_output.data(),dust);
  // The opaque center uses only its unmasked 3x3 neighbors.
  const float expected_blur=(dust_input[0]+dust_input[2]+dust_input[3]+dust_input[5])/4.0f;
  if(std::abs(dust_output[1]-expected_blur)>1e-6f) return 6;
  dust.mask=nullptr; augmatch::sensor_dust_opaque_mask_f32(dust_input.data(),dust_output.data(),dust);
  if(dust_output!=dust_input) return 7;
  for(float value:dust_output) if(!std::isfinite(value)) return 8;
  return 0;
}
