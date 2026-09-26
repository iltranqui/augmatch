#include "augmatch/noise.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

int main() {
  constexpr int w = 4, h = 1, channels = 2, levels = 4;
  const std::vector<float> input{0.05f,0.05f, 0.20f,0.20f, 0.80f,0.80f, 0.95f,0.95f};
  std::vector<float> output(input.size());
  // Endpoint half-widths make the all-ones DNL LUT match nearest ADC codes.
  const std::vector<float> dnl_lut{1.0f,3.0f,1.0f,1.0f, 1.0f,1.0f,1.0f,1.0f};
  augmatch::AdcDifferentialNonLinearityConfig dnl{w,h,channels,levels,dnl_lut.data(),0.0f,17};
  augmatch::adc_differential_non_linearity_f32(input.data(),output.data(),dnl);
  const float expected_dnl[]{0.0f,0.0f,1.0f/3.0f,1.0f/3.0f,2.0f/3.0f,2.0f/3.0f,1.0f,1.0f};
  for (std::size_t i=0;i<input.size();++i) if (std::abs(output[i]-expected_dnl[i])>1e-6f) return 1;
  dnl.lut = nullptr; dnl.stddev = 0.12f;
  std::vector<float> repeat(output.size());
  augmatch::adc_differential_non_linearity_f32(input.data(),output.data(),dnl);
  augmatch::adc_differential_non_linearity_f32(input.data(),repeat.data(),dnl);
  if (output != repeat) return 2;
  for (float value : output) if (value < 0.0f || value > 1.0f) return 3;

  const std::vector<float> inl_lut{0.0f,1.0f,-1.0f,4.0f, 0.0f,0.0f,0.0f,0.0f};
  const std::vector<float> inl_input{0.34f,0.0f,0.67f,0.34f,0.34f,0.0f,0.67f,0.34f};
  augmatch::AdcIntegralNonLinearityConfig inl{w,h,channels,levels,inl_lut.data(),0.0f,3};
  augmatch::adc_integral_non_linearity_f32(inl_input.data(),output.data(),inl);
  const float expected_inl[]{2.0f/3.0f,0.0f,1.0f/3.0f,1.0f/3.0f,2.0f/3.0f,0.0f,1.0f/3.0f,1.0f/3.0f};
  for (std::size_t i=0;i<output.size();++i) if (std::abs(output[i]-expected_inl[i])>1e-6f) return 4;

  const std::vector<float> capacity{0.25f,0.5f,1.2f,2.0f};
  const std::vector<float> well_input{0.1f,0.7f,1.0f,1.1f};
  std::vector<float> well_output(well_input.size());
  augmatch::SensorWellCapacityVariationConfig well{w,h,1,capacity.data(),0.0f,9};
  augmatch::sensor_well_capacity_variation_f32(well_input.data(),well_output.data(),well);
  const float expected_well[]{0.1f,0.5f,1.0f,1.0f};
  for (std::size_t i=0;i<well_output.size();++i) if (std::abs(well_output[i]-expected_well[i])>1e-6f) return 5;

  const int sample_count=2048;
  std::vector<float> statistical_input(sample_count,1.0f), statistical_a(sample_count), statistical_b(sample_count);
  augmatch::SensorWellCapacityVariationConfig seeded{sample_count,1,1,nullptr,0.15f,12345};
  augmatch::sensor_well_capacity_variation_f32(statistical_input.data(),statistical_a.data(),seeded);
  augmatch::sensor_well_capacity_variation_f32(statistical_input.data(),statistical_b.data(),seeded);
  if (statistical_a != statistical_b) return 6;
  float mean=0.0f; int changed=0;
  for (float value : statistical_a) { if (value < 0.0f || value > 1.0f) return 7; mean += value; if (value < 0.999f) ++changed; }
  mean /= sample_count;
  if (changed < 100 || mean < 0.90f || mean > 1.0f) return 8;
  return 0;
}
