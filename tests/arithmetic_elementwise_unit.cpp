#include "augmatch/color/arithmetic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
  constexpr int width = 3, height = 2, channels = 1;
  const std::vector<std::uint8_t> input{0, 10, 100, 200, 255, 50};
  const std::vector<float> offsets{1.5f, -20.0f, 2.5f, 100.0f, -300.0f, 0.0f};
  std::vector<std::uint8_t> output(input.size());
  augmatch::add_elementwise_u8(input.data(), output.data(),
                               {width, height, channels, offsets.data(), 0.0f,
                                0.0f, 0});
  const std::vector<std::uint8_t> add_expected{2, 0, 103, 255, 0, 50};
  if (output != add_expected) return 1;

  const std::vector<float> factors{1.5f, 0.5f, 2.5f, 0.0f, 0.25f, 3.0f};
  augmatch::multiply_elementwise_u8(input.data(), output.data(),
                                    {width, height, channels, factors.data(),
                                     0.0f, 0.0f, 0});
  const std::vector<std::uint8_t> multiply_expected{0, 5, 250, 0, 64, 150};
  if (output != multiply_expected) return 2;

  std::vector<float> generated_a(input.size()), generated_b(input.size());
  augmatch::make_add_elementwise_values(generated_a.data(), width, height,
                                        channels, -12.0f, 18.0f, 77);
  augmatch::make_add_elementwise_values(generated_b.data(), width, height,
                                        channels, -12.0f, 18.0f, 77);
  if (generated_a != generated_b) return 3;
  for (float value : generated_a)
    if (!std::isfinite(value) || value < -12.0f || value >= 18.0f) return 4;

  std::vector<float> generated_multiply(input.size());
  augmatch::make_multiply_elementwise_values(
      generated_multiply.data(), width, height, channels, 0.5f, 1.5f, 91);
  for (float value : generated_multiply)
    if (!std::isfinite(value) || value < 0.5f || value >= 1.5f) return 5;
  std::vector<std::uint8_t> seeded_a(input.size()), seeded_b(input.size());
  const augmatch::AddElementwiseConfig seeded_config{width, height, channels,
                                                      nullptr, -12.0f, 18.0f,
                                                      77};
  augmatch::add_elementwise_u8(input.data(), seeded_a.data(), seeded_config);
  augmatch::add_elementwise_u8(input.data(), seeded_b.data(), seeded_config);
  if (seeded_a != seeded_b) return 6;

  bool rejected = false;
  try {
    augmatch::multiply_elementwise_u8(input.data(), output.data(),
                                      {width, height, channels, nullptr, -1.0f,
                                       1.0f, 0});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 7;
  return 0;
}
