#include "augmatch/color/arithmetic.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
  constexpr int width = 4, height = 2, channels = 1;
  const std::vector<std::uint8_t> input{10, 20, 30, 40, 50, 60, 70, 80};
  const std::vector<std::uint8_t> mask{0, 1, 0, 1, 1, 0, 1, 0};
  const std::vector<std::uint8_t> values{101, 102, 103, 104, 105, 106, 107, 108};
  std::vector<std::uint8_t> output(input.size());
  augmatch::replace_elementwise_u8(input.data(), output.data(),
                                   {width, height, channels, mask.data(), values.data(), 0, 0, 0});
  if (output != std::vector<std::uint8_t>({10, 102, 30, 104, 105, 60, 107, 80})) return 1;

  augmatch::replace_elementwise_u8(input.data(), output.data(),
                                   {width, height, channels, nullptr, nullptr, 255, 1.0f, 99});
  if (output != std::vector<std::uint8_t>(8, 255)) return 2;
  augmatch::replace_elementwise_u8(input.data(), output.data(),
                                   {width, height, channels, nullptr, nullptr, 255, 0.0f, 99});
  if (output != input) return 3;

  std::vector<std::uint8_t> mask_a(input.size()), mask_b(input.size());
  augmatch::make_replace_elementwise_mask(mask_a.data(), width, height, channels, 0.25f, 1234);
  augmatch::make_replace_elementwise_mask(mask_b.data(), width, height, channels, 0.25f, 1234);
  if (mask_a != mask_b) return 4;
  std::vector<std::uint8_t> large_mask(100000);
  augmatch::make_replace_elementwise_mask(large_mask.data(), 100000, 1, 1, 0.25f, 1234);
  std::size_t selected = 0;
  for (std::uint8_t value : large_mask) selected += value != 0;
  if (selected < 24000 || selected > 26000) return 5;

  augmatch::impulse_noise_u8(input.data(), output.data(),
                             {width, height, channels, mask.data(), values.data(), 0.0f, 0.5f, 0});
  if (output != std::vector<std::uint8_t>({10, 102, 30, 104, 105, 60, 107, 80})) return 6;
  augmatch::impulse_noise_u8(input.data(), output.data(),
                             {width, height, channels, nullptr, nullptr, 1.0f, 1.0f, 77});
  if (output != std::vector<std::uint8_t>(8, 255)) return 7;
  std::vector<std::uint8_t> impulse_a(input.size()), impulse_b(input.size());
  const augmatch::ImpulseNoiseConfig impulse_config{width, height, channels, nullptr, nullptr, 0.2f, 0.5f, 77};
  augmatch::impulse_noise_u8(input.data(), impulse_a.data(), impulse_config);
  augmatch::impulse_noise_u8(input.data(), impulse_b.data(), impulse_config);
  if (impulse_a != impulse_b) return 8;
  std::vector<std::uint8_t> large_input(100000, 128), large_output(100000);
  augmatch::impulse_noise_u8(large_input.data(), large_output.data(),
                             {100000, 1, 1, nullptr, nullptr, 0.2f, 0.5f, 77});
  std::size_t changed = 0;
  for (std::uint8_t value : large_output) changed += value != 128;
  if (changed < 18000 || changed > 22000) return 9;

  bool rejected = false;
  try {
    augmatch::replace_elementwise_u8(input.data(), output.data(),
                                     {width, height, channels, nullptr, values.data(), 0, 0.5f, 0});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 10;
  return 0;
}
