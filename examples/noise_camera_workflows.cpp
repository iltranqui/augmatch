#include <cstdint>
#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"

int main() {
  constexpr int width = 8, height = 4, channels = 3;
  std::vector<std::uint8_t> input(width * height * channels, 96);
  std::vector<std::uint8_t> first(input.size()), repeat(input.size());
  auto src = augmatch::make_hwc_view(input.data(), width, height, channels);
  auto dst_a = augmatch::make_hwc_view(first.data(), width, height, channels);
  auto dst_b = augmatch::make_hwc_view(repeat.data(), width, height, channels);
  augmatch::NoiseViewConfig config;
  config.seed = 20240517;
  config.stddev = 0.01f;
  const auto a = augmatch::sensor::fused_pipeline(src.as_const(), dst_a, {config});
  const auto b = augmatch::sensor::fused_pipeline(src.as_const(), dst_b, {config});
  if (!a.ok() || !b.ok() || first != repeat) {
    std::cerr << "camera-noise reproducibility check failed\n";
    return 1;
  }
  std::cout << "reproducible RGB sensor-noise example; first sample="
            << static_cast<int>(first.front()) << '\n';
  return 0;
}
