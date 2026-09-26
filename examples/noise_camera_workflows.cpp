// noise_camera_workflows — reproducible RGB sensor noise.
//
// What it shows: running the same seeded sensor-noise stage twice gives
// byte-identical output. The seed is the only source of randomness.
// API used: make_hwc_view, NoiseViewConfig, sensor::fused_pipeline.
// Output: prints the first noisy sample; exits 1 if the two runs differ.
#include <cstdint>
#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"

int main() {
  constexpr int width = 8, height = 4, channels = 3;
  std::vector<std::uint8_t> input(width * height * channels, 96);  // flat gray RGB image
  std::vector<std::uint8_t> first(input.size()), repeat(input.size());
  auto src = augmatch::make_hwc_view(input.data(), width, height, channels);
  auto dst_a = augmatch::make_hwc_view(first.data(), width, height, channels);
  auto dst_b = augmatch::make_hwc_view(repeat.data(), width, height, channels);
  augmatch::NoiseViewConfig config;
  config.seed = 20240517;                                          // same seed for both runs
  config.stddev = 0.01f;                                           // noise std-dev, fraction of full scale
  const auto a = augmatch::sensor::fused_pipeline(src.as_const(), dst_a, {config});
  const auto b = augmatch::sensor::fused_pipeline(src.as_const(), dst_b, {config});
  if (!a.ok() || !b.ok() || first != repeat) {                     // both must succeed and match exactly
    std::cerr << "camera-noise reproducibility check failed\n";
    return 1;
  }
  std::cout << "reproducible RGB sensor-noise example; first sample="
            << static_cast<int>(first.front()) << '\n';
  return 0;
}
