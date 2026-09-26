#include <iostream>
#include <vector>
#include "augmatch/augmatch.hpp"

int main() {
  std::vector<std::uint8_t> input(4, 128), output(4);
  auto in = augmatch::make_hwc_view(input.data(), 2, 2, 1);
  auto out = augmatch::make_hwc_view(output.data(), 2, 2, 1);
  augmatch::NoiseViewConfig config;
  config.seed = 7; config.stddev = 0.01f;
  auto result = augmatch::sensor::fused_pipeline(in.as_const(), out, {config});
  std::cout << (result.ok() ? "ok" : result.message) << " " << static_cast<int>(output[0]) << '\n';
  return result.ok() ? 0 : 1;
}
