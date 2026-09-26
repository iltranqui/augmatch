#include <iostream>
#include <vector>
#include "augmatch/augmatch.hpp"

int main() {
  std::vector<float> input(4, 0.5f), output(4);
  auto in = augmatch::make_hwc_view(input.data(), 2, 2, 1);
  auto out = augmatch::make_hwc_view(output.data(), 2, 2, 1);
  augmatch::NoiseViewConfig config{.seed = 23, .stddev = 0.0f};
  auto future = augmatch::additive_noise_async(in.as_const(), out, config);
  const auto result = future.get();
  std::cout << (result.ok() ? "ok" : result.message) << " " << output.front() << '\n';
  return result.ok() ? 0 : 1;
}
