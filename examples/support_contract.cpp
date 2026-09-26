// Support-contract example: typed views, one seed, and explicit metadata.
#include "augmatch/augmatch.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  constexpr int width = 2;
  constexpr int height = 2;
  constexpr int channels = 1;
  std::vector<float> image(width * height * channels, 0.5f);
  std::vector<float> output(image.size());
  auto input = augmatch::make_hwc_view(image.data(), width, height, channels);
  auto destination = augmatch::make_hwc_view(output.data(), width, height, channels);

  augmatch::NoiseViewConfig config;
  config.seed = 17;
  config.stddev = 0.0f;
  const augmatch::Status status =
      augmatch::additive_noise_view(input.as_const(), destination, config);
  if (!status.ok()) {
    std::cerr << status.message << '\n';
    return 1;
  }

  const augmatch::Geometry geometry{width, height, channels, 0, 0, width, height, true, false};
  const augmatch::PointXY point{0.0f, 0.0f};
  const augmatch::PointXY flipped = augmatch::transform_point(point, geometry);
  std::cout << "seed=" << config.seed << " flipped_x=" << flipped.x << '\n';
  return 0;
}
