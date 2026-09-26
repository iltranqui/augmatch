// support_contract — typed views, one seed, and explicit geometry metadata.
//
// What it shows: the shared "support contract" in one place. A float32 view,
// a seeded view-based noise call that returns Status, and a keypoint transformed
// by an explicit Geometry (here: a horizontal flip).
// API used: make_hwc_view, NoiseViewConfig, additive_noise_view, Geometry, transform_point.
// Output: prints the seed and the flipped keypoint x coordinate.
#include "augmatch/augmatch.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  constexpr int width = 2;
  constexpr int height = 2;
  constexpr int channels = 1;
  std::vector<float> image(width * height * channels, 0.5f);  // float32 image in [0,1]
  std::vector<float> output(image.size());
  auto input = augmatch::make_hwc_view(image.data(), width, height, channels);
  auto destination = augmatch::make_hwc_view(output.data(), width, height, channels);

  augmatch::NoiseViewConfig config;
  config.seed = 17;
  config.stddev = 0.0f;                                        // zero noise: output equals input
  const augmatch::Status status =
      augmatch::additive_noise_view(input.as_const(), destination, config);
  if (!status.ok()) {
    std::cerr << status.message << '\n';
    return 1;
  }

  // Geometry fields: input w/h/c, crop x/y, output w/h, flip_horizontal, flip_vertical.
  const augmatch::Geometry geometry{width, height, channels, 0, 0, width, height, true, false};
  const augmatch::PointXY point{0.0f, 0.0f};
  const augmatch::PointXY flipped = augmatch::transform_point(point, geometry);  // x -> width - x
  std::cout << "seed=" << config.seed << " flipped_x=" << flipped.x << '\n';
  return 0;
}
