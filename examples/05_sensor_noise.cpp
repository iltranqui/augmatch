// 05_sensor_noise — camera-like ISO noise, and why the seed matters.
//
// What it shows:
//   * Random-looking transforms are deterministic: same seed -> same bytes, on CPU and CUDA.
//   * A different seed gives a different noise pattern.
//   * No global random state: the seed lives in the config you pass.
// API used: ISONoiseConfig, easy::iso_noise.
// Output: 05_sensor_noise.ppm — the test image as if shot at ISO 3200.

#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"
#include "example_utils.hpp"

int main() {
  const int width = 64, height = 48, channels = 3;
  std::vector<std::uint8_t> input = example::make_test_image(width, height);
  std::vector<std::uint8_t> run_a(input.size()), run_b(input.size()), run_c(input.size());

  const auto in = augmatch::make_hwc_u8_view(input.data(), width, height, channels);

  augmatch::ISONoiseConfig noise;
  noise.iso = 3200.0f;            // higher ISO -> more noise (scaled relative to base_iso)
  noise.base_iso = 100.0f;        // the sensor's native ISO
  noise.gaussian_stddev = 0.02f;  // luma noise, as a fraction of full scale
  noise.chroma_stddev = 0.01f;    // per-channel colour noise
  noise.seed = 42;                // the only source of randomness

  // Run twice with the same seed ...
  augmatch::easy::iso_noise(in, augmatch::make_hwc_u8_view(run_a.data(), width, height, channels), noise);
  augmatch::easy::iso_noise(in, augmatch::make_hwc_u8_view(run_b.data(), width, height, channels), noise);
  // ... and once with another seed.
  noise.seed = 43;
  const augmatch::Status status =
      augmatch::easy::iso_noise(in, augmatch::make_hwc_u8_view(run_c.data(), width, height, channels), noise);
  if (!status) { std::cerr << "iso_noise: " << status.message << '\n'; return 1; }

  std::cout << "same seed identical:      " << (run_a == run_b ? "yes" : "no") << '\n';  // expected: yes
  std::cout << "different seed identical: " << (run_a == run_c ? "yes" : "no") << '\n';  // expected: no
  if (run_a != run_b || run_a == run_c) return 1;

  example::write_ppm("05_sensor_noise.ppm", run_a, width, height);
  std::cout << "wrote 05_sensor_noise.ppm\n";
  return 0;
}
