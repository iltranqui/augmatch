// native_api_cpp17 — the native sensor API compiled as C++17.
//
// What it shows: run the fused sensor pipeline (additive noise) on a tiny 2x2
// grayscale image through typed views. native_api_cpp23.cpp is the C++23 twin;
// both exist to prove the public headers compile under either standard.
// API used: make_hwc_view, NoiseViewConfig, sensor::fused_pipeline.
// Output: prints "ok <first output pixel>".
#include <iostream>
#include <vector>
#include "augmatch/augmatch.hpp"

int main() {
  std::vector<std::uint8_t> input(4, 128), output(4);        // 2x2x1 image, mid-gray
  auto in = augmatch::make_hwc_view(input.data(), 2, 2, 1);  // uint8 view (type deduced from the pointer)
  auto out = augmatch::make_hwc_view(output.data(), 2, 2, 1);
  augmatch::NoiseViewConfig config;
  config.seed = 7; config.stddev = 0.01f;                    // tiny noise, fixed seed -> reproducible
  // fused_pipeline takes a list of noise stages; here just one.
  auto result = augmatch::sensor::fused_pipeline(in.as_const(), out, {config});
  std::cout << (result.ok() ? "ok" : result.message) << " " << static_cast<int>(output[0]) << '\n';
  return result.ok() ? 0 : 1;                                // non-zero exit makes the smoke test fail
}
