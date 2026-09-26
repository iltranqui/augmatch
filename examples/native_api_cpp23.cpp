// native_api_cpp23 — the native sensor API compiled as C++23.
//
// What it shows: C++20 designated initializers on augmatch config structs, and
// the host-only asynchronous noise call that returns a std::future<Status>.
// API used: make_hwc_view, NoiseViewConfig, additive_noise_async.
// Output: prints "ok 0.5" (stddev is 0, so the image is unchanged).
#include <iostream>
#include <vector>
#include "augmatch/augmatch.hpp"

int main() {
  std::vector<float> input(4, 0.5f), output(4);              // 2x2x1 float image in [0,1]
  auto in = augmatch::make_hwc_view(input.data(), 2, 2, 1);  // float32 view
  auto out = augmatch::make_hwc_view(output.data(), 2, 2, 1);
  augmatch::NoiseViewConfig config{.seed = 23, .stddev = 0.0f};  // designated initializers (C++20+)
  // Runs on a host thread; keep input/output alive until the future completes.
  auto future = augmatch::additive_noise_async(in.as_const(), out, config);
  const auto result = future.get();                          // wait and fetch the Status
  std::cout << (result.ok() ? "ok" : result.message) << " " << output.front() << '\n';
  return result.ok() ? 0 : 1;
}
