#include <cstdint>
#include <vector>
#include "augmatch/augmatch.hpp"

int main() {
#if AUGMATCH_HAS_CUDA
  // This consumer smoke test verifies CUDA package linkage; device execution
  // is covered by the dedicated validation tests and may have no GPU in CI.
  return 77;
#else
  std::vector<std::uint8_t> input(4, 128), output(4);
  const auto source = augmatch::make_hwc_view(input.data(), 2, 2, 1);
  const auto destination = augmatch::make_hwc_view(output.data(), 2, 2, 1);
  augmatch::NoiseViewConfig config;
  config.stddev = 0.0f;
  const auto status = augmatch::sensor::fused_pipeline(
      source.as_const(), destination, {config});
  return status.ok() ? 0 : 1;
#endif
}
