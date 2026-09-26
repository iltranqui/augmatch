// 10_pipeline_cfg — run a whole augmentation sequence described in text.
//
// What it shows:
//   * A Pipeline is loaded once from INI-style text (or a file) and run many times.
//   * Changing the augmentation recipe needs no recompilation.
//   * ExecutionContext carries the seed: same seed -> same random choices (probability, etc.).
// API used: Pipeline::load_from_string, Pipeline::run, ExecutionContext.
// Pipeline runs in place on host memory in both CPU and CUDA builds.
// Output: 10_pipeline_cfg.ppm.
// See docs/guides/PIPELINE.md for every stage name and parameter.

#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"
#include "example_utils.hpp"

// The same text could live in an `augmentation.cfg` file (Pipeline::load_from_file).
static const char* kRecipe = R"(
# Stages run top to bottom. `probability` is optional (default 1.0).
[Saturation]
multiplier = 1.4

[HueShift]
amount = 0.05

[GaussianBlur]
kernel_size = 5
sigma = 1.0
probability = 0.5

[HorizontalFlip]
probability = 0.5
)";

int main() {
  // 1. Parse and validate once. Unknown stages or parameters are reported here, not at run time.
  auto loaded = augmatch::Pipeline::load_from_string(kRecipe);
  if (!loaded) { std::cerr << "config error: " << loaded.status().message << '\n'; return 1; }
  const augmatch::Pipeline& pipeline = loaded.value();
  std::cout << "loaded " << pipeline.stages().size() << " stages\n";

  // 2. Prepare an image. The pipeline modifies it in place.
  const int width = 64, height = 48, channels = 3;
  std::vector<std::uint8_t> pixels = example::make_test_image(width, height);
  const std::vector<std::uint8_t> original = pixels;
  const auto view = augmatch::make_hwc_u8_view(pixels.data(), width, height, channels);

  // 3. Run with a fixed seed so the random `probability` decisions are reproducible.
  augmatch::ExecutionContext execution;
  execution.seed = 17;
  execution.deterministic = true;
  const augmatch::Status status = pipeline.run(view, execution);
  if (!status) { std::cerr << "run error: " << status.message << '\n'; return 1; }
  if (pixels == original) { std::cerr << "pipeline did not change the image\n"; return 1; }

  example::write_ppm("10_pipeline_cfg.ppm", pixels, width, height);
  std::cout << "wrote 10_pipeline_cfg.ppm\n";
  return 0;
}
