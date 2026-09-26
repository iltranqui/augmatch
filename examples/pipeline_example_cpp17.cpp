// pipeline_example_cpp17 — load augmentation.cfg from disk and run it (C++17).
//
// What it shows: the config-driven Pipeline as you would embed it in a training
// loader. Pass a config path as argv[1] (default: ./augmentation.cfg, which the
// build generates next to the binary). The image is modified in place.
// API used: Pipeline::load_from_file, make_hwc_u8_view, ExecutionContext, Pipeline::run.
// Output: pipeline_cpp17_output.ppm; exit codes 1-4 identify the failing step.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "augmatch/augmatch.hpp"

namespace {
// Write interleaved RGB bytes as a binary PPM (P6) image.
bool write_ppm(const std::string& path, const std::vector<std::uint8_t>& image,
               int width, int height) {
  std::ofstream output(path, std::ios::binary);
  if (!output) return false;
  output << "P6\n" << width << ' ' << height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.data()),
               static_cast<std::streamsize>(image.size()));
  return output.good();
}
}  // namespace

int main(int argc, char** argv) {
  const std::string config_path = argc > 1 ? argv[1] : "augmentation.cfg";
  // Parse + validate the whole config once; errors name the bad stage or key.
  auto loaded = augmatch::Pipeline::load_from_file(config_path);
  if (!loaded.ok()) { std::cerr << loaded.status().message << '\n'; return 1; }

  constexpr int width = 3, height = 2, channels = 3;  // tiny 3x2 RGB image
  std::vector<std::uint8_t> pixels{
      20, 40, 80, 80, 40, 20, 160, 100, 40,
      10, 30, 60, 60, 30, 10, 120, 80, 30};
  const auto before = pixels;  // keep a copy to prove the pipeline changed something

  // Future darknet seam: src-lib/image_opencv.cpp:image_data_augmentation()
  // (currently near line 297) can wrap cv::Mat::data exactly this way.
  auto view = augmatch::make_hwc_u8_view(pixels.data(), width, height, channels);
  // The seed drives every random decision (e.g. `probability`), so runs are reproducible.
  augmatch::ExecutionContext execution;
  execution.seed = 17;
  execution.deterministic = true;
  const augmatch::Status status = loaded.value().run(view, execution);  // in place, host memory
  if (!status.ok()) { std::cerr << status.message << '\n'; return 2; }
  if (pixels == before) { std::cerr << "pipeline did not modify the image\n"; return 3; }

  const std::string output_path = "pipeline_cpp17_output.ppm";
  if (!write_ppm(output_path, pixels, width, height)) return 4;
  std::cout << "ok " << output_path << '\n';
  return 0;
}
