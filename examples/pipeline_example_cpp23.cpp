#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "augmatch/augmatch.hpp"

namespace {
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
  auto loaded = augmatch::Pipeline::load_from_file(config_path);
  if (!loaded.ok()) { std::cerr << loaded.status().message << '\n'; return 1; }

  constexpr int width = 3, height = 2, channels = 3;
  std::vector<std::uint8_t> pixels{
      20, 40, 80, 80, 40, 20, 160, 100, 40,
      10, 30, 60, 60, 30, 10, 120, 80, 30};
  const auto before = pixels;

  // Future darknet seam: src-lib/image_opencv.cpp:image_data_augmentation()
  // (currently near line 297) can wrap cv::Mat::data exactly this way.
  auto view = augmatch::make_hwc_u8_view(pixels.data(), width, height, channels);
  augmatch::ExecutionContext execution{.seed = 23, .deterministic = true};
  const augmatch::Status status = loaded.value().run(view, execution);
  if (!status.ok()) { std::cerr << status.message << '\n'; return 2; }
  if (pixels == before) { std::cerr << "pipeline did not modify the image\n"; return 3; }

  const std::string output_path = "pipeline_cpp23_output.ppm";
  if (!write_ppm(output_path, pixels, width, height)) return 4;
  std::cout << "ok " << output_path << '\n';
  return 0;
}
