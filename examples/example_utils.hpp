#pragma once
// Small helpers shared by the numbered examples, so each example can focus on
// the augmatch call it demonstrates. Not part of the library API.

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace example {

// A synthetic RGB test image: red grows left-to-right, green grows
// top-to-bottom, blue is a checkerboard. Easy to eyeball after a transform.
inline std::vector<std::uint8_t> make_test_image(int width, int height) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      std::uint8_t* p = &pixels[(static_cast<std::size_t>(y) * width + x) * 3];  // HWC: 3 bytes per pixel
      p[0] = static_cast<std::uint8_t>(255 * x / (width - 1));                    // red ramp
      p[1] = static_cast<std::uint8_t>(255 * y / (height - 1));                   // green ramp
      p[2] = ((x / 8 + y / 8) % 2) ? 200 : 40;                                    // blue checkerboard
    }
  }
  return pixels;
}

// Write an interleaved RGB uint8 buffer as a binary PPM (P6), viewable in most image viewers.
inline bool write_ppm(const std::string& path, const std::vector<std::uint8_t>& rgb, int width, int height) {
  std::ofstream file(path, std::ios::binary);
  if (!file) return false;
  file << "P6\n" << width << ' ' << height << "\n255\n";
  file.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  return file.good();
}

}  // namespace example
