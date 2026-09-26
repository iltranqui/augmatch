#include "augmatch/webp.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
  std::vector<std::uint8_t> input(4 * 4 * 3), output(input.size());
  try {
    augmatch::webp_lossy_compression_u8(input.data(), output.data(), {4, 4, 3, 70, false});
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    if (message.find("WebP codec unavailable") != std::string::npos) {
      std::cout << "WebP unavailable fallback is explicit\n";
      return 0;
    }
  }
  return 1;
}
