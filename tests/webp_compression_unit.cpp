#include "augmatch/compression/webp.hpp"
#include <algorithm>
#include <cstdint>
#include <exception>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  constexpr int width = 32;
  constexpr int height = 24;
  std::vector<std::uint8_t> input(static_cast<std::size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    const std::size_t p = (static_cast<std::size_t>(y) * width + x) * 3;
    input[p] = static_cast<std::uint8_t>((x * 9 + y * 3) & 255);
    input[p + 1] = static_cast<std::uint8_t>((y * 11 + x) & 255);
    input[p + 2] = static_cast<std::uint8_t>((x * 5 + y * 7) & 255);
  }
  const augmatch::WebPCompressionConfig lossless{width, height, 3, 75, true};
  const auto lossless_encoded = augmatch::webp_encode_u8(input.data(), lossless);
  if (lossless_encoded.size() < 12 || lossless_encoded[0] != 'R' ||
      lossless_encoded[1] != 'I' || lossless_encoded[2] != 'F' ||
      lossless_encoded[8] != 'W' || lossless_encoded[9] != 'E' ||
      lossless_encoded[10] != 'B' || lossless_encoded[11] != 'P') return 1;
  std::vector<std::uint8_t> decoded(input.size());
  augmatch::webp_decode_u8(lossless_encoded.data(), lossless_encoded.size(), decoded.data(),
                           {width, height, 3});
  if (decoded != input) return 2;

  std::vector<std::uint8_t> explicit_lossless(input.size());
  augmatch::webp_lossless_compression_u8(input.data(), explicit_lossless.data(),
                                         {width, height, 3, 75, false});
  if (explicit_lossless != input) return 4;
  std::vector<std::uint8_t> explicit_lossy(input.size());
  augmatch::webp_lossy_compression_u8(input.data(), explicit_lossy.data(),
                                      {width, height, 3, 70, true});
  const auto lossy_encoded = augmatch::webp_encode_u8(
      input.data(), {width, height, 3, 70, false});
  std::fill(decoded.begin(), decoded.end(), 0);
  augmatch::webp_decode_u8(lossy_encoded.data(), lossy_encoded.size(), decoded.data(),
                           {width, height, 3});
  int maximum_error = 0;
  for (std::size_t i = 0; i < input.size(); ++i)
    maximum_error = std::max(maximum_error,
                             std::abs(static_cast<int>(input[i]) - static_cast<int>(decoded[i])));
  if (maximum_error > 200) return 3;

  bool rejected = false;
  try {
    (void)augmatch::webp_encode_u8(input.data(), {width, height, 3, 101, false});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 5;
  std::cout << "webp compression unit passed (lossy max error " << maximum_error << ")\n";
  return 0;
}
