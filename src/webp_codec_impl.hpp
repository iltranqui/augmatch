#pragma once

#include "augmatch/compression/webp.hpp"
#include <webp/decode.h>
#include <webp/encode.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace augmatch::webp_detail {

inline std::size_t image_bytes(const WebPCompressionConfig& config) {
  if (config.width <= 0 || config.height <= 0 ||
      (config.channels != 1 && config.channels != 3) ||
      config.quality < 0 || config.quality > 100 ||
      config.width > std::numeric_limits<int>::max() / 3)
    throw std::invalid_argument("invalid WebP dimensions, channels, or quality");
  const auto pixels = static_cast<std::uint64_t>(config.width) *
                      static_cast<std::uint64_t>(config.height) *
                      static_cast<std::uint64_t>(config.channels);
  if (pixels > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("WebP image is too large");
  return static_cast<std::size_t>(pixels);
}

inline void validate_decode(const std::uint8_t* encoded, std::size_t encoded_size,
                            std::uint8_t* output, const WebPDecodeConfig& config) {
  if (!encoded || encoded_size == 0 || !output || config.width <= 0 || config.height <= 0 ||
      (config.channels != 1 && config.channels != 3) ||
      config.width > std::numeric_limits<int>::max() / 3)
    throw std::invalid_argument("invalid WebP encoded data or decode dimensions");
  const auto bytes = static_cast<std::uint64_t>(config.width) *
                     static_cast<std::uint64_t>(config.height) *
                     static_cast<std::uint64_t>(config.channels);
  if (bytes > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("WebP image is too large");
}

inline std::vector<std::uint8_t> encode_host(const std::uint8_t* input,
                                             const WebPCompressionConfig& config) {
  const std::size_t bytes = image_bytes(config);
  (void)bytes;
  WebPConfig encoder_config{};
  if (!WebPConfigInit(&encoder_config))
    throw std::runtime_error("WebPConfigInit failed");
  encoder_config.lossless = config.lossless ? 1 : 0;
  encoder_config.quality = static_cast<float>(config.quality);
  encoder_config.method = 4;
  if (!WebPValidateConfig(&encoder_config))
    throw std::invalid_argument("invalid WebP encoder configuration");

  WebPPicture picture{};
  if (!WebPPictureInit(&picture))
    throw std::runtime_error("WebPPictureInit failed");
  picture.width = config.width;
  picture.height = config.height;
  // Keep lossless input in ARGB mode. The default YUV import performs a
  // color conversion that can make an otherwise lossless RGB round trip lossy.
  picture.use_argb = config.lossless ? 1 : 0;
  // libwebp's picture API is RGB/RGBA based. Replicate grayscale samples
  // into RGB so one-channel input remains supported without a non-native codec.
  std::vector<std::uint8_t> expanded;
  const std::uint8_t* rgb_input = input;
  int stride = config.width * 3;
  if (config.channels == 1) {
    expanded.resize(static_cast<std::size_t>(config.width) * config.height * 3);
    for (std::size_t i = 0, p = 0; i < expanded.size(); i += 3, ++p)
      expanded[i] = expanded[i + 1] = expanded[i + 2] = input[p];
    rgb_input = expanded.data();
  } else {
    stride = config.width * config.channels;
  }
  const int imported = WebPPictureImportRGB(&picture, rgb_input, stride);
  if (!imported) {
    const std::string error = picture.error_code == VP8_ENC_OK
        ? "input import failed" : "picture import error " + std::to_string(picture.error_code);
    WebPPictureFree(&picture);
    throw std::runtime_error("WebP encode failed: " + error);
  }

  WebPMemoryWriter writer;
  WebPMemoryWriterInit(&writer);
  picture.writer = WebPMemoryWrite;
  picture.custom_ptr = &writer;
  if (!WebPEncode(&encoder_config, &picture)) {
    const std::string error = "picture error " + std::to_string(picture.error_code);
    WebPMemoryWriterClear(&writer);
    WebPPictureFree(&picture);
    throw std::runtime_error("WebP encode failed: " + error);
  }
  std::vector<std::uint8_t> result(writer.mem, writer.mem + writer.size);
  WebPMemoryWriterClear(&writer);
  WebPPictureFree(&picture);
  return result;
}

inline void decode_host(const std::uint8_t* encoded, std::size_t encoded_size,
                        std::uint8_t* output, const WebPDecodeConfig& config) {
  validate_decode(encoded, encoded_size, output, config);
  int width = 0;
  int height = 0;
  if (!WebPGetInfo(encoded, encoded_size, &width, &height) ||
      width != config.width || height != config.height)
    throw std::invalid_argument("WebP dimensions do not match decode configuration");
  const std::size_t pixels = static_cast<std::size_t>(config.width) *
                             static_cast<std::size_t>(config.height);
  if (config.channels == 3) {
    const std::size_t output_size = pixels * 3;
    if (!WebPDecodeRGBInto(encoded, encoded_size, output, output_size, config.width * 3))
      throw std::runtime_error("WebP decode failed");
    return;
  }
  std::vector<std::uint8_t> decoded_rgb(pixels * 3);
  if (!WebPDecodeRGBInto(encoded, encoded_size, decoded_rgb.data(), decoded_rgb.size(), config.width * 3))
    throw std::runtime_error("WebP decode failed");
  for (std::size_t p = 0; p < pixels; ++p)
    output[p] = decoded_rgb[p * 3];
}

}  // namespace augmatch::webp_detail
