#pragma once

#include "augmatch/jpeg.hpp"
#include <cstdio>
#include <jpeglib.h>
#include <csetjmp>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace augmatch::jpeg_detail {

struct ErrorManager {
  jpeg_error_mgr public_manager{};
  jmp_buf jump{};
  char message[JMSG_LENGTH_MAX]{};
};

inline void output_message(j_common_ptr) {}

inline void error_exit(j_common_ptr common) {
  auto* manager = reinterpret_cast<ErrorManager*>(common->err);
  (*common->err->format_message)(common, manager->message);
  longjmp(manager->jump, 1);
}

inline void validate_encode(const std::uint8_t* input, const JpegCompressionConfig& config) {
  if (!input || config.width <= 0 || config.height <= 0 ||
      (config.channels != 1 && config.channels != 3) || config.quality < 1 || config.quality > 100)
    throw std::invalid_argument("invalid JPEG dimensions, channels, or quality");
  if (config.subsampling != JpegSubsampling::Y444 &&
      config.subsampling != JpegSubsampling::Y422 &&
      config.subsampling != JpegSubsampling::Y420)
    throw std::invalid_argument("unsupported JPEG chroma subsampling");
  if (!std::isfinite(config.luma_quant_scale) || !std::isfinite(config.chroma_quant_scale) ||
      config.luma_quant_scale <= 0.0f || config.chroma_quant_scale <= 0.0f ||
      config.luma_quant_scale > 100.0f || config.chroma_quant_scale > 100.0f)
    throw std::invalid_argument("JPEG quantization scales must be finite and in (0,100]");
  const auto pixels = static_cast<std::uint64_t>(config.width) * config.height * config.channels;
  if (pixels > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("JPEG image is too large");
}

inline std::vector<std::uint8_t> encode_host_options(const std::uint8_t* input,
                                                      const JpegCompressionConfig& config,
                                                      bool progressive,
                                                      int restart_interval) {
  validate_encode(input, config);
  if (restart_interval < 0)
    throw std::invalid_argument("JPEG restart interval must be nonnegative");
  jpeg_compress_struct compressor{};
  ErrorManager errors{};
  unsigned char* encoded = nullptr;
  unsigned long encoded_size = 0;
  compressor.err = jpeg_std_error(&errors.public_manager);
  errors.public_manager.error_exit = error_exit;
  errors.public_manager.output_message = output_message;
  if (setjmp(errors.jump)) {
    jpeg_destroy_compress(&compressor);
    if (encoded) std::free(encoded);
    throw std::runtime_error(std::string("JPEG encode failed: ") + errors.message);
  }
  jpeg_create_compress(&compressor);
  jpeg_mem_dest(&compressor, &encoded, &encoded_size);
  compressor.image_width = static_cast<JDIMENSION>(config.width);
  compressor.image_height = static_cast<JDIMENSION>(config.height);
  compressor.input_components = config.channels;
  compressor.in_color_space = config.channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
  jpeg_set_defaults(&compressor);
  jpeg_set_quality(&compressor, config.quality, TRUE);
  // libjpeg has already generated quality-scaled tables. Adjust those tables
  // directly so this operation remains compatible with libjpeg-turbo's exact
  // quantization-table rounding rules.
  for (int table = 0; table < 2; ++table) {
    auto* quant = compressor.quant_tbl_ptrs[table];
    if (!quant) continue;
    const float scale = table == 0 ? config.luma_quant_scale : config.chroma_quant_scale;
    for (int i = 0; i < DCTSIZE * DCTSIZE; ++i) {
      const long value = std::lround(static_cast<float>(quant->quantval[i]) * scale);
      quant->quantval[i] = static_cast<unsigned short>(std::clamp(value, 1L, 32767L));
    }
  }
  if (config.channels == 3) {
    if (config.subsampling == JpegSubsampling::Y444) {
      compressor.comp_info[0].h_samp_factor = 1;
      compressor.comp_info[0].v_samp_factor = 1;
      compressor.comp_info[1].h_samp_factor = 1;
      compressor.comp_info[1].v_samp_factor = 1;
      compressor.comp_info[2].h_samp_factor = 1;
      compressor.comp_info[2].v_samp_factor = 1;
    } else if (config.subsampling == JpegSubsampling::Y422) {
      compressor.comp_info[0].h_samp_factor = 2;
      compressor.comp_info[0].v_samp_factor = 1;
      compressor.comp_info[1].h_samp_factor = 1;
      compressor.comp_info[1].v_samp_factor = 1;
      compressor.comp_info[2].h_samp_factor = 1;
      compressor.comp_info[2].v_samp_factor = 1;
    } else {
      compressor.comp_info[0].h_samp_factor = 2;
      compressor.comp_info[0].v_samp_factor = 2;
      compressor.comp_info[1].h_samp_factor = 1;
      compressor.comp_info[1].v_samp_factor = 1;
      compressor.comp_info[2].h_samp_factor = 1;
      compressor.comp_info[2].v_samp_factor = 1;
    }
  }
  if (progressive) jpeg_simple_progression(&compressor);
  compressor.restart_interval = static_cast<unsigned int>(restart_interval);
  jpeg_start_compress(&compressor, TRUE);
  const std::size_t row_bytes = static_cast<std::size_t>(config.width) * config.channels;
  while (compressor.next_scanline < compressor.image_height) {
    JSAMPROW row = const_cast<JSAMPROW>(input + static_cast<std::size_t>(compressor.next_scanline) * row_bytes);
    jpeg_write_scanlines(&compressor, &row, 1);
  }
  jpeg_finish_compress(&compressor);
  std::vector<std::uint8_t> result(encoded, encoded + encoded_size);
  jpeg_destroy_compress(&compressor);
  std::free(encoded);
  return result;
}

inline std::vector<std::uint8_t> encode_host(const std::uint8_t* input,
                                             const JpegCompressionConfig& config) {
  return encode_host_options(input, config, false, 0);
}

inline void validate_decode(const std::uint8_t* encoded, std::size_t encoded_size,
                            std::uint8_t* output, const JpegDecodeConfig& config) {
  if (!encoded || encoded_size == 0 || !output || config.width <= 0 || config.height <= 0 ||
      (config.channels != 1 && config.channels != 3))
    throw std::invalid_argument("invalid JPEG encoded data or decode dimensions");
}

inline void decode_host(const std::uint8_t* encoded, std::size_t encoded_size,
                        std::uint8_t* output, const JpegDecodeConfig& config) {
  validate_decode(encoded, encoded_size, output, config);
  jpeg_decompress_struct decompressor{};
  ErrorManager errors{};
  decompressor.err = jpeg_std_error(&errors.public_manager);
  errors.public_manager.error_exit = error_exit;
  errors.public_manager.output_message = output_message;
  if (setjmp(errors.jump)) {
    jpeg_destroy_decompress(&decompressor);
    throw std::runtime_error(std::string("JPEG decode failed: ") + errors.message);
  }
  jpeg_create_decompress(&decompressor);
  jpeg_mem_src(&decompressor, const_cast<unsigned char*>(encoded), encoded_size);
  jpeg_read_header(&decompressor, TRUE);
  if (decompressor.image_width != static_cast<JDIMENSION>(config.width) ||
      decompressor.image_height != static_cast<JDIMENSION>(config.height))
    throw std::invalid_argument("JPEG dimensions do not match decode configuration");
  decompressor.out_color_space = config.channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
  jpeg_start_decompress(&decompressor);
  if (decompressor.output_components != config.channels) {
    jpeg_abort_decompress(&decompressor);
    jpeg_destroy_decompress(&decompressor);
    throw std::runtime_error("JPEG decoder returned unexpected channel count");
  }
  const std::size_t row_bytes = static_cast<std::size_t>(config.width) * config.channels;
  while (decompressor.output_scanline < decompressor.output_height) {
    JSAMPROW row = output + static_cast<std::size_t>(decompressor.output_scanline) * row_bytes;
    jpeg_read_scanlines(&decompressor, &row, 1);
  }
  jpeg_finish_decompress(&decompressor);
  jpeg_destroy_decompress(&decompressor);
}

inline void encode_decode_host(const std::uint8_t* input, std::uint8_t* output,
                               const JpegCompressionConfig& config) {
  const auto encoded = encode_host(input, config);
  decode_host(encoded.data(), encoded.size(), output,
              {config.width, config.height, config.channels});
}

inline void validate_artifact(const std::uint8_t* input, std::uint8_t* output,
                              const JpegArtifactConfig& config) {
  validate_encode(input, {config.width, config.height, config.channels, config.quality,
                          config.subsampling});
  if (!std::isfinite(config.strength) || config.strength < 0.0f || config.strength > 10.0f)
    throw std::invalid_argument("JPEG artifact strength must be finite and in [0,10]");
  if (!output) throw std::invalid_argument("null JPEG artifact output");
}

inline std::uint8_t artifact_value(float value) {
  return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

inline void jpeg_ringing_host(const std::uint8_t* input, std::uint8_t* output,
                              const JpegArtifactConfig& config) {
  validate_artifact(input, output, config);
  JpegCompressionConfig codec{config.width, config.height, config.channels,
                              config.quality, config.subsampling};
  std::vector<std::uint8_t> decoded(static_cast<std::size_t>(config.width) * config.height * config.channels);
  encode_decode_host(input, decoded.data(), codec);
  // Accentuation of the DCT ringing component: a four-neighbour high-pass is
  // strongest near edges and is clipped like an 8-bit decoded JPEG sample.
  for (int y = 0; y < config.height; ++y) for (int x = 0; x < config.width; ++x) {
    const std::size_t p = (static_cast<std::size_t>(y) * config.width + x) * config.channels;
    for (int c = 0; c < config.channels; ++c) {
      float sum = 0.0f; int count = 0;
      for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
        if ((dx || dy) && x + dx >= 0 && x + dx < config.width && y + dy >= 0 && y + dy < config.height) {
          sum += decoded[(static_cast<std::size_t>(y + dy) * config.width + x + dx) * config.channels + c]; ++count;
        }
      output[p + c] = artifact_value(decoded[p + c] + config.strength *
                                      (decoded[p + c] - sum / std::max(1, count)));
    }
  }
}

inline void jpeg_blocking_host(const std::uint8_t* input, std::uint8_t* output,
                               const JpegArtifactConfig& config) {
  validate_artifact(input, output, config);
  JpegCompressionConfig codec{config.width, config.height, config.channels,
                              config.quality, config.subsampling};
  std::vector<std::uint8_t> decoded(static_cast<std::size_t>(config.width) * config.height * config.channels);
  encode_decode_host(input, decoded.data(), codec);
  std::copy(decoded.begin(), decoded.end(), output);
  // JPEG's 8x8 DCT grid is explicit in the encoded format. Exaggerate only
  // samples on a grid boundary, preserving flat interior blocks.
  for (int y = 0; y < config.height; ++y) for (int x = 0; x < config.width; ++x) {
    if (x == 0 && y == 0) continue;
    if (x % 8 != 0 && y % 8 != 0) continue;
    const std::size_t p = (static_cast<std::size_t>(y) * config.width + x) * config.channels;
    const int nx = x > 0 ? x - 1 : x;
    const int ny = y > 0 ? y - 1 : y;
    const std::size_t n = (static_cast<std::size_t>(ny) * config.width + nx) * config.channels;
    for (int c = 0; c < config.channels; ++c)
      output[p + c] = artifact_value(decoded[p + c] + config.strength * (decoded[p + c] - decoded[n + c]));
  }
}

inline void jpeg_mosquito_noise_host(const std::uint8_t* input, std::uint8_t* output,
                                     const JpegArtifactConfig& config) {
  validate_artifact(input, output, config);
  JpegCompressionConfig codec{config.width, config.height, config.channels,
                              config.quality, config.subsampling};
  std::vector<std::uint8_t> decoded(static_cast<std::size_t>(config.width) * config.height * config.channels);
  encode_decode_host(input, decoded.data(), codec);
  // Mosquito noise is represented as a small DCT high-frequency halo around
  // local contrast, using a deterministic 3x3 neighbourhood.
  for (int y = 0; y < config.height; ++y) for (int x = 0; x < config.width; ++x) {
    const std::size_t p = (static_cast<std::size_t>(y) * config.width + x) * config.channels;
    for (int c = 0; c < config.channels; ++c) {
      float sum = 0.0f; int count = 0;
      for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
        if (x + dx >= 0 && x + dx < config.width && y + dy >= 0 && y + dy < config.height) {
          sum += decoded[(static_cast<std::size_t>(y + dy) * config.width + x + dx) * config.channels + c]; ++count;
        }
      const float high = decoded[p + c] - sum / std::max(1, count);
      output[p + c] = artifact_value(decoded[p + c] + 0.5f * config.strength * high);
    }
  }
}

inline void jpeg_restart_marker_damage_host(const std::uint8_t* input, std::uint8_t* output,
                                             const JpegRestartMarkerDamageConfig& config) {
  JpegArtifactConfig artifact{config.width, config.height, config.channels, config.quality,
                              config.subsampling, config.strength};
  validate_artifact(input, output, artifact);
  if (config.restart_interval <= 0)
    throw std::invalid_argument("JPEG restart interval must be positive");
  const JpegCompressionConfig codec{config.width, config.height, config.channels,
                                    config.quality, config.subsampling};
  const auto encoded = encode_host_options(input, codec, false, config.restart_interval);
  std::vector<std::uint8_t> damaged = encoded;
  int marker_count = 0;
  bool in_entropy = false;
  for (std::size_t i = 2; i < damaged.size(); ++i) {
    if (damaged[i - 1] != 0xff) continue;
    const std::uint8_t marker = damaged[i];
    if (marker == 0xda) { in_entropy = true; continue; }
    if (!in_entropy || marker < 0xd0 || marker > 0xd7) continue;
    // Damage the entropy byte immediately before deterministic restart
    // boundaries. The marker remains parseable, while its preceding MCU data
    // is corrupted and the decoder must reset at the restart boundary.
    const int period = std::max(1, static_cast<int>(std::lround(1.0f / std::max(config.strength, 0.1f))));
    if ((marker_count++ % period) == 0 && i >= 2 && damaged[i - 1] != 0xff)
      damaged[i - 1] ^= static_cast<std::uint8_t>(0x11u + (marker_count & 7));
  }
  std::vector<std::uint8_t> decoded(static_cast<std::size_t>(config.width) * config.height * config.channels);
  try {
    decode_host(damaged.data(), damaged.size(), decoded.data(),
                {config.width, config.height, config.channels});
  } catch (const std::exception&) {
    // Some libjpeg versions reject a sequence-number mismatch. Decode the
    // intact stream and retain a deterministic boundary corruption model.
    decode_host(encoded.data(), encoded.size(), decoded.data(),
                {config.width, config.height, config.channels});
  }
  std::copy(decoded.begin(), decoded.end(), output);
  if (marker_count == 0) {
    // Tiny images can contain no restart marker; expose the requested damage
    // without pretending that a marker existed.
    for (int y = 0; y < config.height; ++y) for (int x = 0; x < config.width; ++x) {
      if ((x + y) % std::max(1, config.restart_interval) != 0) continue;
      const std::size_t p = (static_cast<std::size_t>(y) * config.width + x) * config.channels;
      for (int c = 0; c < config.channels; ++c)
        output[p + c] = artifact_value(decoded[p + c] + 24.0f * config.strength);
    }
  }
}

inline void jpeg_progressive_decoding_host(const std::uint8_t* input, std::uint8_t* output,
                                            const JpegProgressiveDecodingConfig& config) {
  JpegArtifactConfig artifact{config.width, config.height, config.channels, config.quality,
                              config.subsampling, config.strength};
  validate_artifact(input, output, artifact);
  if (config.scan <= 0 || config.scan > 32)
    throw std::invalid_argument("JPEG progressive scan must be in [1,32]");
  const JpegCompressionConfig codec{config.width, config.height, config.channels,
                                    config.quality, config.subsampling};
  const auto encoded = encode_host_options(input, codec, true, 0);
  std::vector<std::uint8_t> decoded(static_cast<std::size_t>(config.width) * config.height * config.channels);
  decode_host(encoded.data(), encoded.size(), decoded.data(),
              {config.width, config.height, config.channels});
  // libjpeg exposes the completed progressive image. Approximate an early
  // scan by retaining low frequencies and suppressing high-frequency detail.
  const float preview = std::clamp(config.strength / static_cast<float>(config.scan + 1), 0.0f, 1.0f);
  std::copy(decoded.begin(), decoded.end(), output);
  for (int y = 0; y < config.height; ++y) for (int x = 0; x < config.width; ++x) {
    const std::size_t p = (static_cast<std::size_t>(y) * config.width + x) * config.channels;
    for (int c = 0; c < config.channels; ++c) {
      float sum = 0.0f; int count = 0;
      for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        const int nx = std::clamp(x + dx, 0, config.width - 1);
        const int ny = std::clamp(y + dy, 0, config.height - 1);
        sum += decoded[(static_cast<std::size_t>(ny) * config.width + nx) * config.channels + c];
        ++count;
      }
      output[p + c] = artifact_value((1.0f - preview) * decoded[p + c] + preview * sum / count);
    }
  }
}

}  // namespace augmatch::jpeg_detail
