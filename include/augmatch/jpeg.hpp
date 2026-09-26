#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif

namespace augmatch {

// The sampling factors used by the JPEG YCbCr encoder.  Y444 keeps chroma at
// full resolution; Y422 and Y420 use the standard horizontal and 2-D chroma
// reductions respectively.
enum class JpegSubsampling : int {
  Y444 = 0,
  Y422 = 1,
  Y420 = 2,
  // Source-compatible descriptive aliases.
  S444 = Y444,
  S422 = Y422,
  S420 = Y420
};

// JPEG accepts contiguous interleaved HWC uint8 images with one (gray) or
// three (RGB) channels. Quality is the libjpeg quality scale [1,100]. The
// encoded stream is libjpeg-turbo output; no WebP or AV1 codec is implied.
struct JpegCompressionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int quality = 75;
  JpegSubsampling subsampling = JpegSubsampling::Y420;
  // Optional multiplicative factors for the luminance/chrominance quantization
  // tables. One is the libjpeg quality table; values greater than one add loss.
  float luma_quant_scale = 1.0f;
  float chroma_quant_scale = 1.0f;
};
using ImageCompressionConfig = JpegCompressionConfig;
using JpegQualityVariationConfig = JpegCompressionConfig;

struct JpegQuantizationTableVariationConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int quality = 75;
  JpegSubsampling subsampling = JpegSubsampling::Y420;
  float luminance_scale = 1.0f;
  float chrominance_scale = 1.0f;
};

struct JpegArtifactConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int quality = 35;
  JpegSubsampling subsampling = JpegSubsampling::Y420;
  float strength = 1.0f;
};
using JpegRingingConfig = JpegArtifactConfig;
using JpegBlockingConfig = JpegArtifactConfig;
using JpegMosquitoNoiseConfig = JpegArtifactConfig;

// Restart damage and progressive decoding are still-image codec operations.
// They use the same contiguous HWC uint8 ownership contract as JPEG round trips.
struct JpegRestartMarkerDamageConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int quality = 35;
  JpegSubsampling subsampling = JpegSubsampling::Y420;
  float strength = 1.0f;
  int restart_interval = 4;  // MCU interval used to emit restart markers.
};

struct JpegProgressiveDecodingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int quality = 35;
  JpegSubsampling subsampling = JpegSubsampling::Y420;
  float strength = 1.0f;
  int scan = 1;  // Lower scans retain more of the low-frequency preview.
};
using JpegProgressiveDecodingArtifactsConfig = JpegProgressiveDecodingConfig;
using JpegRestartMarkerConfig = JpegRestartMarkerDamageConfig;
using JpegProgressiveConfig = JpegProgressiveDecodingConfig;

struct JpegDecodeConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
};

// Encode returns a host-owned JPEG byte vector. In a CPU build input is host
// memory. In a CUDA build input is device memory and the implementation copies
// it to the host codec; this is deliberately not GPU-native JPEG encoding.
std::vector<std::uint8_t> jpeg_encode_u8(const std::uint8_t*, const JpegCompressionConfig&, cudaStream_t stream = nullptr);

// Decode a host-owned JPEG byte vector into the caller's output. In a CPU
// build output is host memory. In a CUDA build output is device memory and the
// host-decoded result is copied back synchronously.
void jpeg_decode_u8(const std::uint8_t*, std::size_t, std::uint8_t*, const JpegDecodeConfig&, cudaStream_t stream = nullptr);

// Encode and decode in one operation, preserving the configured dimensions and
// channel count. This is the augmentation operation used by both
// ImageCompression and JpegCompression. CUDA uses the documented host codec
// fallback and synchronizes the supplied stream before returning.
void jpeg_compression_u8(const std::uint8_t*, std::uint8_t*, const JpegCompressionConfig&, cudaStream_t stream = nullptr);
void image_compression_u8(const std::uint8_t*, std::uint8_t*, const ImageCompressionConfig&, cudaStream_t stream = nullptr);
// These still-image catalog operations all preserve the input HWC shape.
void jpeg_quality_variation_u8(const std::uint8_t*, std::uint8_t*, const JpegQualityVariationConfig&, cudaStream_t stream = nullptr);
void jpeg_quantization_table_variation_u8(const std::uint8_t*, std::uint8_t*, const JpegQuantizationTableVariationConfig&, cudaStream_t stream = nullptr);
void jpeg_chroma_subsampling_u8(const std::uint8_t*, std::uint8_t*, const JpegCompressionConfig&, cudaStream_t stream = nullptr);
void jpeg_ringing_u8(const std::uint8_t*, std::uint8_t*, const JpegRingingConfig&, cudaStream_t stream = nullptr);
void jpeg_blocking_u8(const std::uint8_t*, std::uint8_t*, const JpegBlockingConfig&, cudaStream_t stream = nullptr);
void jpeg_mosquito_noise_u8(const std::uint8_t*, std::uint8_t*, const JpegMosquitoNoiseConfig&, cudaStream_t stream = nullptr);
void jpeg_restart_marker_damage_u8(const std::uint8_t*, std::uint8_t*, const JpegRestartMarkerDamageConfig&, cudaStream_t stream = nullptr);
void jpeg_progressive_decoding_u8(const std::uint8_t*, std::uint8_t*, const JpegProgressiveDecodingConfig&, cudaStream_t stream = nullptr);
void jpeg_restart_marker_u8(const std::uint8_t*, std::uint8_t*, const JpegRestartMarkerConfig&, cudaStream_t stream = nullptr);
void jpeg_progressive_u8(const std::uint8_t*, std::uint8_t*, const JpegProgressiveConfig&, cudaStream_t stream = nullptr);

}  // namespace augmatch
