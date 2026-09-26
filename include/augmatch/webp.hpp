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

// WebP accepts contiguous interleaved HWC uint8 images with one grayscale or
// three RGB channels. Quality is libwebp's [0,100] quality scale. When
// lossless is true, libwebp's lossless mode is selected explicitly; quality
// remains part of the configuration for a stable API but does not alter the
// lossless bitstream semantics.
struct WebPCompressionConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int quality = 75;
  bool lossless = false;
};

struct WebPDecodeConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
};
using WebPLossyCompressionConfig = WebPCompressionConfig;
using WebPLosslessCompressionConfig = WebPCompressionConfig;

// Encode returns a host-owned WebP byte vector. In a CPU build input is host
// memory. In a CUDA build input is device memory and the implementation copies
// it to the host libwebp codec; this is deliberately not GPU-native WebP.
std::vector<std::uint8_t> webp_encode_u8(const std::uint8_t*, const WebPCompressionConfig&, cudaStream_t stream = nullptr);

// Decode a host-owned WebP byte vector into the caller's output. In a CPU
// build output is host memory. In a CUDA build output is device memory and
// the host-decoded result is copied back synchronously.
void webp_decode_u8(const std::uint8_t*, std::size_t, std::uint8_t*, const WebPDecodeConfig&, cudaStream_t stream = nullptr);

// Encode and decode in one operation, preserving the configured dimensions and
// channel count. CUDA uses the documented host-codec fallback and synchronizes
// the supplied stream before returning.
void webp_compression_u8(const std::uint8_t*, std::uint8_t*, const WebPCompressionConfig&, cudaStream_t stream = nullptr);
// Explicit catalog entry points force the selected libwebp mode and are useful
// to callers that do not want a mode boolean in a generic compression config.
void webp_lossy_compression_u8(const std::uint8_t*, std::uint8_t*, const WebPLossyCompressionConfig&, cudaStream_t stream = nullptr);
void webp_lossless_compression_u8(const std::uint8_t*, std::uint8_t*, const WebPLosslessCompressionConfig&, cudaStream_t stream = nullptr);
void webp_lossy_u8(const std::uint8_t*, std::uint8_t*, const WebPLossyCompressionConfig&, cudaStream_t stream = nullptr);
void webp_lossless_u8(const std::uint8_t*, std::uint8_t*, const WebPLosslessCompressionConfig&, cudaStream_t stream = nullptr);

}  // namespace augmatch
