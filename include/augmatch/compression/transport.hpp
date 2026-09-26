#pragma once
#include <cstddef>
#include <cstdint>
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {

// Byte transport operations use contiguous host bytes on CPU and contiguous
// device bytes on CUDA. No codec framing or external codec is implied.
struct TransportByteConfig {
  std::size_t bytes = 0;
  std::size_t preserve_bytes = 0;
  std::uint8_t fill = 0;
  float probability = 0.0f;
  std::uint32_t bit_mask = 0;
  std::uint64_t seed = 0;
};
// Copies the first preserve_bytes bytes, then fills the remainder with fill.
void truncated_frame_u8(const std::uint8_t* input, std::uint8_t* output, const TransportByteConfig& config, cudaStream_t stream = nullptr);
// Flips bits selected by bit_mask with per-byte probability, keyed by seed.
void bit_flips_u8(const std::uint8_t* input, std::uint8_t* output, const TransportByteConfig& config, cudaStream_t stream = nullptr);
using TruncatedFrameConfig = TransportByteConfig;
using BitFlipConfig = TransportByteConfig;
inline void truncated_frame_corruption_u8(const std::uint8_t* i, std::uint8_t* o, const TransportByteConfig& c, cudaStream_t s = nullptr) { truncated_frame_u8(i, o, c, s); }
inline void bit_flip_corruption_u8(const std::uint8_t* i, std::uint8_t* o, const TransportByteConfig& c, cudaStream_t s = nullptr) { bit_flips_u8(i, o, c, s); }
inline void truncate_frame_u8(const std::uint8_t* i, std::uint8_t* o, const TransportByteConfig& c, cudaStream_t s = nullptr) { truncated_frame_u8(i, o, c, s); }
inline void bit_flip_u8(const std::uint8_t* i, std::uint8_t* o, const TransportByteConfig& c, cudaStream_t s = nullptr) { bit_flips_u8(i, o, c, s); }

// Frame operations use interleaved HWC uint8 pixels. The block grid is
// floor-divided with a final partial block at each right/bottom edge.
struct TransportFrameConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int block_size = 8;
  int quant_step = 1;
  float strength = 1.0f;
  float probability = 0.0f;
  std::uint8_t fill = 0;
  const std::uint8_t* block_mask = nullptr; // ceil(width/B)*ceil(height/B), nonzero means lost.
  std::uint64_t seed = 0;
};
// Quantizes each block by quant_step, scaled by strength.
void block_quantization_u8(const std::uint8_t* input, std::uint8_t* output, const TransportFrameConfig& config, cudaStream_t stream = nullptr);
// Introduces a deterministic mismatch across block boundaries after deblocking.
void deblocking_filter_mismatch_u8(const std::uint8_t* input, std::uint8_t* output, const TransportFrameConfig& config, cudaStream_t stream = nullptr);
// Fills lost macroblocks (selected by probability, seed, or block_mask) with fill.
void packet_loss_macroblocks_u8(const std::uint8_t* input, std::uint8_t* output, const TransportFrameConfig& config, cudaStream_t stream = nullptr);
using BlockQuantizationConfig = TransportFrameConfig;
using DeblockingFilterMismatchConfig = TransportFrameConfig;
using PacketLossMacroblocksConfig = TransportFrameConfig;
inline void block_quantization(const std::uint8_t* i, std::uint8_t* o, const TransportFrameConfig& c, cudaStream_t s = nullptr) { block_quantization_u8(i, o, c, s); }
inline void deblocking_filter_mismatch(const std::uint8_t* i, std::uint8_t* o, const TransportFrameConfig& c, cudaStream_t s = nullptr) { deblocking_filter_mismatch_u8(i, o, c, s); }
inline void packet_loss_macroblocks(const std::uint8_t* i, std::uint8_t* o, const TransportFrameConfig& c, cudaStream_t s = nullptr) { packet_loss_macroblocks_u8(i, o, c, s); }
inline void packet_loss_macroblock_u8(const std::uint8_t* i, std::uint8_t* o, const TransportFrameConfig& c, cudaStream_t s = nullptr) { packet_loss_macroblocks_u8(i, o, c, s); }

// A plane is a single row-major uint8 raster. Chroma misalignment shifts a
// plane by integer pixels with edge clamping; callers apply it independently
// to U and V, including subsampled planes.
struct TransportPlaneConfig {
  int width = 0;
  int height = 0;
  int stride = 0;
  int dx = 0;
  int dy = 0;
  std::uint8_t fill = 0;
};
void chroma_plane_misalignment_u8(const std::uint8_t* input, std::uint8_t* output, const TransportPlaneConfig& config, cudaStream_t stream = nullptr);
using ChromaPlaneMisalignmentConfig = TransportPlaneConfig;
inline void chroma_plane_shift_u8(const std::uint8_t* i, std::uint8_t* o, const TransportPlaneConfig& c, cudaStream_t s = nullptr) { chroma_plane_misalignment_u8(i, o, c, s); }

// Raw Bayer corruption is deliberately a transport operation on unpacked
// sample values, not CFA sampling or demosaicing. Values are clipped to the
// configured bit depth; mask is an optional HxW corruption map.
struct RawBayerTransportCorruptionConfig {
  int width = 0;
  int height = 0;
  int stride = 0;
  int bit_depth = 12;
  float probability = 0.0f;
  std::uint16_t fill = 0;
  std::uint16_t xor_mask = 0;
  const std::uint8_t* mask = nullptr;
  std::uint64_t seed = 0;
};
void raw_bayer_transport_corruption_u16(const std::uint16_t* input, std::uint16_t* output, const RawBayerTransportCorruptionConfig& config, cudaStream_t stream = nullptr);
// Byte-oriented raw Bayer transport entry point for packed or opaque samples.
void raw_bayer_transport_corruption_u8(const std::uint8_t* input, std::uint8_t* output, const TransportByteConfig& config, cudaStream_t stream = nullptr);
inline void raw_bayer_corruption_u16(const std::uint16_t* i, std::uint16_t* o, const RawBayerTransportCorruptionConfig& c, cudaStream_t s = nullptr) { raw_bayer_transport_corruption_u16(i, o, c, s); }

} // namespace augmatch
