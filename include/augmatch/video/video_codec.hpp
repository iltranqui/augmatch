#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#ifndef AUGMATCH_HAS_VIDEO_CODEC
#define AUGMATCH_HAS_VIDEO_CODEC 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
enum class VideoCodec { H264, HEVC, AV1 };
struct VideoCodecConfig {
  int width = 0;
  int height = 0;
  int channels = 3;
  int quality = 25;
  VideoCodec codec = VideoCodec::H264;
};
struct VideoDecodeConfig {
  int width = 0;
  int height = 0;
  int channels = 3;
  VideoCodec codec = VideoCodec::H264;
};
// Encode a single intra frame as a codec packet. CPU buffers are host-owned;
// CUDA buffers are device-owned and use a synchronous host-codec fallback.
std::vector<std::uint8_t> video_codec_encode_u8(const std::uint8_t* input, const VideoCodecConfig& config, cudaStream_t stream = nullptr);
// Decodes a host-owned codec packet into the caller's output buffer.
void video_codec_decode_u8(const std::uint8_t* input, std::size_t size, std::uint8_t* output, const VideoDecodeConfig& config, cudaStream_t stream = nullptr);
// Encodes then decodes a single intra frame in one operation, preserving the
// configured dimensions and channel count.
void video_codec_roundtrip_u8(const std::uint8_t* input, std::uint8_t* output, const VideoCodecConfig& config, cudaStream_t stream = nullptr);
// Deliberately conservative AV1 artifact contract: this API does not edit an
// existing AV1 bitstream or promise parity with a particular AV1 reference.
void av1_intra_frame_artifacts_u8(const std::uint8_t* input, std::uint8_t* output, const VideoCodecConfig& config, cudaStream_t stream = nullptr);
}
