#include "augmatch/webp.hpp"
#include <stdexcept>

namespace augmatch {
namespace {
[[noreturn]] void unavailable() {
  throw std::runtime_error("WebP codec unavailable: configure with libwebp headers and library");
}
}

std::vector<std::uint8_t> webp_encode_u8(const std::uint8_t*, const WebPCompressionConfig&, cudaStream_t) { unavailable(); }
void webp_decode_u8(const std::uint8_t*, std::size_t, std::uint8_t*, const WebPDecodeConfig&, cudaStream_t) { unavailable(); }
void webp_compression_u8(const std::uint8_t*, std::uint8_t*, const WebPCompressionConfig&, cudaStream_t) { unavailable(); }
void webp_lossy_compression_u8(const std::uint8_t*, std::uint8_t*, const WebPLossyCompressionConfig&, cudaStream_t) { unavailable(); }
void webp_lossless_compression_u8(const std::uint8_t*, std::uint8_t*, const WebPLosslessCompressionConfig&, cudaStream_t) { unavailable(); }
void webp_lossy_u8(const std::uint8_t*, std::uint8_t*, const WebPLossyCompressionConfig&, cudaStream_t) { unavailable(); }
void webp_lossless_u8(const std::uint8_t*, std::uint8_t*, const WebPLosslessCompressionConfig&, cudaStream_t) { unavailable(); }
}  // namespace augmatch
