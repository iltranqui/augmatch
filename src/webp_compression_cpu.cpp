#include "augmatch/compression/webp.hpp"
#include "webp_codec_impl.hpp"

namespace augmatch {

std::vector<std::uint8_t> webp_encode_u8(const std::uint8_t* input,
                                         const WebPCompressionConfig& config,
                                         cudaStream_t) {
  return webp_detail::encode_host(input, config);
}

void webp_decode_u8(const std::uint8_t* encoded, std::size_t encoded_size,
                    std::uint8_t* output, const WebPDecodeConfig& config,
                    cudaStream_t) {
  webp_detail::decode_host(encoded, encoded_size, output, config);
}

void webp_compression_u8(const std::uint8_t* input, std::uint8_t* output,
                         const WebPCompressionConfig& config, cudaStream_t stream) {
  const auto encoded = webp_encode_u8(input, config, stream);
  webp_decode_u8(encoded.data(), encoded.size(), output,
                 {config.width, config.height, config.channels}, stream);
}

void webp_lossy_compression_u8(const std::uint8_t* input, std::uint8_t* output,
                               const WebPLossyCompressionConfig& config, cudaStream_t stream) {
  WebPCompressionConfig forced = config;
  forced.lossless = false;
  webp_compression_u8(input, output, forced, stream);
}

void webp_lossless_compression_u8(const std::uint8_t* input, std::uint8_t* output,
                                  const WebPLosslessCompressionConfig& config, cudaStream_t stream) {
  WebPCompressionConfig forced = config;
  forced.lossless = true;
  webp_compression_u8(input, output, forced, stream);
}

void webp_lossy_u8(const std::uint8_t* input, std::uint8_t* output,
                   const WebPLossyCompressionConfig& config, cudaStream_t stream) {
  webp_lossy_compression_u8(input, output, config, stream);
}

void webp_lossless_u8(const std::uint8_t* input, std::uint8_t* output,
                      const WebPLosslessCompressionConfig& config, cudaStream_t stream) {
  webp_lossless_compression_u8(input, output, config, stream);
}

}  // namespace augmatch
