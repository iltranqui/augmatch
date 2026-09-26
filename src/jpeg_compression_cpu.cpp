#include "augmatch/compression/jpeg.hpp"
#include "jpeg_codec_impl.hpp"

namespace augmatch {

std::vector<std::uint8_t> jpeg_encode_u8(const std::uint8_t* input,
                                         const JpegCompressionConfig& config,
                                         cudaStream_t) {
  return jpeg_detail::encode_host(input, config);
}

void jpeg_decode_u8(const std::uint8_t* encoded, std::size_t encoded_size,
                    std::uint8_t* output, const JpegDecodeConfig& config,
                    cudaStream_t) {
  jpeg_detail::decode_host(encoded, encoded_size, output, config);
}

void jpeg_compression_u8(const std::uint8_t* input, std::uint8_t* output,
                         const JpegCompressionConfig& config, cudaStream_t stream) {
  const auto encoded = jpeg_encode_u8(input, config, stream);
  jpeg_decode_u8(encoded.data(), encoded.size(), output,
                 {config.width, config.height, config.channels}, stream);
}

void image_compression_u8(const std::uint8_t* input, std::uint8_t* output,
                          const ImageCompressionConfig& config, cudaStream_t stream) {
  jpeg_compression_u8(input, output, config, stream);
}

void jpeg_quality_variation_u8(const std::uint8_t* input, std::uint8_t* output,
                               const JpegQualityVariationConfig& config, cudaStream_t stream) {
  jpeg_compression_u8(input, output, config, stream);
}

void jpeg_quantization_table_variation_u8(const std::uint8_t* input, std::uint8_t* output,
                                          const JpegQuantizationTableVariationConfig& config,
                                          cudaStream_t stream) {
  JpegCompressionConfig codec{config.width, config.height, config.channels, config.quality,
                              config.subsampling, config.luminance_scale, config.chrominance_scale};
  jpeg_compression_u8(input, output, codec, stream);
}

void jpeg_chroma_subsampling_u8(const std::uint8_t* input, std::uint8_t* output,
                                const JpegCompressionConfig& config, cudaStream_t stream) {
  jpeg_compression_u8(input, output, config, stream);
}

void jpeg_ringing_u8(const std::uint8_t* input, std::uint8_t* output,
                     const JpegRingingConfig& config, cudaStream_t) {
  jpeg_detail::jpeg_ringing_host(input, output, config);
}

void jpeg_blocking_u8(const std::uint8_t* input, std::uint8_t* output,
                      const JpegBlockingConfig& config, cudaStream_t) {
  jpeg_detail::jpeg_blocking_host(input, output, config);
}

void jpeg_mosquito_noise_u8(const std::uint8_t* input, std::uint8_t* output,
                            const JpegMosquitoNoiseConfig& config, cudaStream_t) {
  jpeg_detail::jpeg_mosquito_noise_host(input, output, config);
}

void jpeg_restart_marker_damage_u8(const std::uint8_t* input, std::uint8_t* output,
                                   const JpegRestartMarkerDamageConfig& config, cudaStream_t) {
  jpeg_detail::jpeg_restart_marker_damage_host(input, output, config);
}

void jpeg_progressive_decoding_u8(const std::uint8_t* input, std::uint8_t* output,
                                  const JpegProgressiveDecodingConfig& config, cudaStream_t) {
  jpeg_detail::jpeg_progressive_decoding_host(input, output, config);
}

void jpeg_restart_marker_u8(const std::uint8_t* input, std::uint8_t* output,
                            const JpegRestartMarkerConfig& config, cudaStream_t stream) {
  jpeg_restart_marker_damage_u8(input, output, config, stream);
}

void jpeg_progressive_u8(const std::uint8_t* input, std::uint8_t* output,
                         const JpegProgressiveConfig& config, cudaStream_t stream) {
  jpeg_progressive_decoding_u8(input, output, config, stream);
}

}  // namespace augmatch
