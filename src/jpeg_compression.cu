#include "augmatch/compression/jpeg.hpp"
#include "jpeg_codec_impl.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {
void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
std::size_t image_bytes(const JpegCompressionConfig& config) {
  if (config.width <= 0 || config.height <= 0 || (config.channels != 1 && config.channels != 3))
    throw std::invalid_argument("invalid JPEG dimensions or channels");
  const auto pixels = static_cast<std::uint64_t>(config.width) * config.height * config.channels;
  if (pixels > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("JPEG image is too large");
  return static_cast<std::size_t>(pixels);
}
std::size_t artifact_bytes(const JpegArtifactConfig& config) {
  JpegCompressionConfig base{config.width, config.height, config.channels, config.quality, config.subsampling};
  return image_bytes(base);
}
template <typename Config>
void run_artifact(const std::uint8_t* input, std::uint8_t* output, const Config& config,
                  cudaStream_t stream,
                  void (*host_fn)(const std::uint8_t*, std::uint8_t*, const Config&)) {
  const std::size_t bytes = artifact_bytes(config);
  if (!input || !output) throw std::invalid_argument("null CUDA JPEG artifact buffer");
  std::vector<std::uint8_t> host_input(bytes), host_output(bytes);
  check_cuda(cudaMemcpyAsync(host_input.data(), input, bytes, cudaMemcpyDeviceToHost, stream),
             "copy JPEG artifact input to host");
  check_cuda(cudaStreamSynchronize(stream), "synchronize JPEG artifact input");
  host_fn(host_input.data(), host_output.data(), config);
  check_cuda(cudaMemcpyAsync(output, host_output.data(), bytes, cudaMemcpyHostToDevice, stream),
             "copy JPEG artifact result to device");
  check_cuda(cudaStreamSynchronize(stream), "synchronize JPEG artifact result");
}

template <typename Config>
void run_mode_artifact(const std::uint8_t* input, std::uint8_t* output, const Config& config,
                       cudaStream_t stream,
                       void (*host_fn)(const std::uint8_t*, std::uint8_t*, const Config&)) {
  const auto bytes = static_cast<std::size_t>(config.width) * config.height * config.channels;
  if (config.width <= 0 || config.height <= 0 || (config.channels != 1 && config.channels != 3) ||
      !input || !output)
    throw std::invalid_argument("invalid CUDA JPEG artifact configuration or buffer");
  std::vector<std::uint8_t> host_input(bytes), host_output(bytes);
  check_cuda(cudaMemcpyAsync(host_input.data(), input, bytes, cudaMemcpyDeviceToHost, stream),
             "copy JPEG mode artifact input to host");
  check_cuda(cudaStreamSynchronize(stream), "synchronize JPEG mode artifact input");
  host_fn(host_input.data(), host_output.data(), config);
  check_cuda(cudaMemcpyAsync(output, host_output.data(), bytes, cudaMemcpyHostToDevice, stream),
             "copy JPEG mode artifact result to device");
  check_cuda(cudaStreamSynchronize(stream), "synchronize JPEG mode artifact result");
}
}

std::vector<std::uint8_t> jpeg_encode_u8(const std::uint8_t* input,
                                         const JpegCompressionConfig& config,
                                         cudaStream_t stream) {
  const std::size_t bytes = image_bytes(config);
  if (!input) throw std::invalid_argument("null CUDA JPEG input");
  std::vector<std::uint8_t> host_input(bytes);
  check_cuda(cudaMemcpyAsync(host_input.data(), input, bytes, cudaMemcpyDeviceToHost, stream),
             "copy JPEG input to host");
  check_cuda(cudaStreamSynchronize(stream), "synchronize JPEG host encode input");
  return jpeg_detail::encode_host(host_input.data(), config);
}

void jpeg_decode_u8(const std::uint8_t* encoded, std::size_t encoded_size,
                    std::uint8_t* output, const JpegDecodeConfig& config,
                    cudaStream_t stream) {
  if (!output || config.width <= 0 || config.height <= 0 ||
      (config.channels != 1 && config.channels != 3))
    throw std::invalid_argument("invalid CUDA JPEG output configuration");
  const std::size_t bytes = static_cast<std::size_t>(config.width) * config.height * config.channels;
  std::vector<std::uint8_t> host_output(bytes);
  jpeg_detail::decode_host(encoded, encoded_size, host_output.data(), config);
  check_cuda(cudaMemcpyAsync(output, host_output.data(), bytes, cudaMemcpyHostToDevice, stream),
             "copy JPEG decode result to device");
  check_cuda(cudaStreamSynchronize(stream), "synchronize JPEG host decode result");
}

void jpeg_compression_u8(const std::uint8_t* input, std::uint8_t* output,
                         const JpegCompressionConfig& config, cudaStream_t stream) {
  const std::size_t bytes = image_bytes(config);
  if (!input || !output) throw std::invalid_argument("null CUDA JPEG buffer");
  std::vector<std::uint8_t> host_input(bytes), host_output(bytes);
  check_cuda(cudaMemcpyAsync(host_input.data(), input, bytes, cudaMemcpyDeviceToHost, stream),
             "copy JPEG round-trip input to host");
  check_cuda(cudaStreamSynchronize(stream), "synchronize JPEG round-trip input");
  const auto encoded = jpeg_detail::encode_host(host_input.data(), config);
  jpeg_detail::decode_host(encoded.data(), encoded.size(), host_output.data(),
                            {config.width, config.height, config.channels});
  check_cuda(cudaMemcpyAsync(output, host_output.data(), bytes, cudaMemcpyHostToDevice, stream),
             "copy JPEG round-trip result to device");
  check_cuda(cudaStreamSynchronize(stream), "synchronize JPEG round-trip result");
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
                     const JpegRingingConfig& config, cudaStream_t stream) {
  run_artifact(input, output, config, stream, jpeg_detail::jpeg_ringing_host);
}

void jpeg_blocking_u8(const std::uint8_t* input, std::uint8_t* output,
                      const JpegBlockingConfig& config, cudaStream_t stream) {
  run_artifact(input, output, config, stream, jpeg_detail::jpeg_blocking_host);
}

void jpeg_mosquito_noise_u8(const std::uint8_t* input, std::uint8_t* output,
                            const JpegMosquitoNoiseConfig& config, cudaStream_t stream) {
  run_artifact(input, output, config, stream, jpeg_detail::jpeg_mosquito_noise_host);
}

void jpeg_restart_marker_damage_u8(const std::uint8_t* input, std::uint8_t* output,
                                   const JpegRestartMarkerDamageConfig& config, cudaStream_t stream) {
  run_mode_artifact(input, output, config, stream, jpeg_detail::jpeg_restart_marker_damage_host);
}

void jpeg_progressive_decoding_u8(const std::uint8_t* input, std::uint8_t* output,
                                  const JpegProgressiveDecodingConfig& config, cudaStream_t stream) {
  run_mode_artifact(input, output, config, stream, jpeg_detail::jpeg_progressive_decoding_host);
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
