#include "augmatch/compression/webp.hpp"
#include "webp_codec_impl.hpp"
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

std::size_t image_bytes(const WebPCompressionConfig& config) {
  if (config.width <= 0 || config.height <= 0 ||
      (config.channels != 1 && config.channels != 3) ||
      config.width > std::numeric_limits<int>::max() / 3)
    throw std::invalid_argument("invalid CUDA WebP dimensions or channels");
  const auto pixels = static_cast<std::uint64_t>(config.width) *
                      static_cast<std::uint64_t>(config.height) *
                      static_cast<std::uint64_t>(config.channels);
  if (pixels > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("WebP image is too large");
  return static_cast<std::size_t>(pixels);
}
}  // namespace

std::vector<std::uint8_t> webp_encode_u8(const std::uint8_t* input,
                                         const WebPCompressionConfig& config,
                                         cudaStream_t stream) {
  const std::size_t bytes = image_bytes(config);
  if (!input) throw std::invalid_argument("null CUDA WebP input");
  std::vector<std::uint8_t> host_input(bytes);
  check_cuda(cudaMemcpyAsync(host_input.data(), input, bytes, cudaMemcpyDeviceToHost, stream),
             "copy WebP input to host");
  check_cuda(cudaStreamSynchronize(stream), "synchronize WebP host encode input");
  return webp_detail::encode_host(host_input.data(), config);
}

void webp_decode_u8(const std::uint8_t* encoded, std::size_t encoded_size,
                    std::uint8_t* output, const WebPDecodeConfig& config,
                    cudaStream_t stream) {
  if (!output || config.width <= 0 || config.height <= 0 ||
      (config.channels != 1 && config.channels != 3) ||
      config.width > std::numeric_limits<int>::max() / 3)
    throw std::invalid_argument("invalid CUDA WebP output configuration");
  const auto bytes = static_cast<std::uint64_t>(config.width) *
                     static_cast<std::uint64_t>(config.height) *
                     static_cast<std::uint64_t>(config.channels);
  if (bytes > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("WebP image is too large");
  std::vector<std::uint8_t> host_output(static_cast<std::size_t>(bytes));
  webp_detail::decode_host(encoded, encoded_size, host_output.data(), config);
  check_cuda(cudaMemcpyAsync(output, host_output.data(), host_output.size(),
                             cudaMemcpyHostToDevice, stream),
             "copy WebP decode result to device");
  check_cuda(cudaStreamSynchronize(stream), "synchronize WebP host decode result");
}

void webp_compression_u8(const std::uint8_t* input, std::uint8_t* output,
                         const WebPCompressionConfig& config, cudaStream_t stream) {
  const std::size_t bytes = image_bytes(config);
  if (!input || !output) throw std::invalid_argument("null CUDA WebP buffer");
  std::vector<std::uint8_t> host_input(bytes), host_output(bytes);
  check_cuda(cudaMemcpyAsync(host_input.data(), input, bytes, cudaMemcpyDeviceToHost, stream),
             "copy WebP round-trip input to host");
  check_cuda(cudaStreamSynchronize(stream), "synchronize WebP round-trip input");
  const auto encoded = webp_detail::encode_host(host_input.data(), config);
  webp_detail::decode_host(encoded.data(), encoded.size(), host_output.data(),
                           {config.width, config.height, config.channels});
  check_cuda(cudaMemcpyAsync(output, host_output.data(), bytes, cudaMemcpyHostToDevice, stream),
             "copy WebP round-trip result to device");
  check_cuda(cudaStreamSynchronize(stream), "synchronize WebP round-trip result");
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
