#include "augmatch/augmatch.hpp"
#include "augmatch/noise_view.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

int main() {
#if AUGMATCH_HAS_CUDA
  // This unit test owns host buffers; CUDA stream/device execution is covered
  // by the CUDA validation suite when a device is available.
  return 77;
#endif
  constexpr int width = 3, height = 2, channels = 2;
  std::vector<std::uint8_t> source(width * height * channels + 4, 0);
  for (int i = 0; i < width * height * channels; ++i) source[i + 1] = static_cast<std::uint8_t>(i * 10);
  std::vector<float> first(width * height * channels, 0.0f);
  std::vector<float> second(first.size(), 0.0f);
  // The extra leading byte exercises a genuinely strided HWC view.
  augmatch::ImageView input{source.data() + 1, width, height, channels,
                            channels, width * channels + 1, 1,
                            augmatch::DataType::UInt8, augmatch::Layout::HWC,
                            augmatch::MemorySpace::Host};
  auto output = augmatch::make_hwc_view(first.data(), width, height, channels);
  augmatch::NoiseViewConfig config;
  config.seed = 1234;
  config.stddev = 0.05f;
  config.quantization = augmatch::NoiseQuantizationPolicy::Preserve;
  augmatch::NoiseChannelMetadata channel_meta[2]{{1.0f, 0.1f, 1.0f, 0},
                                                  {2.0f, -0.1f, 0.5f, 1}};
  augmatch::NoisePlaneMetadata plane_meta[2]{{1.0f, 0.0f, 1.0f},
                                              {0.5f, 0.0f, 2.0f}};
  augmatch::NoiseMetadataView metadata{channel_meta, 2, plane_meta, 2, nullptr,
                                       augmatch::MemorySpace::Host};
  if (!augmatch::additive_noise_view(input, output, config, metadata)) return 1;
  output.data = second.data();
  if (!augmatch::additive_noise_view(input, output, config, metadata)) return 2;
  if (first != second) return 3;
  if (std::abs(first[0] - 0.1f) > 0.2f) return 4;

  // A CHW float view is accepted and has deterministic per logical coordinate.
  std::vector<float> chw(width * height * channels, 0.25f), chw_out(chw.size());
  auto chw_input = augmatch::make_chw_view(chw.data(), width, height, channels);
  auto chw_output = augmatch::make_chw_view(chw_out.data(), width, height, channels);
  config.stddev = 0.0f;
  config.mean = 0.25f;
  metadata = {};
  if (!augmatch::additive_noise_view(chw_input.as_const(), chw_output, config, metadata)) return 5;
  for (float value : chw_out) if (std::abs(value - 0.5f) > 1e-6f) return 6;

  std::uint16_t integer_input[] = {0, 65535};
  std::uint16_t integer_output[] = {0, 0};
  auto integer_view = augmatch::make_hwc_u16_view(integer_input, 2, 1, 1);
  auto integer_result = augmatch::make_hwc_u16_view(integer_output, 2, 1, 1);
  config.mean = 0.25f;
  config.quantization = augmatch::NoiseQuantizationPolicy::RoundHalfUp;
  if (!augmatch::additive_noise_view(integer_view.as_const(), integer_result, config, {})) return 7;
  if (integer_output[0] != 16384 || integer_output[1] != 65535) return 8;

  if (augmatch::resolve_noise_coordinate(-1, 4, augmatch::NoiseBorderPolicy::Clamp) != 0) return 9;
  if (augmatch::resolve_noise_coordinate(-1, 4, augmatch::NoiseBorderPolicy::Reflect101) != 1) return 10;
  bool constant = false;
  if (augmatch::resolve_noise_coordinate(-1, 4, augmatch::NoiseBorderPolicy::Constant, &constant) != -1 || !constant) return 11;
  auto bad = input;
  bad.width = 0;
  if (augmatch::validate_noise_view(bad, output, config, metadata).code !=
      augmatch::StatusCode::InvalidView) return 12;
  return 0;
}
