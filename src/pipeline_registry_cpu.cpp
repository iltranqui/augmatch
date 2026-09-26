#include "augmatch/pipeline_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "augmatch/color.hpp"
#include "augmatch/filter.hpp"
#include "augmatch/hsv.hpp"
#include "augmatch/transforms.hpp"
#if AUGMATCH_HAS_CUDA
#include "pipeline_host_primitives.hpp"
#endif

namespace augmatch {
namespace {

struct ExposureContext { float multiplier; };
struct SaturationContext { float multiplier; };
struct HueShiftContext { float amount; };
struct ColorJitterContext { float contrast; float brightness; };
struct GaussianBlurContext { int kernel_size; float sigma; };

using Registry = std::unordered_map<std::string, CpuStageFactory>;

constexpr float kMaxHsvMultiplier = 8000000.0f;
constexpr float kMaxHueAmount = 1000000000.0f;

Result<RegisteredCpuStage> invalid(const char* message) {
  return Result<RegisteredCpuStage>::failure({StatusCode::InvalidArgument, message});
}

bool has_only(const StageParameters& params,
              std::initializer_list<const char*> allowed) {
  for (const auto& entry : params) {
    bool found = false;
    for (const char* name : allowed) {
      if (entry.first == name) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

const float* find_parameter(const StageParameters& params, const char* name) {
  const auto found = params.find(name);
  return found == params.end() ? nullptr : &found->second;
}

template <typename Context>
Result<RegisteredCpuStage> owned_stage(Context context, CpuStageCallback callback) {
  auto owner = std::make_shared<Context>(std::move(context));
  RegisteredCpuStage registered;
  registered.stage.callback = callback;
  registered.stage.context = owner.get();
  registered.context = std::move(owner);
  return Result<RegisteredCpuStage>::success(std::move(registered));
}

void exposure_stage(MutableImageView image, const ExecutionContext& execution,
                    void* opaque) {
  const auto& context = *static_cast<const ExposureContext*>(opaque);
  auto* bytes = static_cast<std::uint8_t*>(image.data);
#if AUGMATCH_HAS_CUDA
  augmatch_pipeline_host::multiply_brightness_u8(
      bytes, bytes, {image.width, image.height, image.channels, context.multiplier},
      static_cast<void*>(execution.stream));
#else
  multiply_brightness_u8(bytes, bytes,
                         {image.width, image.height, image.channels, context.multiplier},
                         execution.stream);
#endif
}

void saturation_stage(MutableImageView image, const ExecutionContext& execution,
                      void* opaque) {
  const auto& context = *static_cast<const SaturationContext*>(opaque);
  auto* bytes = static_cast<std::uint8_t*>(image.data);
#if AUGMATCH_HAS_CUDA
  augmatch_pipeline_host::multiply_saturation_u8(
      bytes, bytes, {image.width, image.height, image.channels, context.multiplier},
      static_cast<void*>(execution.stream));
#else
  multiply_saturation_u8(bytes, bytes,
                         {image.width, image.height, image.channels, context.multiplier},
                         execution.stream);
#endif
}

void hue_shift_stage(MutableImageView image, const ExecutionContext& execution,
                     void* opaque) {
  const auto& context = *static_cast<const HueShiftContext*>(opaque);
  auto* bytes = static_cast<std::uint8_t*>(image.data);
#if AUGMATCH_HAS_CUDA
  augmatch_pipeline_host::add_to_hue_u8(
      bytes, bytes, {image.width, image.height, image.channels, context.amount},
      static_cast<void*>(execution.stream));
#else
  add_to_hue_u8(bytes, bytes,
                {image.width, image.height, image.channels, context.amount},
                execution.stream);
#endif
}

void color_jitter_stage(MutableImageView image, const ExecutionContext& execution,
                        void* opaque) {
  const auto& context = *static_cast<const ColorJitterContext*>(opaque);
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  ColorJitterConfig config;
  config.width = image.width;
  config.height = image.height;
  config.channels = image.channels;
  config.brightness = context.brightness;
  config.contrast = context.contrast;
#if AUGMATCH_HAS_CUDA
  augmatch_pipeline_host::ColorJitterConfig host_config;
  host_config.width = config.width;
  host_config.height = config.height;
  host_config.channels = config.channels;
  host_config.brightness = config.brightness;
  host_config.contrast = config.contrast;
  host_config.saturation = config.saturation;
  host_config.hue = config.hue;
  augmatch_pipeline_host::color_jitter_u8(
      bytes, bytes, host_config, static_cast<void*>(execution.stream));
#else
  color_jitter_u8(bytes, bytes, config, execution.stream);
#endif
}

void gaussian_blur_stage(MutableImageView image, const ExecutionContext& execution,
                         void* opaque) {
  const auto& context = *static_cast<const GaussianBlurContext*>(opaque);
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  const std::size_t count = static_cast<std::size_t>(image.width) * image.height * image.channels;
  std::vector<std::uint8_t> output(count);
#if AUGMATCH_HAS_CUDA
  augmatch_pipeline_host::blur_u8(
      bytes, output.data(),
      {image.width, image.height, image.channels, context.kernel_size,
       context.sigma, augmatch_pipeline_host::BlurMode::Gaussian},
      static_cast<void*>(execution.stream));
#else
  blur_u8(bytes, output.data(),
          {image.width, image.height, image.channels, context.kernel_size,
           context.sigma, BlurMode::Gaussian}, execution.stream);
#endif
  std::copy(output.begin(), output.end(), bytes);
}

void horizontal_flip_stage(MutableImageView image, const ExecutionContext& execution,
                           void*) {
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  const std::size_t count = static_cast<std::size_t>(image.width) * image.height * image.channels;
  std::vector<std::uint8_t> output(count);
#if AUGMATCH_HAS_CUDA
  augmatch_pipeline_host::horizontal_flip_u8(
      bytes, output.data(), image.width, image.height, image.channels,
      static_cast<void*>(execution.stream));
#else
  horizontal_flip_u8(bytes, output.data(), image.width, image.height, image.channels,
                     execution.stream);
#endif
  std::copy(output.begin(), output.end(), bytes);
}

Result<RegisteredCpuStage> make_exposure(const StageParameters& params) {
  if (!has_only(params, {"multiplier"})) return invalid("Exposure has an unknown parameter");
  const float* multiplier = find_parameter(params, "multiplier");
  if (multiplier == nullptr) return invalid("Exposure requires multiplier");
  if (!std::isfinite(*multiplier) || *multiplier < 0.0f ||
      *multiplier > kMaxHsvMultiplier)
    return invalid("Exposure multiplier is outside the safe HSV range");
  return owned_stage(ExposureContext{*multiplier}, exposure_stage);
}

Result<RegisteredCpuStage> make_saturation(const StageParameters& params) {
  if (!has_only(params, {"multiplier"})) return invalid("Saturation has an unknown parameter");
  const float* multiplier = find_parameter(params, "multiplier");
  if (multiplier == nullptr) return invalid("Saturation requires multiplier");
  if (!std::isfinite(*multiplier) || *multiplier < 0.0f ||
      *multiplier > kMaxHsvMultiplier)
    return invalid("Saturation multiplier is outside the safe HSV range");
  return owned_stage(SaturationContext{*multiplier}, saturation_stage);
}

Result<RegisteredCpuStage> make_hue_shift(const StageParameters& params) {
  if (!has_only(params, {"amount"})) return invalid("HueShift has an unknown parameter");
  const float* amount = find_parameter(params, "amount");
  if (amount == nullptr) return invalid("HueShift requires amount");
  if (!std::isfinite(*amount) || std::fabs(*amount) > kMaxHueAmount)
    return invalid("HueShift amount is outside the safe HSV range");
  return owned_stage(HueShiftContext{*amount}, hue_shift_stage);
}

Result<RegisteredCpuStage> make_color_jitter(const StageParameters& params,
                                             bool require_contrast) {
  if (!has_only(params, {"contrast", "brightness"}))
    return invalid("Contrast/Brightness has an unknown parameter");
  const float* contrast = find_parameter(params, "contrast");
  const float* brightness = find_parameter(params, "brightness");
  if (require_contrast && contrast == nullptr) return invalid("Contrast requires contrast");
  if (!require_contrast && brightness == nullptr) return invalid("Brightness requires brightness");
  const float contrast_value = contrast == nullptr ? 1.0f : *contrast;
  const float brightness_value = brightness == nullptr ? 0.0f : *brightness;
  if (!std::isfinite(contrast_value) || contrast_value < 0.0f ||
      !std::isfinite(brightness_value))
    return invalid("Contrast/Brightness parameters are invalid");
  return owned_stage(ColorJitterContext{contrast_value, brightness_value},
                     color_jitter_stage);
}

Result<RegisteredCpuStage> make_gaussian_blur(const StageParameters& params) {
  if (!has_only(params, {"kernel_size", "sigma"}))
    return invalid("GaussianBlur has an unknown parameter");
  const float* kernel = find_parameter(params, "kernel_size");
  const float* sigma = find_parameter(params, "sigma");
  if (kernel == nullptr || sigma == nullptr)
    return invalid("GaussianBlur requires kernel_size and sigma");
  if (!std::isfinite(*kernel) || std::floor(*kernel) != *kernel || *kernel < 1.0f ||
      *kernel > 17.0f || (static_cast<int>(*kernel) % 2) == 0)
    return invalid("GaussianBlur kernel_size must be an odd integer in [1,17]");
  if (!std::isfinite(*sigma) || *sigma < 0.0f)
    return invalid("GaussianBlur sigma must be finite and nonnegative");
  const float minimum_positive_sigma =
      std::sqrt(0.5f / std::numeric_limits<float>::max());
  if (*sigma > 0.0f && *sigma < minimum_positive_sigma)
    return invalid("GaussianBlur sigma is too small for finite inverse variance");
  // The underlying finite-kernel implementation requires a positive sigma;
  // zero selects its documented/default scale for cfg compatibility.
  const float effective_sigma = *sigma == 0.0f ? 1.0f : *sigma;
  return owned_stage(GaussianBlurContext{static_cast<int>(*kernel), effective_sigma},
                     gaussian_blur_stage);
}

Result<RegisteredCpuStage> make_horizontal_flip(const StageParameters& params) {
  if (!params.empty()) return invalid("HorizontalFlip takes no parameters");
  RegisteredCpuStage registered;
  registered.stage.callback = horizontal_flip_stage;
  return Result<RegisteredCpuStage>::success(std::move(registered));
}

Registry make_builtin_registry() {
  Registry registry;
  registry.emplace("Exposure", make_exposure);
  registry.emplace("Saturation", make_saturation);
  registry.emplace("HueShift", make_hue_shift);
  registry.emplace("Contrast", [](const StageParameters& params) {
    return make_color_jitter(params, true);
  });
  registry.emplace("Brightness", [](const StageParameters& params) {
    return make_color_jitter(params, false);
  });
  registry.emplace("GaussianBlur", make_gaussian_blur);
  registry.emplace("HorizontalFlip", make_horizontal_flip);
  return registry;
}

Registry& registry() {
  static Registry value = make_builtin_registry();
  return value;
}

std::mutex& registry_mutex() {
  static std::mutex value;
  return value;
}

}  // namespace

Status register_stage(const std::string& name, CpuStageFactory factory) noexcept {
  try {
    if (name.empty() || !factory)
      return {StatusCode::InvalidArgument, "stage name and factory are required"};
    std::lock_guard<std::mutex> lock(registry_mutex());
    if (registry().find(name) != registry().end())
      return {StatusCode::InvalidArgument, "stage name is already registered"};
    registry().emplace(name, std::move(factory));
    return Status::success();
  } catch (...) {
    return {StatusCode::ExecutionError, "failed to register stage"};
  }
}

Result<RegisteredCpuStage> make_registered_cpu_stage(
    const std::string& name, const StageParameters& params) noexcept {
  try {
    CpuStageFactory factory;
    {
      std::lock_guard<std::mutex> lock(registry_mutex());
      const auto found = registry().find(name);
      if (found == registry().end())
        return invalid("unknown pipeline stage");
      factory = found->second;
    }
    auto result = factory(params);
    if (result.ok() && result.value().stage.callback == nullptr)
      return invalid("stage factory returned a null callback");
    return result;
  } catch (const std::invalid_argument&) {
    return invalid("stage factory rejected its parameters");
  } catch (...) {
    return Result<RegisteredCpuStage>::failure(
        {StatusCode::ExecutionError, "stage factory failed"});
  }
}

}  // namespace augmatch
