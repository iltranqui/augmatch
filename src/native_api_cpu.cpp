#include "augmatch/sensor/native_api.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace augmatch {

namespace {
Status bad(StatusCode code, const char* message) noexcept { return {code, message}; }
}

bool RawBayerView::valid() const noexcept {
  return data != nullptr && width > 0 && height > 0 &&
         stride_bytes >= static_cast<std::ptrdiff_t>(width) *
                         static_cast<std::ptrdiff_t>(bytes_per_element(type));
}
bool MutableRawBayerView::valid() const noexcept {
  return data != nullptr && width > 0 && height > 0 &&
         stride_bytes >= static_cast<std::ptrdiff_t>(width) *
                         static_cast<std::ptrdiff_t>(bytes_per_element(type));
}
RawBayerView make_raw_bayer_view(const void* data, int width, int height,
                                 std::ptrdiff_t stride_bytes, DataType type,
                                 BayerPattern pattern, MemorySpace memory) noexcept {
  return {data, width, height, stride_bytes, type, pattern, memory};
}
MutableRawBayerView make_raw_bayer_view(void* data, int width, int height,
                                        std::ptrdiff_t stride_bytes, DataType type,
                                        BayerPattern pattern, MemorySpace memory) noexcept {
  return {data, width, height, stride_bytes, type, pattern, memory};
}
Status validate_raw_bayer_view(const RawBayerView& input,
                               const MutableRawBayerView& output) noexcept {
  if (!input.valid() || !output.valid()) return bad(StatusCode::InvalidView, "invalid raw Bayer view");
  if (input.width != output.width || input.height != output.height || input.type != output.type ||
      input.pattern != output.pattern || input.memory != output.memory)
    return bad(StatusCode::InvalidView, "raw Bayer views differ");
  if (input.type != DataType::UInt8 && input.type != DataType::UInt16 && input.type != DataType::Float32)
    return bad(StatusCode::InvalidView, "unsupported raw Bayer type");
  return Status::success();
}

Status validate_camera_profile(const CameraProfileConfig& profile) noexcept {
  if (profile.points.empty()) return bad(StatusCode::InvalidMetadata, "camera profile has no points");
  if (!std::isfinite(profile.iso) || profile.iso <= 0.0f)
    return bad(StatusCode::InvalidMetadata, "camera profile ISO is invalid");
  float previous = 0.0f;
  for (const auto& point : profile.points) {
    if (!std::isfinite(point.iso) || point.iso <= 0.0f ||
        (previous != 0.0f && point.iso <= previous) ||
        !std::isfinite(point.gain) || !std::isfinite(point.shot_scale) ||
        !std::isfinite(point.read_noise_stddev) || !std::isfinite(point.fpn_stddev) ||
        point.gain < 0.0f || point.shot_scale < 0.0f || point.read_noise_stddev < 0.0f ||
        point.fpn_stddev < 0.0f || point.black_level < 0.0f || point.black_level > 1.0f ||
        point.saturation_level < 0.0f || point.saturation_level > 1.0f)
      return bad(StatusCode::InvalidMetadata, "invalid or unsorted camera profile point");
    previous = point.iso;
  }
  return Status::success();
}

Status serialize_camera_profile(const CameraProfileConfig& profile, std::string* text) noexcept {
  if (!text) return bad(StatusCode::InvalidArgument, "profile output is null");
  const Status valid = validate_camera_profile(profile);
  if (!valid) return valid;
  try {
    std::ostringstream out;
    out << "AUGMATCH_CAMERA_PROFILE_V1 " << std::quoted(profile.name) << ' ' << profile.iso << ' '
        << profile.points.size() << '\n';
    for (const auto& p : profile.points)
      out << p.iso << ' ' << p.gain << ' ' << p.shot_noise_scale << ' ' << p.read_noise_scale << ' '
          << p.fpn_scale << ' ' << p.black_level << ' ' << p.saturation_level << ' ' << p.shot_scale
          << ' ' << p.read_noise_stddev << ' ' << p.fpn_stddev << '\n';
    *text = out.str();
    return Status::success();
  } catch (...) { return bad(StatusCode::ExecutionError, "profile serialization failed"); }
}

Status deserialize_camera_profile(const std::string& text, CameraProfileConfig* profile) noexcept {
  if (!profile) return bad(StatusCode::InvalidArgument, "profile output is null");
  try {
    std::istringstream in(text);
    std::string magic, name;
    std::size_t count = 0;
    float iso = 0.0f;
    if (!(in >> magic >> std::quoted(name) >> iso >> count) || magic != "AUGMATCH_CAMERA_PROFILE_V1" || count == 0 || count > 100000)
      return bad(StatusCode::InvalidArgument, "invalid camera profile header");
    CameraProfileConfig parsed;
    parsed.name = name; parsed.iso = iso; parsed.points.resize(count);
    for (auto& p : parsed.points)
      if (!(in >> p.iso >> p.gain >> p.shot_noise_scale >> p.read_noise_scale >> p.fpn_scale >>
            p.black_level >> p.saturation_level >> p.shot_scale >> p.read_noise_stddev >> p.fpn_stddev))
        return bad(StatusCode::InvalidArgument, "truncated camera profile");
    const Status valid = validate_camera_profile(parsed);
    if (!valid) return valid;
    *profile = std::move(parsed);
    return Status::success();
  } catch (...) { return bad(StatusCode::ExecutionError, "profile deserialization failed"); }
}

std::size_t sensor_pipeline_workspace_bytes(const ImageView&, const ImageView&) noexcept { return 0; }
Status validate_workspace(const Workspace& workspace, std::size_t required,
                          MemorySpace execution) noexcept {
  if (workspace.bytes < required || (required != 0 && !workspace.data))
    return bad(StatusCode::InvalidArgument, "workspace is too small");
  if (workspace.bytes != 0 && workspace.memory != execution)
    return bad(StatusCode::InvalidArgument, "workspace memory does not match execution");
  return Status::success();
}

Status fused_sensor_pipeline(const ImageView& input, const MutableImageView& output,
                             const SensorPipelineConfig& config, const Workspace& workspace,
                             cudaStream_t stream) noexcept {
  const Status workspace_status = validate_workspace(workspace, sensor_pipeline_workspace_bytes(input, output.as_const()), input.memory);
  if (!workspace_status) return workspace_status;
  if (!std::isfinite(config.analog_gain) || config.analog_gain < 0.0f)
    return bad(StatusCode::InvalidArgument, "analog gain is invalid");
#if defined(__CUDACC__)
  if (input.memory == MemorySpace::Device && output.memory == MemorySpace::Device) {
    // Metadata is stream-ordered so the caller can enqueue the fused kernel
    // without a host synchronization. Profile points are borrowed; a device
    // profile is intentionally not dereferenced by this host-side setup.
    std::vector<NoiseChannelMetadata> host_metadata(static_cast<std::size_t>(input.channels));
    float profile_gain = 1.0f;
    if (config.profile_memory == MemorySpace::Host && config.profile.points && config.profile.point_count)
      profile_gain = camera_profile_gain(config.iso, config.profile);
    for (auto& metadata : host_metadata) metadata.gain = config.analog_gain * profile_gain;
    NoiseChannelMetadata* device_metadata = nullptr;
    if (cudaMallocAsync(&device_metadata, host_metadata.size() * sizeof(NoiseChannelMetadata), stream) != cudaSuccess)
      return bad(StatusCode::ExecutionError, "CUDA workspace metadata allocation failed");
    if (cudaMemcpyAsync(device_metadata, host_metadata.data(), host_metadata.size() * sizeof(NoiseChannelMetadata),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      cudaFreeAsync(device_metadata, stream);
      return bad(StatusCode::ExecutionError, "CUDA metadata upload failed");
    }
    NoiseMetadataView device_view{device_metadata, host_metadata.size(), nullptr, 0, nullptr, MemorySpace::Device};
    const Status result = additive_noise_view(input, output, config.noise, device_view, stream);
    cudaFreeAsync(device_metadata, stream);
    return result;
  }
#endif
  if (input.memory != MemorySpace::Host || output.memory != MemorySpace::Host)
    return bad(StatusCode::Unsupported, "device fused pipelines require CUDA");
  try {
    std::vector<NoiseChannelMetadata> channel_metadata(static_cast<std::size_t>(input.channels));
    const float profile_gain = config.profile.points && config.profile.point_count
        ? camera_profile_gain(config.iso, config.profile) : 1.0f;
    for (auto& metadata : channel_metadata) metadata.gain = config.analog_gain * profile_gain;
    NoiseMetadataView metadata{channel_metadata.data(), channel_metadata.size(), nullptr, 0, nullptr, MemorySpace::Host};
    return additive_noise_view(input, output, config.noise, metadata, stream);
  } catch (...) { return bad(StatusCode::ExecutionError, "fused sensor pipeline allocation failed"); }
}

Status additive_noise_batch(const ImageView* inputs, const MutableImageView* outputs,
                            std::size_t count, const NoiseViewConfig& config,
                            const NoiseMetadataView& metadata, cudaStream_t stream) noexcept {
  if (count && (!inputs || !outputs)) return bad(StatusCode::InvalidArgument, "batch arrays are null");
  for (std::size_t index = 0; index < count; ++index) {
    const Status result = additive_noise_view(inputs[index], outputs[index], config, metadata, stream);
    if (!result) return result;
  }
  return Status::success();
}

std::future<Status> additive_noise_async(const ImageView& input, const MutableImageView& output,
                                         const NoiseViewConfig& config,
                                         const NoiseMetadataView& metadata) {
  if (input.memory != MemorySpace::Host || output.memory != MemorySpace::Host)
    return std::async(std::launch::deferred, [] { return Status{StatusCode::Unsupported, "async API is host-only"}; });
  return std::async(std::launch::async, [input, output, config, metadata] {
    return additive_noise_view(input, output, config, metadata, nullptr);
  });
}

}  // namespace augmatch
