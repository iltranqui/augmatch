#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <vector>
#include "augmatch/sensor/bayer.hpp"
#include "augmatch/sensor/iso_profile.hpp"
#include "augmatch/sensor/noise_view.hpp"

namespace augmatch {

// Non-owning raw Bayer view. Samples are one plane in row-major sensor order;
// the CFA pattern describes the color assigned to each 2x2 position.
struct RawBayerView {
  const void* data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t stride_bytes = 0;
  DataType type = DataType::UInt16;
  BayerPattern pattern = BayerPattern::RGGB;
  MemorySpace memory = MemorySpace::Host;
  bool valid() const noexcept;
};
struct MutableRawBayerView {
  void* data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t stride_bytes = 0;
  DataType type = DataType::UInt16;
  BayerPattern pattern = BayerPattern::RGGB;
  MemorySpace memory = MemorySpace::Host;
  bool valid() const noexcept;
};
RawBayerView make_raw_bayer_view(const void* data, int width, int height,
                                 std::ptrdiff_t stride_bytes, DataType type = DataType::UInt16,
                                 BayerPattern pattern = BayerPattern::RGGB,
                                 MemorySpace memory = MemorySpace::Host) noexcept;
MutableRawBayerView make_raw_bayer_view(void* data, int width, int height,
                                        std::ptrdiff_t stride_bytes, DataType type = DataType::UInt16,
                                        BayerPattern pattern = BayerPattern::RGGB,
                                        MemorySpace memory = MemorySpace::Host) noexcept;
// Validates a RawBayerView/MutableRawBayerView pair's shape, stride, dtype,
// and memory-space agreement before a raw Bayer operation.
Status validate_raw_bayer_view(const RawBayerView& view, const MutableRawBayerView& mutable_view) noexcept;

// A value-owning profile is convenient for configuration files and examples.
// Its view() can be passed to device code only after the caller copies points
// to device memory; borrowed views remain valid for the duration of a stream.
struct CameraProfileConfig {
  std::string name;
  float iso = 100.0f;
  std::vector<CameraProfileLookupPoint> points;
  CameraProfileLookupTable view() const noexcept {
    return {points.data(), points.size()};
  }
};
// Validates that profile's points are sorted by increasing ISO and finite.
Status validate_camera_profile(const CameraProfileConfig& profile) noexcept;
// Serializes profile as versioned AUGMATCH_CAMERA_PROFILE_V1 text into *text.
Status serialize_camera_profile(const CameraProfileConfig& profile, std::string* text) noexcept;
// Parses AUGMATCH_CAMERA_PROFILE_V1 text into *profile.
Status deserialize_camera_profile(const std::string& text, CameraProfileConfig* profile) noexcept;

// Caller-owned temporary storage. A zero-byte workspace is valid for the
// current fused operation. This explicit contract lets applications use pools.
struct Workspace {
  void* data = nullptr;
  std::size_t bytes = 0;
  MemorySpace memory = MemorySpace::Host;
};
// Returns the workspace byte count fused_sensor_pipeline requires for the
// given input/output view shapes (currently zero).
std::size_t sensor_pipeline_workspace_bytes(const ImageView& input, const ImageView& output) noexcept;
// Validates that workspace has at least required bytes in the given
// execution memory space.
Status validate_workspace(const Workspace& workspace, std::size_t required, MemorySpace execution) noexcept;

struct SensorPipelineConfig {
  NoiseViewConfig noise{};
  float iso = 100.0f;
  float analog_gain = 1.0f;
  CameraProfileLookupTable profile{};
  MemorySpace profile_memory = MemorySpace::Host;
};
// Applies the fused sensor pipeline (noise, ISO/analog gain, profile lookup)
// to input, writing to output. Uses workspace scratch memory (zero bytes for
// the current operation) and enqueues CUDA work on stream.
Status fused_sensor_pipeline(const ImageView& input, const MutableImageView& output,
                             const SensorPipelineConfig& config, const Workspace& workspace = {},
                             cudaStream_t stream = nullptr) noexcept;

// All arrays are borrowed and must contain count entries. Each pair is
// independent and may be strided; CUDA launches are ordered on stream.
Status additive_noise_batch(const ImageView* inputs, const MutableImageView* outputs,
                            std::size_t count, const NoiseViewConfig& config,
                            const NoiseMetadataView& metadata = {},
                            cudaStream_t stream = nullptr) noexcept;

// Host-only asynchronous wrapper. It captures non-owning views and metadata;
// callers must keep their storage alive until the future is ready. Device views
// are rejected before scheduling because callbacks/futures cannot own CUDA data.
std::future<Status> additive_noise_async(const ImageView& input, const MutableImageView& output,
                                         const NoiseViewConfig& config,
                                         const NoiseMetadataView& metadata = {});

namespace sensor {
using ::augmatch::RawBayerView;
using ::augmatch::MutableRawBayerView;
using ::augmatch::CameraProfileConfig;
using ::augmatch::SensorPipelineConfig;
using ::augmatch::Workspace;
inline Status fused_pipeline(const ImageView& i, const MutableImageView& o,
                             const SensorPipelineConfig& c, const Workspace& w = {},
                             cudaStream_t s = nullptr) noexcept {
  return fused_sensor_pipeline(i, o, c, w, s);
}
}
namespace optics {
using ::augmatch::NoiseBorderPolicy;
using ::augmatch::resolve_noise_coordinate;
}
namespace isp {
using ::augmatch::NoiseClipPolicy;
using ::augmatch::NoiseQuantizationPolicy;
using ::augmatch::Workspace;
}
namespace compression {
using ::augmatch::Status;
using ::augmatch::StatusCode;
}
namespace video {
using ::augmatch::additive_noise_batch;
using ::augmatch::additive_noise_async;
}

}  // namespace augmatch
