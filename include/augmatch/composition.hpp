#pragma once

#include <cstdint>
#include <vector>

#include "augmatch/execution.hpp"
#include "augmatch/image.hpp"

namespace augmatch {

// CPU callbacks are synchronous, in-place stages. The callback and context are
// borrowed: the caller owns both and keeps them alive until the pipeline call
// returns. A callback must not retain context or the image view after return.
using CpuStageCallback = void (*)(MutableImageView image,
                                  const ExecutionContext& execution,
                                  void* context);

// A stage with a non-default range receives a view whose channel zero is
// `first`; strides still describe the original image. A count of -1 means all
// channels. This contract works for both HWC and CHW views without a copy.
struct ChannelRange {
  int first = 0;
  int count = -1;
};

// A channel selection is the union of explicit channel indices and contiguous
// ranges. Empty selections mean all channels when used as a stage default.
// Ranges are normalized before callbacks run, so overlapping entries do not
// apply a stage more than once. HWC callbacks receive one view per contiguous
// run; CHW callbacks can receive a multi-channel run without a copy.
struct ChannelSelection {
  std::vector<int> indices;
  std::vector<ChannelRange> ranges;
};

struct CpuStage {
  CpuStageCallback callback = nullptr;
  void* context = nullptr;
  ChannelRange channel_range{};
  ChannelSelection channel_selection{};
};

struct CpuPipeline {
  std::vector<CpuStage> stages;
};

// Device stages are deliberately not CpuStage callbacks. A launch callback is
// a host function that enqueues work for the supplied CUDA stream; it is not a
// __device__ function and must not dereference device image memory on the host.
// Its context is borrowed exactly like CpuStage::context. Identity and Noop
// use a null launch callback and are handled by the pipeline itself.
using DeviceStageLaunch = void (*)(MutableImageView image,
                                   cudaStream_t stream,
                                   const ExecutionContext& execution,
                                   const void* context);

struct DeviceStage {
  DeviceStageLaunch launch = nullptr;
  const void* context = nullptr;
  ChannelRange channel_range{};
  ChannelSelection channel_selection{};
};

struct DevicePipeline {
  std::vector<DeviceStage> stages;
};

// These constructors preserve stage order. Compose and Sequential currently
// have identical deterministic semantics; separate names mirror the catalog
// and leave room for future control policies without changing callers.
CpuPipeline make_compose(std::vector<CpuStage> stages);
CpuPipeline make_sequential(std::vector<CpuStage> stages);
DevicePipeline make_compose(std::vector<DeviceStage> stages);
DevicePipeline make_sequential(std::vector<DeviceStage> stages);

// Apply an in-place CPU pipeline. The image and callback contexts remain owned
// by the caller. The execution context is read-only and no global RNG state is
// consulted; repeated calls with deterministic callbacks are reproducible.
void compose(MutableImageView image, const CpuPipeline& pipeline,
             const ExecutionContext& execution = {});
void sequential(MutableImageView image, const CpuPipeline& pipeline,
                const ExecutionContext& execution = {});

// Apply an explicit device pipeline. Input must be a valid device view. A
// DeviceStage launch callback runs on the host and is responsible for enqueuing
// its own device work on execution.stream; callers synchronize that stream.
void compose(MutableImageView image, const DevicePipeline& pipeline,
             const ExecutionContext& execution = {});
void sequential(MutableImageView image, const DevicePipeline& pipeline,
                const ExecutionContext& execution = {});

// Identity and Noop are synchronous validation-preserving operations. They do
// not modify image bytes, consume seed state, or synchronize a CUDA stream.
void identity(MutableImageView image, const ExecutionContext& execution = {});
void noop(MutableImageView image, const ExecutionContext& execution = {});

// ReplayCompose is an explicit top-level boundary for a recorded composition.
// It preserves the pipeline's order while allowing nested control nodes to
// consume the supplied replay entries.
void replay_compose(MutableImageView image, const CpuPipeline& pipeline,
                    const ExecutionContext& execution = {});
void replay_compose(MutableImageView image, const DevicePipeline& pipeline,
                    const ExecutionContext& execution = {});

// Control nodes select complete child pipelines. Selection is derived only from
// ExecutionContext::seed, or read from/write to its explicit ReplayRecord.
void one_of(MutableImageView image, const std::vector<CpuPipeline>& choices,
            const ExecutionContext& execution = {});
void one_of(MutableImageView image, const std::vector<DevicePipeline>& choices,
            const ExecutionContext& execution = {});
void one_or_other(MutableImageView image, const CpuPipeline& first,
                  const CpuPipeline& second, const ExecutionContext& execution = {});
void one_or_other(MutableImageView image, const DevicePipeline& first,
                  const DevicePipeline& second, const ExecutionContext& execution = {});
void random_apply(MutableImageView image, const CpuPipeline& pipeline, double probability,
                  const ExecutionContext& execution = {});
void random_apply(MutableImageView image, const DevicePipeline& pipeline, double probability,
                  const ExecutionContext& execution = {});

// SomeOf applies exactly `count` children. Without replacement is the default;
// replacement can be enabled explicitly. The replay record stores the chosen
// child indices in their application order.
void some_of(MutableImageView image, const std::vector<CpuPipeline>& choices,
             std::size_t count, const ExecutionContext& execution = {},
             bool replace = false);
void some_of(MutableImageView image, const std::vector<CpuPipeline>& choices,
             std::size_t count, bool replace, const ExecutionContext& execution = {});
void some_of(MutableImageView image, const std::vector<DevicePipeline>& choices,
             std::size_t count, const ExecutionContext& execution = {},
             bool replace = false);
void some_of(MutableImageView image, const std::vector<DevicePipeline>& choices,
             std::size_t count, bool replace, const ExecutionContext& execution = {});

// RandomOrder applies every child once using a recorded permutation.
void random_order(MutableImageView image, const std::vector<CpuPipeline>& choices,
                  const ExecutionContext& execution = {});
void random_order(MutableImageView image, const std::vector<DevicePipeline>& choices,
                  const ExecutionContext& execution = {});

// SelectiveChannelTransform applies a pipeline to borrowed channel ranges or
// explicit channel indices. Stages receive restricted views through the same
// CPU/CUDA stage contract; no channel copy is made.
void selective_channel_transform(MutableImageView image, const CpuPipeline& pipeline,
                                 ChannelSelection channels,
                                 const ExecutionContext& execution = {});
void selective_channel_transform(MutableImageView image, const DevicePipeline& pipeline,
                                 ChannelSelection channels,
                                 const ExecutionContext& execution = {});
inline void selective_channel_transform(MutableImageView image, const CpuPipeline& pipeline,
                                       ChannelRange channels,
                                       const ExecutionContext& execution = {}) {
  selective_channel_transform(image, pipeline, ChannelSelection{{}, {channels}}, execution);
}
inline void selective_channel_transform(MutableImageView image, const DevicePipeline& pipeline,
                                       ChannelRange channels,
                                       const ExecutionContext& execution = {}) {
  selective_channel_transform(image, pipeline, ChannelSelection{{}, {channels}}, execution);
}
inline void selective_channel_transform(MutableImageView image, const CpuPipeline& pipeline,
                                       int first, int count,
                                       const ExecutionContext& execution = {}) {
  selective_channel_transform(image, pipeline, ChannelRange{first, count}, execution);
}
inline void selective_channel_transform(MutableImageView image, const DevicePipeline& pipeline,
                                       int first, int count,
                                       const ExecutionContext& execution = {}) {
  selective_channel_transform(image, pipeline, ChannelRange{first, count}, execution);
}

// WithChannels is the imgaug meta operation. It applies the child pipeline to
// only the listed channels/ranges and leaves every other channel untouched.
// An empty index list is a no-op, matching an empty imgaug channel selection.
void with_channels(MutableImageView image, const CpuPipeline& pipeline,
                   const std::vector<int>& channels,
                   const ExecutionContext& execution = {});
void with_channels(MutableImageView image, const DevicePipeline& pipeline,
                   const std::vector<int>& channels,
                   const ExecutionContext& execution = {});
void with_channels(MutableImageView image, const CpuPipeline& pipeline,
                   ChannelRange channels,
                   const ExecutionContext& execution = {});
void with_channels(MutableImageView image, const DevicePipeline& pipeline,
                   ChannelRange channels,
                   const ExecutionContext& execution = {});
void with_channels(MutableImageView image, const CpuPipeline& pipeline,
                   const ChannelSelection& channels,
                   const ExecutionContext& execution = {});
void with_channels(MutableImageView image, const DevicePipeline& pipeline,
                   const ChannelSelection& channels,
                   const ExecutionContext& execution = {});

}  // namespace augmatch
