#include "augmatch/composition.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {
struct DeviceStageState {
  int calls = 0;
  int channels = 0;
  const void* data = nullptr;
  cudaStream_t stream = nullptr;
};

void observe_device_stage(augmatch::MutableImageView image, cudaStream_t stream,
                          const augmatch::ExecutionContext& execution, const void* opaque) {
  auto& state = *static_cast<DeviceStageState*>(const_cast<void*>(opaque));
  if (image.memory != augmatch::MemorySpace::Device || stream != execution.stream) state.calls = -99;
  ++state.calls;
  state.channels = image.channels;
  state.data = image.data;
  state.stream = stream;
}
}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  constexpr int width = 4;
  constexpr int height = 1;
  constexpr int channels = 1;
  const std::uint8_t expected[] = {3, 5, 7, 9};
  std::uint8_t* device_data = nullptr;
  if (cudaMalloc(&device_data, sizeof(expected)) != cudaSuccess) return 1;
  if (cudaMemcpy(device_data, expected, sizeof(expected), cudaMemcpyHostToDevice) != cudaSuccess)
    return 2;

  const auto view = augmatch::make_hwc_u8_view(device_data, width, height, channels,
                                                augmatch::MemorySpace::Device);
  DeviceStageState stage_state;
  const auto pipeline = augmatch::make_compose(
      std::vector<augmatch::DeviceStage>{{nullptr, nullptr},
                                         {observe_device_stage, &stage_state},
                                         {nullptr, nullptr}});
  augmatch::ExecutionContext execution;
  augmatch::compose(view, pipeline, execution);
  if (stage_state.calls != 1 || stage_state.stream != execution.stream) return 6;
  augmatch::ReplayRecord record;
  augmatch::ExecutionContext recording{execution.stream, 44, true, &record, nullptr};
  augmatch::one_of(view, std::vector<augmatch::DevicePipeline>{
                              augmatch::make_compose(std::vector<augmatch::DeviceStage>{}),
                              pipeline},
                   recording);
  if (record.selections.size() != 1) return 7;
  augmatch::ExecutionContext replaying{execution.stream, 999, true, nullptr, &record};
  augmatch::one_of(view, std::vector<augmatch::DevicePipeline>{
                              augmatch::make_compose(std::vector<augmatch::DeviceStage>{}),
                              pipeline},
                   replaying);
  const int expected_calls = 1 + (record.selections[0].selected == 1 ? 2 : 0);
  if (stage_state.calls != expected_calls) return 8;
  augmatch::sequential(view, augmatch::make_sequential(std::vector<augmatch::DeviceStage>{}),
                       execution);
  DeviceStageState range_state;
  augmatch::selective_channel_transform(
      view, augmatch::make_compose(std::vector<augmatch::DeviceStage>{{observe_device_stage,
                                                                       &range_state}}),
      augmatch::ChannelRange{0, 1}, execution);
  if (range_state.calls != 1 || range_state.channels != 1 || range_state.data != device_data)
    return 9;
  augmatch::identity(view, execution);
  augmatch::noop(view, execution);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;

  std::vector<std::uint8_t> result(sizeof(expected));
  if (cudaMemcpy(result.data(), device_data, sizeof(expected), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 4;
  cudaFree(device_data);
  for (std::size_t i = 0; i < result.size(); ++i)
    if (result[i] != expected[i]) return 5;
  return 0;
}
