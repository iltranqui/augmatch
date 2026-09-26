#include "augmatch/composition.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {
struct State {
  int calls = 0;
  int channels = -1;
  std::vector<const void*> data;
};

void observe(augmatch::MutableImageView image, cudaStream_t stream,
             const augmatch::ExecutionContext& execution, const void* opaque) {
  auto& state = *static_cast<State*>(const_cast<void*>(opaque));
  if (stream != execution.stream || image.memory != augmatch::MemorySpace::Device) state.calls = -99;
  ++state.calls;
  state.channels = image.channels;
  state.data.push_back(image.data);
}
}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  std::uint8_t* device_data = nullptr;
  if (cudaMalloc(&device_data, 8) != cudaSuccess) return 1;
  const auto view = augmatch::make_hwc_u8_view(device_data, 2, 1, 4,
                                                augmatch::MemorySpace::Device);
  State state;
  const auto pipeline = augmatch::make_compose(
      std::vector<augmatch::DeviceStage>{{observe, &state}});
  augmatch::with_channels(view, pipeline, std::vector<int>{0, 2});
  if (state.calls != 2 || state.channels != 1 || state.data.size() != 2 ||
      state.data[0] != device_data ||
      state.data[1] != device_data + 2)
    return 2;
  cudaFree(device_data);
  return 0;
}
