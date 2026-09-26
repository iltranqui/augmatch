#include <cassert>
#include <cstdint>
#include <future>
#include <string>
#include <vector>
#include "augmatch/augmatch.hpp"

int main() {
  using namespace augmatch;
  std::vector<std::uint8_t> source(12, 127), result(12), batch_result(12);
  auto input_mutable = make_hwc_view(source.data(), 2, 2, 3);
  auto input = input_mutable.as_const();
  auto output = make_hwc_view(result.data(), 2, 2, 3);
  NoiseViewConfig noise; noise.seed = 42; noise.stddev = 0.0f;
  assert(fused_sensor_pipeline(input, output, {noise}).ok());
  assert(result == source);

  ImageView inputs[] = {input};
  MutableImageView outputs[] = {make_hwc_view(batch_result.data(), 2, 2, 3)};
  assert(additive_noise_batch(inputs, outputs, 1, noise).ok());
  assert(batch_result == source);
  auto future = additive_noise_async(input, output, noise);
  assert(future.get().ok());

  std::vector<std::uint16_t> raw(4), raw_out(4);
  const auto raw_in = make_raw_bayer_view(static_cast<const void*>(raw.data()), 2, 2, 4);
  auto raw_view_out = make_raw_bayer_view(raw_out.data(), 2, 2, 4);
  assert(validate_raw_bayer_view(raw_in, raw_view_out).ok());
  assert(sensor_pipeline_workspace_bytes(input, output.as_const()) == 0);
  assert(validate_workspace({}, 0, MemorySpace::Host).ok());

  CameraProfileConfig profile; profile.name = "unit";
  profile.points.push_back({100.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 100.0f, 0.01f, 0.0f});
  std::string encoded;
  assert(serialize_camera_profile(profile, &encoded).ok());
  CameraProfileConfig decoded;
  assert(deserialize_camera_profile(encoded, &decoded).ok());
  assert(decoded.name == profile.name && decoded.points.size() == 1);
  return 0;
}
