#include "augmatch/pipeline.hpp"
#include "augmatch/pipeline_registry.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool close_u8(std::uint8_t actual, int expected, int tolerance = 1) {
  return std::abs(static_cast<int>(actual) - expected) <= tolerance;
}

void add_one(augmatch::MutableImageView image, const augmatch::ExecutionContext&, void*) {
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  const std::size_t count = static_cast<std::size_t>(image.width) * image.height * image.channels;
  for (std::size_t index = 0; index < count; ++index)
    bytes[index] = static_cast<std::uint8_t>(bytes[index] + 1U);
}

}  // namespace

int main() {
  const std::string config = R"cfg(
# The parser preserves this order.
[HueShift]
amount = 0.0

[Saturation]
multiplier = 2.0

[GaussianBlur]
kernel_size = 3
sigma = 0.0
probability = 0.25

[HorizontalFlip]
probability = 1.0
)cfg";

  auto parsed = augmatch::Pipeline::load_from_string(config);
  if (!parsed.ok()) return 1;
  const auto& stages = parsed.value().stages();
  if (stages.size() != 4 || stages[0].name != "HueShift" ||
      stages[1].name != "Saturation" || stages[2].name != "GaussianBlur" ||
      stages[3].name != "HorizontalFlip")
    return 2;
  if (stages[0].params.at("amount") != 0.0f ||
      stages[1].params.at("multiplier") != 2.0f ||
      stages[2].params.at("kernel_size") != 3.0f ||
      stages[2].params.at("sigma") != 0.0f ||
      stages[2].probability != 0.25f || stages[3].probability != 1.0f)
    return 3;

  auto saturation = augmatch::Pipeline::load_from_string(
      "[Saturation]\nmultiplier = 2\n");
  if (!saturation.ok()) return 4;
  std::vector<std::uint8_t> color{100, 80, 80};
  augmatch::ExecutionContext execution;
  execution.seed = 7;
  const auto saturation_status = saturation.value().run(
      augmatch::make_hwc_u8_view(color.data(), 1, 1, 3), execution);
  if (!saturation_status.ok() || !close_u8(color[0], 100) || !close_u8(color[1], 60) ||
      !close_u8(color[2], 60))
    return 5;

  auto flip = augmatch::Pipeline::load_from_string("[HorizontalFlip]\n");
  if (!flip.ok()) return 6;
  std::vector<std::uint8_t> row{1, 2, 3, 10, 20, 30, 100, 110, 120};
  const auto flip_status = flip.value().run(
      augmatch::make_hwc_u8_view(row.data(), 3, 1, 3), execution);
  const std::vector<std::uint8_t> expected_flip{100, 110, 120, 10, 20, 30, 1, 2, 3};
  if (!flip_status.ok() || row != expected_flip) return 7;

  auto never_flip = augmatch::Pipeline::load_from_string(
      "[HorizontalFlip]\nprobability = 0\n");
  if (!never_flip.ok()) return 8;
  std::vector<std::uint8_t> unchanged{1, 2, 3, 4, 5, 6};
  const auto unchanged_before = unchanged;
  if (!never_flip.value()
           .run(augmatch::make_hwc_u8_view(unchanged.data(), 2, 1, 3), execution)
           .ok() ||
      unchanged != unchanged_before)
    return 9;

  const auto registration = augmatch::register_stage(
      "AddOne", [](const augmatch::StageParameters& params) {
        if (!params.empty())
          return augmatch::Result<augmatch::RegisteredCpuStage>::failure(
              {augmatch::StatusCode::InvalidArgument, "AddOne takes no parameters"});
        augmatch::RegisteredCpuStage registered;
        registered.stage.callback = add_one;
        return augmatch::Result<augmatch::RegisteredCpuStage>::success(
            std::move(registered));
      });
  if (!registration.ok()) return 10;
  auto custom = augmatch::Pipeline::load_from_string("[AddOne]\n");
  if (!custom.ok()) return 11;
  std::vector<std::uint8_t> custom_bytes{1, 2, 3};
  if (!custom.value()
           .run(augmatch::make_hwc_u8_view(custom_bytes.data(), 1, 1, 3), execution)
           .ok() ||
      custom_bytes != std::vector<std::uint8_t>({2, 3, 4}))
    return 12;

  auto bad_probability = augmatch::Pipeline::load_from_string(
      "[HorizontalFlip]\nprobability = 1.5\n");
  auto bad_number = augmatch::Pipeline::load_from_string(
      "[Exposure]\nmultiplier = bright\n");
  auto unknown_stage = augmatch::Pipeline::load_from_string("[DoesNotExist]\n");
  auto unknown_parameter = augmatch::Pipeline::load_from_string(
      "[HorizontalFlip]\naxis = 1\n");
  auto duplicate_probability = augmatch::Pipeline::load_from_string(
      "[HorizontalFlip]\nprobability = 1\nprobability = 0\n");
  auto huge_exposure = augmatch::Pipeline::load_from_string(
      "[Exposure]\nmultiplier = 1e20\n");
  auto huge_saturation = augmatch::Pipeline::load_from_string(
      "[Saturation]\nmultiplier = 1e20\n");
  auto huge_hue = augmatch::Pipeline::load_from_string(
      "[HueShift]\namount = 1e20\n");
  auto tiny_sigma = augmatch::Pipeline::load_from_string(
      "[GaussianBlur]\nkernel_size = 3\nsigma = 1e-30\n");
  // An embedded NUL must not let strtof silently truncate the value and
  // accept "1.5" out of "1.5\0garbage".
  auto embedded_nul = augmatch::Pipeline::load_from_string(
      std::string("[Exposure]\nmultiplier = 1.5") + '\0' + "garbage\n");
  if (bad_probability.ok() || bad_number.ok() || unknown_stage.ok() ||
      unknown_parameter.ok() || duplicate_probability.ok() || huge_exposure.ok() ||
      huge_saturation.ok() || huge_hue.ok() || tiny_sigma.ok() || embedded_nul.ok())
    return 13;

  auto planned_blur = augmatch::Pipeline::load_from_string(
      "[GaussianBlur]\nkernel_size = 17\nsigma = 0\n");
  if (!planned_blur.ok()) return 18;
  std::vector<std::uint8_t> blur_pixel{20, 40, 80};
  if (!planned_blur.value()
           .run(augmatch::make_hwc_u8_view(blur_pixel.data(), 1, 1, 3), execution)
           .ok() ||
      blur_pixel != std::vector<std::uint8_t>({20, 40, 80}))
    return 19;

  auto exposure = augmatch::Pipeline::load_from_string(
      "[Exposure]\nmultiplier = 1.1\n");
  if (!exposure.ok()) return 14;
  float float_pixel[3] = {0.1f, 0.2f, 0.3f};
  const auto invalid_status = exposure.value().run(
      augmatch::make_hwc_view(float_pixel, 1, 1, 3), execution);
  if (invalid_status.code != augmatch::StatusCode::InvalidView) return 15;

  const std::string file_path = "pipeline_unit.cfg";
  {
    std::ofstream file(file_path);
    if (!file) return 16;
    file << "[Brightness]\nbrightness = 0.1\n";
  }
  auto from_file = augmatch::Pipeline::load_from_file(file_path);
  if (!from_file.ok() || from_file.value().stages().size() != 1 ||
      from_file.value().stages()[0].name != "Brightness")
    return 17;

  return 0;
}
