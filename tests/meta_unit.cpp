#include "augmatch/meta.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {
void add_one(augmatch::MutableImageView image, augmatch::MutableTargetAnnotations targets,
             const augmatch::ExecutionContext&, void*) {
  auto* pixels = static_cast<std::uint8_t*>(image.data);
  pixels[0] = static_cast<std::uint8_t>(pixels[0] + 1);
  if (targets.box_count) targets.boxes[0].x1 += 1.0f;
}
bool pixel_and_target(augmatch::ImageView image, augmatch::TargetAnnotations targets,
                      const augmatch::ExecutionContext&, void*) {
  const auto* pixels = static_cast<const std::uint8_t*>(image.data);
  return pixels[0] == 8 && targets.box_count == 1 && targets.boxes[0].x1 == 1.0f;
}
}  // namespace

int main() {
  using namespace augmatch;
  std::uint8_t pixels[2 * 2 * 1] = {7, 0, 0, 0};
  BoxXYXY boxes[] = {{0, 0, 1, 1}};
  PointXY points[] = {{0, 0}};
  auto image = make_hwc_u8_view(pixels, 2, 2, 1);
  MutableTargetAnnotations targets{boxes, 1, points, 1};

  lambda(image, LambdaConfig{add_one, nullptr}, targets);
  if (pixels[0] != 8 || boxes[0].x1 != 1.0f) return 1;
  assert_lambda(image, AssertLambdaConfig{pixel_and_target, nullptr, nullptr},
                TargetAnnotations{boxes, 1, points, 1});
  assert_shape(image, AssertShapeConfig{2, 2, 1});

  bool rejected = false;
  try {
    assert_shape(image, AssertShapeConfig{3, -1, -1});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 2;

  BoxXYXY filtered[] = {{-1, 0, 1, 2}, {0, 0, 1, 1}, {5, 0, 6, 1}};
  PointXY filtered_points[] = {{0, 0}, {-1, 0}, {2, 1}};
  std::size_t box_indices[3]{}, point_indices[3]{};
  const auto result = remove_cbas_by_out_of_image_fraction(
      {filtered, 3, filtered_points, 3}, 2, 2, 0.5f, box_indices, point_indices);
  if (result.box_count != 2 || result.keypoint_count != 1 || box_indices[0] != 0 ||
      box_indices[1] != 1 || point_indices[0] != 0)
    return 3;

  BoxXYXY clipped[] = {{-2, -1, 4, 3}};
  PointXY clipped_points[] = {{-1, 4}};
  clip_cbas_to_image_planes({clipped, 1, clipped_points, 1}, 3, 2);
  if (clipped[0].x1 != 0 || clipped[0].y1 != 0 || clipped[0].x2 != 3 || clipped[0].y2 != 2 ||
      clipped_points[0].x != 0 || clipped_points[0].y != 1)
    return 4;

  bool callback_rejected = false;
  try {
    lambda(image, LambdaConfig{}, {});
  } catch (const std::invalid_argument&) {
    callback_rejected = true;
  }
  if (!callback_rejected) return 5;
  bool device_rejected = false;
  try {
    lambda(make_hwc_u8_view(pixels, 2, 2, 1, MemorySpace::Device),
           LambdaConfig{add_one, nullptr}, {});
  } catch (const std::invalid_argument&) {
    device_rejected = true;
  }
  if (!device_rejected) return 6;
  return 0;
}
