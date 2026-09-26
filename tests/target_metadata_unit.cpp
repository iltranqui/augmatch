#include "augmatch/augmatch.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

int main() {
  using namespace augmatch;
  const Geometry g{4, 3, 1, 1, 0, 3, 3, true, false};

  std::uint8_t source[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::uint8_t destination[9]{};
  AdditionalImageTarget extra{make_hwc_u8_view(static_cast<const std::uint8_t*>(source), 4, 3, 1),
                               make_hwc_u8_view(destination, 3, 3, 1)};
  transform_additional_images(&extra, 1, g);
  if (destination[0] != 3 || destination[1] != 2 || destination[2] != 1 || destination[3] != 7)
    return 1;
  std::uint8_t mask_destination[9]{};
  AdditionalMaskTarget mask{make_hwc_u8_view(static_cast<const std::uint8_t*>(source), 4, 3, 1),
                            make_hwc_u8_view(mask_destination, 3, 3, 1)};
  transform_additional_masks(&mask, 1, g);
  if (mask_destination[0] != 3 || mask_destination[8] != 9) return 2;

  const COCOBox coco{1, 2, 3, 4};
  const BoxXYXY pascal = coco_to_pascal(coco, 10, 10);
  if (!near(pascal.x2, 4) || !near(pascal.y2, 6)) return 3;
  const YOLOBox yolo = pascal_to_yolo(pascal, 10, 10);
  if (!near(yolo.center_x, .25f) || !near(yolo.height, .4f)) return 4;
  const AlbumentationsBox normalized = pascal_to_albumentations(pascal, 10, 10);
  if (!near(normalized.x1, .1f) || !near(normalized.y2, .6f)) return 5;

  COCOBox transformed_coco[1]{};
  transform_coco_boxes(&coco, transformed_coco, 1, 10, 10,
                       Geometry{10, 10, 1, 0, 0, 10, 10, true, false});
  if (!near(transformed_coco[0].x, 6) || !near(transformed_coco[0].width, 3)) return 6;

  KeypointXYV keypoint{1, 1, 1};
  KeypointXYV transformed_keypoint{};
  transform_keypoints_with_visibility(&keypoint, &transformed_keypoint, 1,
                                      Geometry{4, 3, 1, 0, 0, 4, 3, true, false});
  if (!near(transformed_keypoint.x, 2) || !near(transformed_keypoint.visibility, 1)) return 7;
  keypoint = {5, 1, 1};
  transform_keypoints_with_visibility(&keypoint, &transformed_keypoint, 1,
                                      Geometry{4, 3, 1, 0, 0, 4, 3, false, false});
  if (!near(transformed_keypoint.visibility, 0)) return 8;

  bool rejected = false;
  try { validate_yolo_boxes(&yolo, 1, 10, 10); } catch (...) { rejected = true; }
  if (rejected) return 9;
  const YOLOBox invalid{.5f, .5f, 1.2f, .5f};
  try { validate_yolo_boxes(&invalid, 1, 10, 10); } catch (const std::invalid_argument&) { return 0; }
  return 10;
}
