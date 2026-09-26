// 09_boxes_keypoints — keep bounding boxes and keypoints aligned with a crop + flip.
//
// What it shows:
//   * Annotations are transformed with the same Geometry that transforms the pixels.
//   * Box formats: augmatch works in Pascal VOC (x1,y1,x2,y2); helpers convert COCO/YOLO.
//   * transform_boxes_filtered clips boxes to the new image and drops mostly-cut ones.
// API used: Geometry, transform_point, transform_boxes_filtered, BoxTransformOptions,
//           coco_to_pascal, pascal_to_yolo. (All annotation helpers run on the host.)
// Output: text only — before/after coordinates.

#include <iostream>

#include "augmatch/augmatch.hpp"

int main() {
  // Crop a 40x30 window at (10, 5) out of a 64x48 image, then mirror it horizontally.
  augmatch::Geometry geometry{};
  geometry.input_width = 64;
  geometry.input_height = 48;
  geometry.channels = 3;
  geometry.crop_x = 10;             // top-left corner of the crop
  geometry.crop_y = 5;
  geometry.output_width = 40;       // crop size = output image size
  geometry.output_height = 30;
  geometry.flip_horizontal = true;  // applied after the crop
  geometry.flip_vertical = false;

  // A keypoint (e.g. an eye) at pixel (20, 10) in the original image.
  const augmatch::PointXY eye{20.0f, 10.0f};
  const augmatch::PointXY eye_after = augmatch::transform_point(eye, geometry);
  std::cout << "keypoint (20,10) -> (" << eye_after.x << "," << eye_after.y << ")\n";

  // Two boxes, given in COCO format (x, y, width, height) and converted to Pascal VOC.
  const augmatch::BoxXYXY boxes[2] = {
      augmatch::coco_to_pascal({15.0f, 10.0f, 10.0f, 10.0f}, 64, 48),  // fully inside the crop
      augmatch::coco_to_pascal({45.0f, 30.0f, 15.0f, 15.0f}, 64, 48),  // mostly outside the crop
  };

  augmatch::BoxTransformOptions options;
  options.clip = true;              // cut boxes at the new image border
  options.min_visibility = 0.5f;    // drop boxes that keep less than 50% of their area

  augmatch::BoxXYXY kept[2];
  std::size_t source_index[2];      // which input box each kept box came from (to filter labels)
  const std::size_t count =
      augmatch::transform_boxes_filtered(boxes, kept, 2, geometry, options, source_index);

  std::cout << "kept " << count << " of 2 boxes\n";
  for (std::size_t i = 0; i < count; ++i) {
    const augmatch::YOLOBox yolo = augmatch::pascal_to_yolo(kept[i], geometry.output_width, geometry.output_height);
    std::cout << "  box " << source_index[i] << ": xyxy=(" << kept[i].x1 << "," << kept[i].y1 << ","
              << kept[i].x2 << "," << kept[i].y2 << ")  yolo centre=(" << yolo.center_x << ","
              << yolo.center_y << ")\n";
  }
  return count == 1 ? 0 : 1;        // the second box should have been dropped
}
