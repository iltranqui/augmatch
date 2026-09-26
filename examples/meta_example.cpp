// meta_example — a custom callback stage that edits pixels *and* annotations.
//
// What it shows: `lambda` runs your own function as an augmentation stage,
// receiving the image view and the mutable boxes/keypoints. Afterwards,
// clip_cbas_to_image_planes clips boxes and keypoints to the image bounds.
// API used: MutableTargetAnnotations, lambda, clip_cbas_to_image_planes.
// Output: prints the changed pixel, the clipped box x-range, and the keypoint x.
#include "augmatch/augmatch.hpp"

#include <cstdint>
#include <iostream>

// Callback signature: (image, annotations, execution context, user pointer).
static void brighten(augmatch::MutableImageView image,
                     augmatch::MutableTargetAnnotations targets,
                     const augmatch::ExecutionContext&, void*) {
  auto* pixel = static_cast<std::uint8_t*>(image.data);
  pixel[0] = static_cast<std::uint8_t>(pixel[0] + 1);        // brighten the first pixel by 1
  if (targets.box_count) targets.boxes[0].x1 += 1.0f;        // and nudge the first box
}

int main() {
  std::uint8_t pixels[4] = {0, 0, 0, 0};                     // 2x2 grayscale image
  augmatch::BoxXYXY boxes[] = {{-1, 0, 2, 2}};               // x1=-1 starts outside the image
  augmatch::PointXY points[] = {{3, 1}};                     // x=3 is outside a 2-wide image
  auto image = augmatch::make_hwc_u8_view(pixels, 2, 2, 1);
  augmatch::MutableTargetAnnotations targets{boxes, 1, points, 1};  // borrowed arrays + counts
  augmatch::lambda(image, {brighten, nullptr}, targets);     // {callback, user pointer}
  augmatch::clip_cbas_to_image_planes(targets, image.width, image.height);
  std::cout << "pixel=" << static_cast<int>(pixels[0]) << " box=" << boxes[0].x1
            << "," << boxes[0].x2 << " keypoint=" << points[0].x << "\n";
}
