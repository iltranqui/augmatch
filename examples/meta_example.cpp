#include "augmatch/augmatch.hpp"

#include <cstdint>
#include <iostream>

static void brighten(augmatch::MutableImageView image,
                     augmatch::MutableTargetAnnotations targets,
                     const augmatch::ExecutionContext&, void*) {
  auto* pixel = static_cast<std::uint8_t*>(image.data);
  pixel[0] = static_cast<std::uint8_t>(pixel[0] + 1);
  if (targets.box_count) targets.boxes[0].x1 += 1.0f;
}

int main() {
  std::uint8_t pixels[4] = {0, 0, 0, 0};
  augmatch::BoxXYXY boxes[] = {{-1, 0, 2, 2}};
  augmatch::PointXY points[] = {{3, 1}};
  auto image = augmatch::make_hwc_u8_view(pixels, 2, 2, 1);
  augmatch::MutableTargetAnnotations targets{boxes, 1, points, 1};
  augmatch::lambda(image, {brighten, nullptr}, targets);
  augmatch::clip_cbas_to_image_planes(targets, image.width, image.height);
  std::cout << "pixel=" << static_cast<int>(pixels[0]) << " box=" << boxes[0].x1
            << "," << boxes[0].x2 << " keypoint=" << points[0].x << "\n";
}
