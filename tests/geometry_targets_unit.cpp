#include "augmatch/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
bool close(float a, float b) { return std::fabs(a - b) < 1e-6f; }
bool same(augmatch::BoxXYXY a, augmatch::BoxXYXY b) {
  return close(a.x1, b.x1) && close(a.y1, b.y1) && close(a.x2, b.x2) && close(a.y2, b.y2);
}
}  // namespace

int main() {
  using namespace augmatch;
  const Geometry g{6, 5, 1, 1, 1, 4, 3, true, true};

  // Pixel-centre points use width-1/height-1 under flips; edge boxes use
  // width-x2/width-x1. The array APIs must preserve source order.
  const PointXY points[]{{1.0f, 1.0f}, {4.0f, 3.0f}, {2.5f, 2.25f}};
  PointXY transformed_points[3]{};
  transform_points(points, transformed_points, 3, g);
  if (!close(transformed_points[0].x, 3.0f) || !close(transformed_points[0].y, 2.0f) ||
      !close(transformed_points[1].x, 0.0f) || !close(transformed_points[1].y, -0.0f)) return 1;

  const BoxXYXY boxes[]{{1.0f, 1.0f, 4.0f, 3.0f}, {-2.0f, 0.0f, 2.0f, 2.0f},
                        {2.0f, 1.0f, 2.0f, 4.0f}};
  BoxXYXY transformed_boxes[3]{};
  transform_boxes(boxes, transformed_boxes, 3, g);
  if (!same(transformed_boxes[0], {1.0f, 1.0f, 4.0f, 3.0f})) return 2;
  if (!same(transformed_boxes[1], {3.0f, 2.0f, 4.0f, 3.0f})) return 3;

  if (!same(clip_box({-2.0f, 1.0f, 7.0f, 6.0f}, 4, 3), {0, 1, 4, 3})) return 4;
  if (!box_is_valid({0, 0, 1, 1}) || box_is_valid({0, 0, 0, 1}) ||
      box_area({0, 0, 2, 3}) != 6.0f) return 5;

  BoxXYXY filtered[3]{};
  std::size_t output_indices[3]{};
  const std::size_t kept = transform_boxes_filtered(
      boxes, filtered, 3, g, BoxTransformOptions{true, 0.5f, 2.0f}, output_indices);
  if (kept != 1 || output_indices[0] != 0 || !same(filtered[0], {1, 1, 4, 3})) return 6;
  const BoxXYXY already[]{{0, 0, 1, 1}, {0, 0, 2, 2}, {1, 1, 1, 3}};
  if (filter_boxes(already, filtered, 3, 2.0f) != 1 || !same(filtered[0], {0, 0, 2, 2})) return 7;

  // Both policies are explicit even though Geometry currently has no scale:
  // crop and flips map output centres to exact input centres.
  const std::uint8_t mask[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 26, 27, 28, 29};
  std::uint8_t nearest[12]{}, linear[12]{};
  transform_mask_u8(mask, nearest, g, MaskInterpolation::Nearest);
  transform_mask_u8(mask, linear, g, MaskInterpolation::Linear);
  for (int i = 0; i < 12; ++i) if (nearest[i] != linear[i]) return 8;

  bool rejected = false;
  try {
    transform_boxes_filtered(boxes, filtered, 3, g, BoxTransformOptions{true, 1.5f, 0});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 9;

  // Property check: batch metadata helpers are exactly the scalar contract for
  // arbitrary finite points and boxes, including boxes crossing the crop.
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> coordinate(-3.0f, 9.0f);
  std::vector<PointXY> random_points(128), batch_points(128);
  std::vector<BoxXYXY> random_boxes(128), batch_boxes(128);
  for (std::size_t i = 0; i < random_points.size(); ++i) {
    random_points[i] = {coordinate(rng), coordinate(rng)};
    const float x = coordinate(rng), y = coordinate(rng);
    random_boxes[i] = {std::min(x, coordinate(rng)), std::min(y, coordinate(rng)),
                       std::max(x, coordinate(rng)), std::max(y, coordinate(rng))};
  }
  transform_points(random_points.data(), batch_points.data(), random_points.size(), g);
  transform_boxes(random_boxes.data(), batch_boxes.data(), random_boxes.size(), g);
  for (std::size_t i = 0; i < random_points.size(); ++i) {
    const PointXY scalar_point = transform_point(random_points[i], g);
    const BoxXYXY scalar_box = transform_box(random_boxes[i], g);
    if (!close(batch_points[i].x, scalar_point.x) || !close(batch_points[i].y, scalar_point.y) ||
        !same(batch_boxes[i], scalar_box)) return 10;
  }
  return 0;
}
