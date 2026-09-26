#include "augmatch/annotations/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace augmatch {
namespace {
void validate_geometry(const Geometry& g) {
  if (g.input_width <= 0 || g.input_height <= 0 || g.channels <= 0 ||
      g.output_width <= 0 || g.output_height <= 0 || g.crop_x < 0 || g.crop_y < 0 ||
      g.crop_x + g.output_width > g.input_width || g.crop_y + g.output_height > g.input_height)
    throw std::invalid_argument("invalid image or crop geometry");
}

BoxXYXY transform_box_unclipped(BoxXYXY b, const Geometry& g) {
  b.x1 -= static_cast<float>(g.crop_x);
  b.x2 -= static_cast<float>(g.crop_x);
  b.y1 -= static_cast<float>(g.crop_y);
  b.y2 -= static_cast<float>(g.crop_y);
  if (g.flip_horizontal) {
    const float x1 = static_cast<float>(g.output_width) - b.x2;
    const float x2 = static_cast<float>(g.output_width) - b.x1;
    b.x1 = x1;
    b.x2 = x2;
  }
  if (g.flip_vertical) {
    const float y1 = static_cast<float>(g.output_height) - b.y2;
    const float y2 = static_cast<float>(g.output_height) - b.y1;
    b.y1 = y1;
    b.y2 = y2;
  }
  return b;
}
}  // namespace

PointXY transform_point(PointXY p, const Geometry& g) {
  validate_geometry(g);
  p.x -= static_cast<float>(g.crop_x);
  p.y -= static_cast<float>(g.crop_y);
  if (g.flip_horizontal) p.x = static_cast<float>(g.output_width - 1) - p.x;
  if (g.flip_vertical) p.y = static_cast<float>(g.output_height - 1) - p.y;
  return p;
}

BoxXYXY transform_box(BoxXYXY box, const Geometry& g) {
  validate_geometry(g);
  box = transform_box_unclipped(box, g);
  return clip_box(box, g.output_width, g.output_height);
}

void transform_points(const PointXY* input, PointXY* output, std::size_t count,
                      const Geometry& g) {
  validate_geometry(g);
  if (count != 0 && (!input || !output))
    throw std::invalid_argument("point arrays must be non-null when count is nonzero");
  for (std::size_t i = 0; i < count; ++i) output[i] = transform_point(input[i], g);
}

void transform_boxes(const BoxXYXY* input, BoxXYXY* output, std::size_t count,
                     const Geometry& g) {
  validate_geometry(g);
  if (count != 0 && (!input || !output))
    throw std::invalid_argument("box arrays must be non-null when count is nonzero");
  for (std::size_t i = 0; i < count; ++i) output[i] = transform_box(input[i], g);
}

float box_area(BoxXYXY box) noexcept {
  if (!std::isfinite(box.x1) || !std::isfinite(box.y1) ||
      !std::isfinite(box.x2) || !std::isfinite(box.y2) ||
      box.x2 <= box.x1 || box.y2 <= box.y1)
    return 0.0f;
  return (box.x2 - box.x1) * (box.y2 - box.y1);
}

bool box_is_valid(BoxXYXY box) noexcept {
  return std::isfinite(box.x1) && std::isfinite(box.y1) &&
         std::isfinite(box.x2) && std::isfinite(box.y2) &&
         box.x2 > box.x1 && box.y2 > box.y1;
}

BoxXYXY clip_box(BoxXYXY box, int width, int height) noexcept {
  box.x1 = std::clamp(box.x1, 0.0f, static_cast<float>(std::max(width, 0)));
  box.x2 = std::clamp(box.x2, 0.0f, static_cast<float>(std::max(width, 0)));
  box.y1 = std::clamp(box.y1, 0.0f, static_cast<float>(std::max(height, 0)));
  box.y2 = std::clamp(box.y2, 0.0f, static_cast<float>(std::max(height, 0)));
  return box;
}

std::size_t transform_boxes_filtered(const BoxXYXY* input, BoxXYXY* output,
                                     std::size_t count, const Geometry& g,
                                     const BoxTransformOptions& options,
                                     std::size_t* output_indices) {
  validate_geometry(g);
  if (count != 0 && (!input || !output))
    throw std::invalid_argument("box arrays must be non-null when count is nonzero");
  if (!std::isfinite(options.min_visibility) || options.min_visibility < 0.0f ||
      options.min_visibility > 1.0f || !std::isfinite(options.min_area) || options.min_area < 0.0f)
    throw std::invalid_argument("box filtering thresholds must be finite and non-negative");

  std::size_t written = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const float original_area = box_area(input[i]);
    if (original_area <= 0.0f) continue;
    BoxXYXY transformed = transform_box_unclipped(input[i], g);
    const BoxXYXY retained = clip_box(transformed, g.output_width, g.output_height);
    const BoxXYXY candidate = options.clip ? retained : transformed;
    const float retained_area = box_area(retained);
    const float candidate_area = box_area(candidate);
    if (candidate_area < options.min_area || retained_area / original_area < options.min_visibility ||
        !box_is_valid(candidate))
      continue;
    output[written] = candidate;
    if (output_indices) output_indices[written] = i;
    ++written;
  }
  return written;
}

std::size_t filter_boxes(const BoxXYXY* input, BoxXYXY* output, std::size_t count,
                         float min_area) {
  if (count != 0 && (!input || !output))
    throw std::invalid_argument("box arrays must be non-null when count is nonzero");
  if (!std::isfinite(min_area) || min_area < 0.0f)
    throw std::invalid_argument("minimum box area must be finite and non-negative");
  std::size_t written = 0;
  for (std::size_t i = 0; i < count; ++i)
    if (box_is_valid(input[i]) && box_area(input[i]) >= min_area) output[written++] = input[i];
  return written;
}
}  // namespace augmatch
