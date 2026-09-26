#include "augmatch/meta.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace augmatch {
namespace {
void validate_targets(MutableTargetAnnotations targets) {
  if (targets.box_count != 0 && targets.boxes == nullptr)
    throw std::invalid_argument("box array is null for nonzero box_count");
  if (targets.keypoint_count != 0 && targets.keypoints == nullptr)
    throw std::invalid_argument("keypoint array is null for nonzero keypoint_count");
}
void validate_targets(TargetAnnotations targets) {
  if (targets.box_count != 0 && targets.boxes == nullptr)
    throw std::invalid_argument("box array is null for nonzero box_count");
  if (targets.keypoint_count != 0 && targets.keypoints == nullptr)
    throw std::invalid_argument("keypoint array is null for nonzero keypoint_count");
}
void validate_cpu_image(const ImageView& image) {
  if (!image.valid() || image.stride_x <= 0 || image.stride_y <= 0 || image.stride_c <= 0)
    throw std::invalid_argument("meta operation requires a valid image view with positive strides");
  if (image.memory != MemorySpace::Host)
    throw std::invalid_argument("meta callbacks and assertions are host-only; device views are unsupported");
}
void validate_dimensions(int width, int height) {
  if (width <= 0 || height <= 0) throw std::invalid_argument("image plane dimensions must be positive");
}
float outside_fraction(BoxXYXY box, int width, int height) {
  const float area = box_area(box);
  if (area <= 0.0f) return 1.0f;
  const BoxXYXY clipped = clip_box(box, width, height);
  const float retained = box_area(clipped);
  return std::max(0.0f, std::min(1.0f, 1.0f - retained / area));
}
}  // namespace

void lambda(MutableImageView image, const LambdaConfig& config,
            MutableTargetAnnotations targets, const ExecutionContext& execution) {
  validate_cpu_image(image.as_const());
  validate_targets(targets);
  if (config.callback == nullptr) throw std::invalid_argument("Lambda callback must not be null");
  config.callback(image, targets, execution, config.context);
}

void assert_lambda(MutableImageView image, const AssertLambdaConfig& config,
                   TargetAnnotations targets, const ExecutionContext& execution) {
  validate_cpu_image(image.as_const());
  validate_targets(targets);
  if (config.predicate == nullptr)
    throw std::invalid_argument("AssertLambda predicate must not be null");
  const bool passed = config.predicate(image.as_const(), targets, execution, config.context);
  if (!passed) {
    throw std::invalid_argument(config.message == nullptr ? "AssertLambda predicate returned false"
                                                          : config.message);
  }
}

namespace {
void check_shape(ImageView image, const AssertShapeConfig& config) {
  validate_cpu_image(image);
  if (config.width < -1 || config.height < -1 || config.channels < -1)
    throw std::invalid_argument("AssertShape dimensions must be positive or wildcard (-1)");
  if (config.width >= 0 && image.width != config.width)
    throw std::invalid_argument("AssertShape width mismatch");
  if (config.height >= 0 && image.height != config.height)
    throw std::invalid_argument("AssertShape height mismatch");
  if (config.channels >= 0 && image.channels != config.channels)
    throw std::invalid_argument("AssertShape channel count mismatch");
  if (!config.type_any && image.type != config.type)
    throw std::invalid_argument("AssertShape data type mismatch");
  if (!config.layout_any && image.layout != config.layout)
    throw std::invalid_argument("AssertShape layout mismatch");
}
}  // namespace

void assert_shape(ImageView image, const AssertShapeConfig& config) {
  check_shape(image, config);
}

void assert_shape(MutableImageView image, const AssertShapeConfig& config) {
  check_shape(image.as_const(), config);
}

RemoveCBAsResult remove_cbas_by_out_of_image_fraction(
    MutableTargetAnnotations targets, int width, int height,
    float max_out_of_image_fraction, std::size_t* box_output_indices,
    std::size_t* keypoint_output_indices) {
  validate_targets(targets);
  validate_dimensions(width, height);
  if (!std::isfinite(max_out_of_image_fraction) || max_out_of_image_fraction < 0.0f ||
      max_out_of_image_fraction > 1.0f)
    throw std::invalid_argument("max_out_of_image_fraction must be finite in [0,1]");

  RemoveCBAsResult result{};
  for (std::size_t i = 0; i < targets.box_count; ++i) {
    if (outside_fraction(targets.boxes[i], width, height) > max_out_of_image_fraction) continue;
    if (result.box_count != i) targets.boxes[result.box_count] = targets.boxes[i];
    if (box_output_indices) box_output_indices[result.box_count] = i;
    ++result.box_count;
  }
  for (std::size_t i = 0; i < targets.keypoint_count; ++i) {
    const PointXY p = targets.keypoints[i];
    const float fraction = (std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0.0f &&
                            p.y >= 0.0f && p.x < static_cast<float>(width) &&
                            p.y < static_cast<float>(height)) ? 0.0f : 1.0f;
    if (fraction > max_out_of_image_fraction) continue;
    if (result.keypoint_count != i) targets.keypoints[result.keypoint_count] = p;
    if (keypoint_output_indices) keypoint_output_indices[result.keypoint_count] = i;
    ++result.keypoint_count;
  }
  return result;
}

void clip_cbas_to_image_planes(MutableTargetAnnotations targets, int width, int height) {
  validate_targets(targets);
  validate_dimensions(width, height);
  for (std::size_t i = 0; i < targets.box_count; ++i) {
    if (!std::isfinite(targets.boxes[i].x1) || !std::isfinite(targets.boxes[i].y1) ||
        !std::isfinite(targets.boxes[i].x2) || !std::isfinite(targets.boxes[i].y2))
      throw std::invalid_argument("cannot clip non-finite bounding box");
    targets.boxes[i] = clip_box(targets.boxes[i], width, height);
  }
  const float max_x = static_cast<float>(width - 1);
  const float max_y = static_cast<float>(height - 1);
  for (std::size_t i = 0; i < targets.keypoint_count; ++i) {
    if (!std::isfinite(targets.keypoints[i].x) || !std::isfinite(targets.keypoints[i].y))
      throw std::invalid_argument("cannot clip non-finite keypoint");
    targets.keypoints[i].x = std::clamp(targets.keypoints[i].x, 0.0f, max_x);
    targets.keypoints[i].y = std::clamp(targets.keypoints[i].y, 0.0f, max_y);
  }
}
}  // namespace augmatch
