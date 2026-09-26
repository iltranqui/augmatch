#pragma once

#include <cstddef>
#include <cstdint>

#include "augmatch/core/execution.hpp"
#include "augmatch/core/image.hpp"
#include "augmatch/annotations/transforms.hpp"

namespace augmatch {

// Target arrays are borrowed. The caller retains ownership and capacity; meta
// operations compact boxes/keypoints in place and return their new counts.
struct MutableTargetAnnotations {
  BoxXYXY* boxes = nullptr;
  std::size_t box_count = 0;
  PointXY* keypoints = nullptr;
  std::size_t keypoint_count = 0;
};

struct TargetAnnotations {
  const BoxXYXY* boxes = nullptr;
  std::size_t box_count = 0;
  const PointXY* keypoints = nullptr;
  std::size_t keypoint_count = 0;
};

// Lambda callbacks are synchronous host callbacks. Both callback and context
// are borrowed and must remain valid until the call returns; neither may be
// retained. Callbacks may modify the supplied image and target arrays.
using LambdaCallback = void (*)(MutableImageView image, MutableTargetAnnotations targets,
                                const ExecutionContext& execution, void* context);
using AssertLambdaPredicate = bool (*)(ImageView image, TargetAnnotations targets,
                                       const ExecutionContext& execution, void* context);

struct LambdaConfig {
  LambdaCallback callback = nullptr;
  void* context = nullptr;
};

struct AssertLambdaConfig {
  AssertLambdaPredicate predicate = nullptr;
  void* context = nullptr;
  const char* message = "AssertLambda predicate returned false";
};

// A negative dimension is a wildcard. Shape checks are synchronous and do not
// alter image bytes or execution state. type/layout checks are optional when
// type_any/layout_any are true.
struct AssertShapeConfig {
  int width = -1;
  int height = -1;
  int channels = -1;
  DataType type = DataType::UInt8;
  Layout layout = Layout::HWC;
  bool type_any = true;
  bool layout_any = true;
};

void lambda(MutableImageView image, const LambdaConfig& config,
            MutableTargetAnnotations targets = {},
            const ExecutionContext& execution = {});
void assert_lambda(MutableImageView image, const AssertLambdaConfig& config,
                   TargetAnnotations targets = {},
                   const ExecutionContext& execution = {});
void assert_shape(ImageView image, const AssertShapeConfig& config);
void assert_shape(MutableImageView image, const AssertShapeConfig& config);

// Returns the number of boxes and keypoints retained. Box out-of-image
// fraction is 1 - intersection_area / original_area; invalid or empty boxes
// have fraction 1. Keypoints use fraction 0 when their pixel centre is inside
// [0,width) x [0,height), otherwise 1. Arrays are compacted in place.
struct RemoveCBAsResult {
  std::size_t box_count = 0;
  std::size_t keypoint_count = 0;
};
RemoveCBAsResult remove_cbas_by_out_of_image_fraction(
    MutableTargetAnnotations targets, int width, int height,
    float max_out_of_image_fraction, std::size_t* box_output_indices = nullptr,
    std::size_t* keypoint_output_indices = nullptr);

// Clips continuous box edges to [0,width] x [0,height] and keypoint centres
// to [0,width-1] x [0,height-1]. The arrays remain caller-owned.
void clip_cbas_to_image_planes(MutableTargetAnnotations targets, int width, int height);

}  // namespace augmatch
