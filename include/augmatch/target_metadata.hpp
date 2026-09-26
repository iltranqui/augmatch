#pragma once

#include <cstddef>
#include <cstdint>

#include "augmatch/image.hpp"
#include "augmatch/transforms.hpp"

namespace augmatch {

// Additional image and mask targets are borrowed source/destination pairs. All
// pairs use the same Geometry as the primary image; no target is retained.
struct AdditionalImageTarget {
  ImageView input{};
  MutableImageView output{};
};
using AdditionalMaskTarget = AdditionalImageTarget;

// Validate dimensions, type, layout, and host ownership for extra targets.
// Image targets use the image interpolation policy; masks default to nearest.
void validate_additional_image_targets(const AdditionalImageTarget* targets,
                                      std::size_t count, const Geometry& geometry);
void validate_additional_mask_targets(const AdditionalMaskTarget* targets,
                                      std::size_t count, const Geometry& geometry);
void transform_additional_images(const AdditionalImageTarget* targets, std::size_t count,
                                const Geometry& geometry);
void transform_additional_masks(const AdditionalMaskTarget* targets, std::size_t count,
                               const Geometry& geometry,
                               MaskInterpolation interpolation = MaskInterpolation::Nearest);
// Descriptive aliases matching the catalog terminology.
inline void validate_additional_image_target(const AdditionalImageTarget* targets,
                                             std::size_t count, const Geometry& geometry) {
  validate_additional_image_targets(targets, count, geometry);
}
inline void validate_additional_mask_target(const AdditionalMaskTarget* targets,
                                            std::size_t count, const Geometry& geometry) {
  validate_additional_mask_targets(targets, count, geometry);
}
inline void transform_additional_image_targets(const AdditionalImageTarget* targets,
                                               std::size_t count, const Geometry& geometry) {
  transform_additional_images(targets, count, geometry);
}
inline void transform_additional_mask_targets(const AdditionalMaskTarget* targets,
                                              std::size_t count, const Geometry& geometry,
                                              MaskInterpolation interpolation = MaskInterpolation::Nearest) {
  transform_additional_masks(targets, count, geometry, interpolation);
}

// Absolute Pascal VOC boxes use continuous half-open image-edge coordinates
// (x1,y1,x2,y2). COCO uses absolute (x,y,width,height). YOLO uses normalized
// (centre_x,centre_y,width,height), and Albumentations uses normalized
// (x1,y1,x2,y2). Normalized values are fractions of width/height, not pixels.
struct COCOBox { float x, y, width, height; };
struct YOLOBox { float center_x, center_y, width, height; };
struct AlbumentationsBox { float x1, y1, x2, y2; };

using BoxXYWH = COCOBox;
using BoxCXCYWH = YOLOBox;
using NormalizedBoxXYXY = AlbumentationsBox;
using COCOBoxXYWH = COCOBox;
using YOLOBoxCXCYWH = YOLOBox;
using AlbumentationsBoxXYXY = AlbumentationsBox;

BoxXYXY coco_to_pascal(COCOBox box, int image_width, int image_height);
COCOBox pascal_to_coco(BoxXYXY box, int image_width, int image_height);
BoxXYXY yolo_to_pascal(YOLOBox box, int image_width, int image_height);
YOLOBox pascal_to_yolo(BoxXYXY box, int image_width, int image_height);
BoxXYXY albumentations_to_pascal(AlbumentationsBox box, int image_width, int image_height);
AlbumentationsBox pascal_to_albumentations(BoxXYXY box, int image_width, int image_height);
inline BoxXYXY coco_box_to_pascal(COCOBox box, int width, int height) { return coco_to_pascal(box, width, height); }
inline COCOBox pascal_box_to_coco(BoxXYXY box, int width, int height) { return pascal_to_coco(box, width, height); }
inline BoxXYXY yolo_box_to_pascal(YOLOBox box, int width, int height) { return yolo_to_pascal(box, width, height); }
inline YOLOBox pascal_box_to_yolo(BoxXYXY box, int width, int height) { return pascal_to_yolo(box, width, height); }
inline BoxXYXY normalized_box_to_pascal(AlbumentationsBox box, int width, int height) { return albumentations_to_pascal(box, width, height); }
inline AlbumentationsBox pascal_box_to_normalized(BoxXYXY box, int width, int height) { return pascal_to_albumentations(box, width, height); }

void validate_pascal_voc_boxes(const BoxXYXY* boxes, std::size_t count,
                               int image_width, int image_height,
                               bool require_inside = false);
void validate_coco_boxes(const COCOBox* boxes, std::size_t count,
                        int image_width, int image_height);
void validate_yolo_boxes(const YOLOBox* boxes, std::size_t count,
                        int image_width, int image_height);
void validate_albumentations_boxes(const AlbumentationsBox* boxes, std::size_t count,
                                   int image_width, int image_height);
inline void validate_pascal_voc_box(BoxXYXY box, int width, int height,
                                    bool require_inside = false) {
  validate_pascal_voc_boxes(&box, 1, width, height, require_inside);
}

// These transforms compact no records and preserve order. Like transform_boxes,
// output boxes are clipped to the output image before conversion back to the
// requested format. Input and output may alias.
void transform_coco_boxes(const COCOBox* input, COCOBox* output, std::size_t count,
                          int image_width, int image_height, const Geometry& geometry);
void transform_yolo_boxes(const YOLOBox* input, YOLOBox* output, std::size_t count,
                          int image_width, int image_height, const Geometry& geometry);
void transform_albumentations_boxes(const AlbumentationsBox* input,
                                    AlbumentationsBox* output, std::size_t count,
                                    int image_width, int image_height,
                                    const Geometry& geometry);
inline void transform_pascal_voc_boxes(const BoxXYXY* input, BoxXYXY* output,
                                       std::size_t count, const Geometry& geometry) {
  transform_boxes(input, output, count, geometry);
}

// Visibility is a finite confidence in [0,1]: 0 means not visible/not labeled,
// 1 means visible, and intermediate values represent caller-defined occlusion.
// Geometry never changes visibility for an in-frame point. By default an
// out-of-frame point is marked 0; set mark_out_of_frame_invisible false to
// preserve the caller's value while retaining the transformed coordinates.
struct KeypointXYV { float x, y, visibility; };
using KeypointVisibility = KeypointXYV;
using KeypointXYVisibility = KeypointXYV;
void validate_keypoint_visibility(const KeypointXYV* keypoints, std::size_t count,
                                 int image_width, int image_height,
                                 bool require_inside = false);
void transform_keypoints_with_visibility(const KeypointXYV* input, KeypointXYV* output,
                                         std::size_t count, const Geometry& geometry,
                                         bool mark_out_of_frame_invisible = true);
inline void transform_keypoints_visibility(const KeypointXYV* input, KeypointXYV* output,
                                           std::size_t count, const Geometry& geometry,
                                           bool mark_out_of_frame_invisible = true) {
  transform_keypoints_with_visibility(input, output, count, geometry, mark_out_of_frame_invisible);
}

// Polygon geometry is host metadata in both CPU and CUDA builds. Each polygon
// contains one or more closed rings represented without a repeated endpoint;
// ring_offsets has ring_count+1 entries, begins at zero, and ends at
// vertex_count. Transform preserves ring and vertex order and does not clip.
struct PolygonView {
  const PointXY* vertices = nullptr;
  std::size_t vertex_count = 0;
  const std::size_t* ring_offsets = nullptr;
  std::size_t ring_count = 0;
};
void validate_polygon(const PolygonView& polygon, int image_width, int image_height);
void transform_polygon(const PolygonView& input, PointXY* output_vertices,
                       const Geometry& geometry);

// Ordered open line strings require at least two vertices. Host-only metadata
// storage applies in CUDA builds; coordinates use image pixel-centre convention.
struct LineStringView {
  const PointXY* points = nullptr;
  std::size_t point_count = 0;
};
void validate_line_string(const LineStringView& line, int image_width, int image_height);
void transform_line_string(const LineStringView& input, PointXY* output_points,
                           const Geometry& geometry);

// Continuous scalar heatmaps are host float32 views with explicit byte strides.
// Geometry performs only crop/flips (no resampling); channels are independent.
struct HeatmapView {
  const float* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 1;
  std::ptrdiff_t stride_x = sizeof(float);
  std::ptrdiff_t stride_y = 0;
  std::ptrdiff_t stride_c = 0;
};
struct MutableHeatmapView {
  float* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 1;
  std::ptrdiff_t stride_x = sizeof(float);
  std::ptrdiff_t stride_y = 0;
  std::ptrdiff_t stride_c = 0;
};
void validate_heatmap(const HeatmapView& heatmap);
void validate_heatmap(const MutableHeatmapView& heatmap);
void transform_heatmap(const HeatmapView& input, const MutableHeatmapView& output,
                       const Geometry& geometry);

}  // namespace augmatch
