#include "augmatch/target_metadata.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace augmatch {
namespace {
void dimensions(int width, int height) {
  if (width <= 0 || height <= 0) throw std::invalid_argument("image dimensions must be positive");
}
void finite_value(float value, const char* what) {
  if (!std::isfinite(value)) throw std::invalid_argument(std::string(what) + " must be finite");
}
void validate_target_view(const AdditionalImageTarget& target, const Geometry& g) {
  const auto check = [&](const ImageView& v, int width, int height, const char* name) {
    if (!v.valid() || v.memory != MemorySpace::Host || v.type != DataType::UInt8 ||
        v.stride_x <= 0 || v.stride_y <= 0 || v.stride_c <= 0)
      throw std::invalid_argument(std::string(name) + " must be a strided host uint8 view");
    if (v.width != width || v.height != height)
      throw std::invalid_argument(std::string(name) + " dimensions do not match geometry");
  };
  check(target.input, g.input_width, g.input_height, "additional target input");
  check(target.output.as_const(), g.output_width, g.output_height, "additional target output");
  if (target.input.channels != target.output.channels)
    throw std::invalid_argument("additional target channel counts must match");
}
void validate_geometry_dimensions(const Geometry& g) {
  dimensions(g.input_width, g.input_height);
  dimensions(g.output_width, g.output_height);
  if (g.crop_x < 0 || g.crop_y < 0 || g.crop_x + g.output_width > g.input_width ||
      g.crop_y + g.output_height > g.input_height)
    throw std::invalid_argument("geometry crop is outside the input image");
}
void transform_target(const AdditionalImageTarget& target, const Geometry& g) {
  validate_target_view(target, g);
  const int channels = target.input.channels;
  std::vector<std::uint8_t> values(static_cast<std::size_t>(g.output_width) * g.output_height * channels);
  const auto* source = static_cast<const std::uint8_t*>(target.input.data);
  auto* destination = static_cast<std::uint8_t*>(target.output.data);
  for (int y = 0; y < g.output_height; ++y) {
    const int source_y = g.crop_y + (g.flip_vertical ? g.output_height - 1 - y : y);
    for (int x = 0; x < g.output_width; ++x) {
      const int source_x = g.crop_x + (g.flip_horizontal ? g.output_width - 1 - x : x);
      const auto* source_pixel = source + source_y * target.input.stride_y +
                                 source_x * target.input.stride_x;
      auto* packed_pixel = values.data() +
                           (static_cast<std::size_t>(y) * g.output_width + x) * channels;
      for (int c = 0; c < channels; ++c)
        packed_pixel[c] = source_pixel[c * target.input.stride_c];
    }
  }
  for (int y = 0; y < g.output_height; ++y) {
    for (int x = 0; x < g.output_width; ++x) {
      const auto* packed_pixel = values.data() +
                                 (static_cast<std::size_t>(y) * g.output_width + x) * channels;
      auto* destination_pixel = destination + y * target.output.stride_y + x * target.output.stride_x;
      for (int c = 0; c < channels; ++c)
        destination_pixel[c * target.output.stride_c] = packed_pixel[c];
    }
  }
}
void validate_box_array(const BoxXYXY* boxes, std::size_t count, int width, int height,
                        bool require_inside) {
  dimensions(width, height);
  if (count != 0 && boxes == nullptr) throw std::invalid_argument("box array is null");
  for (std::size_t i = 0; i < count; ++i) {
    if (!box_is_valid(boxes[i])) throw std::invalid_argument("Pascal VOC box is invalid");
    if (require_inside && (boxes[i].x1 < 0 || boxes[i].y1 < 0 ||
                           boxes[i].x2 > width || boxes[i].y2 > height))
      throw std::invalid_argument("Pascal VOC box is outside the image");
  }
}
void validate_geometry_for_boxes(int width, int height, const Geometry& g) {
  dimensions(width, height);
  validate_geometry_dimensions(g);
  if (g.input_width != width || g.input_height != height)
    throw std::invalid_argument("box dimensions do not match geometry input");
}
void normalized(float value, const char* what) {
  finite_value(value, what);
  if (value < 0.0f || value > 1.0f)
    throw std::invalid_argument(std::string(what) + " must be in [0,1]");
}
}  // namespace

void validate_additional_image_targets(const AdditionalImageTarget* targets,
                                      std::size_t count, const Geometry& geometry) {
  validate_geometry_dimensions(geometry);
  if (count != 0 && targets == nullptr) throw std::invalid_argument("additional target array is null");
  for (std::size_t i = 0; i < count; ++i) validate_target_view(targets[i], geometry);
}
void validate_additional_mask_targets(const AdditionalMaskTarget* targets,
                                     std::size_t count, const Geometry& geometry) {
  validate_additional_image_targets(targets, count, geometry);
}
void transform_additional_images(const AdditionalImageTarget* targets, std::size_t count,
                                const Geometry& geometry) {
  validate_additional_image_targets(targets, count, geometry);
  for (std::size_t i = 0; i < count; ++i) transform_target(targets[i], geometry);
}
void transform_additional_masks(const AdditionalMaskTarget* targets, std::size_t count,
                               const Geometry& geometry, MaskInterpolation interpolation) {
  if (interpolation != MaskInterpolation::Nearest && interpolation != MaskInterpolation::Linear)
    throw std::invalid_argument("unknown mask interpolation policy");
  validate_additional_mask_targets(targets, count, geometry);
  for (std::size_t i = 0; i < count; ++i) transform_target(targets[i], geometry);
}

BoxXYXY coco_to_pascal(COCOBox box, int width, int height) {
  dimensions(width, height);
  finite_value(box.x, "COCO x"); finite_value(box.y, "COCO y");
  if (!std::isfinite(box.width) || !std::isfinite(box.height) || box.width <= 0 || box.height <= 0)
    throw std::invalid_argument("COCO width and height must be finite and positive");
  const float x2 = box.x + box.width;
  const float y2 = box.y + box.height;
  if (!std::isfinite(x2) || !std::isfinite(y2))
    throw std::invalid_argument("COCO box extent must be finite");
  return {box.x, box.y, x2, y2};
}
COCOBox pascal_to_coco(BoxXYXY box, int width, int height) {
  validate_box_array(&box, 1, width, height, false);
  return {box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1};
}
BoxXYXY yolo_to_pascal(YOLOBox box, int width, int height) {
  dimensions(width, height);
  normalized(box.center_x, "YOLO center_x"); normalized(box.center_y, "YOLO center_y");
  normalized(box.width, "YOLO width"); normalized(box.height, "YOLO height");
  if (box.width <= 0 || box.height <= 0) throw std::invalid_argument("YOLO dimensions must be positive");
  return {(box.center_x - box.width / 2) * width, (box.center_y - box.height / 2) * height,
          (box.center_x + box.width / 2) * width, (box.center_y + box.height / 2) * height};
}
YOLOBox pascal_to_yolo(BoxXYXY box, int width, int height) {
  validate_box_array(&box, 1, width, height, true);
  return {(box.x1 + box.x2) / (2 * width), (box.y1 + box.y2) / (2 * height),
          (box.x2 - box.x1) / width, (box.y2 - box.y1) / height};
}
BoxXYXY albumentations_to_pascal(AlbumentationsBox box, int width, int height) {
  dimensions(width, height);
  normalized(box.x1, "Albumentations x1"); normalized(box.y1, "Albumentations y1");
  normalized(box.x2, "Albumentations x2"); normalized(box.y2, "Albumentations y2");
  if (box.x2 <= box.x1 || box.y2 <= box.y1)
    throw std::invalid_argument("Albumentations box must be non-empty");
  return {box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height};
}
AlbumentationsBox pascal_to_albumentations(BoxXYXY box, int width, int height) {
  validate_box_array(&box, 1, width, height, true);
  return {box.x1 / width, box.y1 / height, box.x2 / width, box.y2 / height};
}

void validate_pascal_voc_boxes(const BoxXYXY* boxes, std::size_t count,
                               int width, int height, bool require_inside) {
  validate_box_array(boxes, count, width, height, require_inside);
}
void validate_coco_boxes(const COCOBox* boxes, std::size_t count, int width, int height) {
  dimensions(width, height);
  if (count != 0 && !boxes) throw std::invalid_argument("COCO box array is null");
  for (std::size_t i = 0; i < count; ++i) {
    finite_value(boxes[i].x, "COCO x"); finite_value(boxes[i].y, "COCO y");
    if (!std::isfinite(boxes[i].width) || !std::isfinite(boxes[i].height) ||
        boxes[i].width <= 0 || boxes[i].height <= 0)
      throw std::invalid_argument("COCO dimensions must be finite and positive");
  }
}
void validate_yolo_boxes(const YOLOBox* boxes, std::size_t count, int width, int height) {
  dimensions(width, height);
  if (count != 0 && !boxes) throw std::invalid_argument("YOLO box array is null");
  for (std::size_t i = 0; i < count; ++i) {
    normalized(boxes[i].center_x, "YOLO center_x"); normalized(boxes[i].center_y, "YOLO center_y");
    normalized(boxes[i].width, "YOLO width"); normalized(boxes[i].height, "YOLO height");
    if (boxes[i].width <= 0 || boxes[i].height <= 0)
      throw std::invalid_argument("YOLO dimensions must be positive");
  }
}
void validate_albumentations_boxes(const AlbumentationsBox* boxes, std::size_t count,
                                   int width, int height) {
  dimensions(width, height);
  if (count != 0 && !boxes) throw std::invalid_argument("Albumentations box array is null");
  for (std::size_t i = 0; i < count; ++i) {
    normalized(boxes[i].x1, "Albumentations x1"); normalized(boxes[i].y1, "Albumentations y1");
    normalized(boxes[i].x2, "Albumentations x2"); normalized(boxes[i].y2, "Albumentations y2");
    if (boxes[i].x2 <= boxes[i].x1 || boxes[i].y2 <= boxes[i].y1)
      throw std::invalid_argument("Albumentations box must be non-empty");
  }
}

void transform_coco_boxes(const COCOBox* input, COCOBox* output, std::size_t count,
                          int width, int height, const Geometry& geometry) {
  validate_coco_boxes(input, count, width, height); validate_geometry_for_boxes(width, height, geometry);
  if (count != 0 && !output) throw std::invalid_argument("COCO output array is null");
  for (std::size_t i = 0; i < count; ++i) output[i] = pascal_to_coco(transform_box(coco_to_pascal(input[i], width, height), geometry), geometry.output_width, geometry.output_height);
}
void transform_yolo_boxes(const YOLOBox* input, YOLOBox* output, std::size_t count,
                          int width, int height, const Geometry& geometry) {
  validate_yolo_boxes(input, count, width, height); validate_geometry_for_boxes(width, height, geometry);
  if (count != 0 && !output) throw std::invalid_argument("YOLO output array is null");
  for (std::size_t i = 0; i < count; ++i) output[i] = pascal_to_yolo(transform_box(yolo_to_pascal(input[i], width, height), geometry), geometry.output_width, geometry.output_height);
}
void transform_albumentations_boxes(const AlbumentationsBox* input, AlbumentationsBox* output,
                                    std::size_t count, int width, int height,
                                    const Geometry& geometry) {
  validate_albumentations_boxes(input, count, width, height); validate_geometry_for_boxes(width, height, geometry);
  if (count != 0 && !output) throw std::invalid_argument("Albumentations output array is null");
  for (std::size_t i = 0; i < count; ++i) output[i] = pascal_to_albumentations(transform_box(albumentations_to_pascal(input[i], width, height), geometry), geometry.output_width, geometry.output_height);
}

void validate_keypoint_visibility(const KeypointXYV* keypoints, std::size_t count,
                                 int width, int height, bool require_inside) {
  dimensions(width, height);
  if (count != 0 && !keypoints) throw std::invalid_argument("keypoint array is null");
  for (std::size_t i = 0; i < count; ++i) {
    finite_value(keypoints[i].x, "keypoint x"); finite_value(keypoints[i].y, "keypoint y");
    if (!std::isfinite(keypoints[i].visibility) || keypoints[i].visibility < 0 || keypoints[i].visibility > 1)
      throw std::invalid_argument("keypoint visibility must be finite in [0,1]");
    if (require_inside && (keypoints[i].x < 0 || keypoints[i].y < 0 || keypoints[i].x >= width || keypoints[i].y >= height))
      throw std::invalid_argument("keypoint is outside the image");
  }
}
void transform_keypoints_with_visibility(const KeypointXYV* input, KeypointXYV* output,
                                         std::size_t count, const Geometry& geometry,
                                         bool mark_out_of_frame_invisible) {
  validate_geometry_dimensions(geometry);
  validate_keypoint_visibility(input, count, geometry.input_width, geometry.input_height, false);
  if (count != 0 && !output) throw std::invalid_argument("keypoint output array is null");
  for (std::size_t i = 0; i < count; ++i) {
    const PointXY transformed = transform_point({input[i].x, input[i].y}, geometry);
    output[i] = {transformed.x, transformed.y, input[i].visibility};
    if (mark_out_of_frame_invisible && (transformed.x < 0 || transformed.y < 0 ||
                                        transformed.x >= geometry.output_width ||
                                        transformed.y >= geometry.output_height))
      output[i].visibility = 0.0f;
  }
}

void validate_polygon(const PolygonView& polygon, int width, int height) {
  dimensions(width, height);
  if (!polygon.vertices || !polygon.ring_offsets || polygon.ring_count == 0 ||
      polygon.vertex_count == 0 || polygon.ring_offsets[0] != 0 ||
      polygon.ring_offsets[polygon.ring_count] != polygon.vertex_count)
    throw std::invalid_argument("polygon requires vertices and valid ring offsets");
  for (std::size_t r = 0; r < polygon.ring_count; ++r) {
    const auto begin = polygon.ring_offsets[r], end = polygon.ring_offsets[r + 1];
    if (end < begin || end - begin < 3) throw std::invalid_argument("polygon rings need at least three vertices");
    for (std::size_t i = begin; i < end; ++i) {
      finite_value(polygon.vertices[i].x, "polygon x"); finite_value(polygon.vertices[i].y, "polygon y");
      if (polygon.vertices[i].x < 0 || polygon.vertices[i].y < 0 ||
          polygon.vertices[i].x >= width || polygon.vertices[i].y >= height)
        throw std::invalid_argument("polygon vertex is outside the image");
    }
  }
}
void transform_polygon(const PolygonView& input, PointXY* output, const Geometry& geometry) {
  validate_geometry_dimensions(geometry);
  if (geometry.input_width <= 0 || geometry.input_height <= 0) throw std::invalid_argument("invalid polygon geometry");
  validate_polygon(input, geometry.input_width, geometry.input_height);
  if (!output) throw std::invalid_argument("polygon output vertices are null");
  std::vector<PointXY> temp(input.vertex_count);
  for (std::size_t i = 0; i < input.vertex_count; ++i) temp[i] = transform_point(input.vertices[i], geometry);
  std::copy(temp.begin(), temp.end(), output);
}
void validate_line_string(const LineStringView& line, int width, int height) {
  dimensions(width, height);
  if (!line.points || line.point_count < 2) throw std::invalid_argument("line string requires at least two points");
  for (std::size_t i = 0; i < line.point_count; ++i) {
    finite_value(line.points[i].x, "line x"); finite_value(line.points[i].y, "line y");
    if (line.points[i].x < 0 || line.points[i].y < 0 || line.points[i].x >= width || line.points[i].y >= height)
      throw std::invalid_argument("line string point is outside the image");
  }
}
void transform_line_string(const LineStringView& input, PointXY* output, const Geometry& geometry) {
  validate_geometry_dimensions(geometry);
  validate_line_string(input, geometry.input_width, geometry.input_height);
  if (!output) throw std::invalid_argument("line string output points are null");
  std::vector<PointXY> temp(input.point_count);
  for (std::size_t i = 0; i < input.point_count; ++i) temp[i] = transform_point(input.points[i], geometry);
  std::copy(temp.begin(), temp.end(), output);
}
namespace {
template <class View> void validate_heatmap_view(const View& v) {
  if (!v.data || v.width <= 0 || v.height <= 0 || v.channels <= 0 || v.stride_x < static_cast<std::ptrdiff_t>(sizeof(float)) ||
      v.stride_y < v.stride_x * v.width || (v.channels > 1 && v.stride_c < v.stride_y * v.height))
    throw std::invalid_argument("heatmap must be a positive-stride host float32 view");
}
}
void validate_heatmap(const HeatmapView& heatmap) { validate_heatmap_view(heatmap); }
void validate_heatmap(const MutableHeatmapView& heatmap) { validate_heatmap_view(heatmap); }
void transform_heatmap(const HeatmapView& input, const MutableHeatmapView& output, const Geometry& geometry) {
  validate_geometry_dimensions(geometry);
  validate_heatmap(input); validate_heatmap(output);
  if (input.width != geometry.input_width || input.height != geometry.input_height ||
      output.width != geometry.output_width || output.height != geometry.output_height ||
      input.channels != output.channels)
    throw std::invalid_argument("heatmap dimensions do not match geometry");
  std::vector<float> temp(static_cast<std::size_t>(output.width) * output.height * output.channels);
  for (int y = 0; y < output.height; ++y) {
    const int sy = geometry.crop_y + (geometry.flip_vertical ? output.height - 1 - y : y);
    for (int x = 0; x < output.width; ++x) {
      const int sx = geometry.crop_x + (geometry.flip_horizontal ? output.width - 1 - x : x);
      for (int c = 0; c < input.channels; ++c)
        temp[(static_cast<std::size_t>(y) * output.width + x) * output.channels + c] =
          *reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(input.data) + sy * input.stride_y + sx * input.stride_x + c * input.stride_c);
    }
  }
  for (int y = 0; y < output.height; ++y) for (int x = 0; x < output.width; ++x) for (int c = 0; c < output.channels; ++c)
    *reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(output.data) + y * output.stride_y + x * output.stride_x + c * output.stride_c) =
      temp[(static_cast<std::size_t>(y) * output.width + x) * output.channels + c];
}
}  // namespace augmatch
