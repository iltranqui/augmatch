#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include "augmatch/image.hpp"
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
struct ConvertConfig { int width=0,height=0,channels=0; };
void to_float_u8(const std::uint8_t*,float*,const ConvertConfig&,cudaStream_t stream=nullptr);
void from_float_u8(const float*,std::uint8_t*,const ConvertConfig&,cudaStream_t stream=nullptr);
struct NormalizeConfig { int width=0,height=0,channels=0; float mean0=0.485f,mean1=0.456f,mean2=0.406f; float std0=0.229f,std1=0.224f,std2=0.225f; };
void normalize_u8_f32(const std::uint8_t*,float*,const NormalizeConfig&,cudaStream_t stream=nullptr);

// TensorViewF32 is a non-owning, contiguous CHW float32 tensor contract. Its
// strides are measured in float elements (unlike ImageView's byte strides).
struct TensorViewF32 {
  const float* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::ptrdiff_t stride_x = 0;
  std::ptrdiff_t stride_y = 0;
  std::ptrdiff_t stride_c = 0;
  DataType type = DataType::Float32;
  Layout layout = Layout::CHW;
  MemorySpace memory = MemorySpace::Host;

  bool valid() const noexcept {
    return data != nullptr && width > 0 && height > 0 && channels > 0 &&
           type == DataType::Float32 && layout == Layout::CHW;
  }
  bool contiguous() const noexcept {
    return valid() && stride_x == 1 && stride_y == width &&
           stride_c == static_cast<std::ptrdiff_t>(width) * height;
  }
};

struct MutableTensorViewF32 {
  float* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::ptrdiff_t stride_x = 0;
  std::ptrdiff_t stride_y = 0;
  std::ptrdiff_t stride_c = 0;
  DataType type = DataType::Float32;
  Layout layout = Layout::CHW;
  MemorySpace memory = MemorySpace::Host;

  TensorViewF32 as_const() const noexcept {
    return {data, width, height, channels, stride_x, stride_y, stride_c, type, layout, memory};
  }
  bool valid() const noexcept { return as_const().valid(); }
  bool contiguous() const noexcept { return as_const().contiguous(); }
};

inline MutableTensorViewF32 make_chw_f32_view(float* data, int width, int height, int channels,
                                               MemorySpace memory = MemorySpace::Host) {
  return {data, width, height, channels, 1, width,
          static_cast<std::ptrdiff_t>(width) * height, DataType::Float32, Layout::CHW, memory};
}

inline TensorViewF32 make_chw_f32_view(const float* data, int width, int height, int channels,
                                       MemorySpace memory = MemorySpace::Host) {
  return {data, width, height, channels, 1, width,
          static_cast<std::ptrdiff_t>(width) * height, DataType::Float32, Layout::CHW, memory};
}

// ToTensorV2Config describes native HWC uint8 -> CHW float32 conversion. When
// normalize is false each output is input / 255. When true, output channel c is
// (input / 255 - mean[min(c,2)]) / std[min(c,2)], using the three channel
// parameters below; channels after the third reuse the third parameter.
// This is a tensor-buffer conversion only; it has no PyTorch dependency.
struct ToTensorV2Config {
  int width = 0;
  int height = 0;
  int channels = 0;
  bool normalize = false;
  float mean0 = 0.485f;
  float mean1 = 0.456f;
  float mean2 = 0.406f;
  float std0 = 0.229f;
  float std1 = 0.224f;
  float std2 = 0.225f;
};
using TensorConfig = ToTensorV2Config;

struct TensorNormalizeConfig {
  bool enabled = false;
  float mean0 = 0.485f;
  float mean1 = 0.456f;
  float mean2 = 0.406f;
  float std0 = 0.229f;
  float std1 = 0.224f;
  float std2 = 0.225f;
};

void to_tensor_v2_u8_f32(const std::uint8_t*, float*, const ToTensorV2Config&, cudaStream_t stream=nullptr);
void to_tensor_v2(const std::uint8_t*, float*, const ToTensorV2Config&, cudaStream_t stream=nullptr);
void to_tensor_chw_f32(const std::uint8_t*, float*, const ToTensorV2Config&, cudaStream_t stream=nullptr);
void to_tensor_v2(const ImageView&, const MutableTensorViewF32&, const TensorNormalizeConfig& = {}, cudaStream_t stream=nullptr);

// ToTensor3DConfig describes native contiguous DHWC uint8 -> CDHW float32
// conversion. Input index is (((d * height + y) * width + x) * channels + c)
// and output index is (((c * depth + d) * height + y) * width + x). When
// normalize is false each output is input / 255. When true, output channel c
// is (input / 255 - mean[min(c,2)]) / std[min(c,2)]; channels after the third
// reuse the third parameter. The raw-pointer API does not allocate or copy.
struct ToTensor3DConfig {
  int width = 0;
  int height = 0;
  int depth = 0;
  int channels = 0;
  bool normalize = false;
  float mean0 = 0.485f;
  float mean1 = 0.456f;
  float mean2 = 0.406f;
  float std0 = 0.229f;
  float std1 = 0.224f;
  float std2 = 0.225f;
};
using Tensor3DConfig = ToTensor3DConfig;

// These entry points have identical CPU and CUDA semantics. On CUDA, input
// and output pointers refer to device memory and completion follows the
// supplied stream. On CPU, pointers refer to host memory and stream is ignored.
void to_tensor_3d_u8_f32(const std::uint8_t*, float*, const ToTensor3DConfig&, cudaStream_t stream=nullptr);
void to_tensor_3d(const std::uint8_t*, float*, const ToTensor3DConfig&, cudaStream_t stream=nullptr);
void to_tensor_cdhw_f32(const std::uint8_t*, float*, const ToTensor3DConfig&, cudaStream_t stream=nullptr);
}
