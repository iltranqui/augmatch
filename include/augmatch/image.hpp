#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace augmatch {

enum class DataType : std::uint8_t { UInt8, UInt16, Float32 };
enum class Layout : std::uint8_t { HWC, CHW };
enum class MemorySpace : std::uint8_t { Host, Device };

constexpr std::size_t bytes_per_element(DataType type) noexcept {
  switch (type) {
    case DataType::UInt8: return 1;
    case DataType::UInt16: return 2;
    case DataType::Float32: return 4;
  }
  return 0;
}

struct ImageView {
  const void* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::ptrdiff_t stride_x = 0;
  std::ptrdiff_t stride_y = 0;
  std::ptrdiff_t stride_c = 0;
  DataType type = DataType::UInt8;
  Layout layout = Layout::HWC;
  MemorySpace memory = MemorySpace::Host;

  bool valid() const noexcept {
    return data != nullptr && width > 0 && height > 0 && channels > 0 &&
           bytes_per_element(type) != 0;
  }

  bool contiguous() const noexcept {
    const auto element = static_cast<std::ptrdiff_t>(bytes_per_element(type));
    if (layout == Layout::HWC) {
      return stride_c == element && stride_x == channels * element &&
             stride_y == static_cast<std::ptrdiff_t>(width) * channels * element;
    }
    return stride_x == element && stride_y == static_cast<std::ptrdiff_t>(width) * element &&
           stride_c == static_cast<std::ptrdiff_t>(width) * height * element;
  }

  void require_valid() const {
    if (!valid()) throw std::invalid_argument("invalid image view");
  }
};

struct MutableImageView {
  void* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::ptrdiff_t stride_x = 0;
  std::ptrdiff_t stride_y = 0;
  std::ptrdiff_t stride_c = 0;
  DataType type = DataType::UInt8;
  Layout layout = Layout::HWC;
  MemorySpace memory = MemorySpace::Host;

  ImageView as_const() const noexcept {
    return {data, width, height, channels, stride_x, stride_y, stride_c, type, layout, memory};
  }

  bool valid() const noexcept {
    return data != nullptr && width > 0 && height > 0 && channels > 0 &&
           bytes_per_element(type) != 0;
  }

  bool contiguous() const noexcept { return as_const().contiguous(); }

  void require_valid() const { as_const().require_valid(); }
};

// An image/mask pair is a non-owning target-aware transform view. The mask may
// be invalid when a caller intentionally requests the fallback crop; otherwise
// image and mask dimensions must match. Ownership stays with the caller.
struct ImageMaskView {
  ImageView image{};
  ImageView mask{};
};

struct MutableImageMaskView {
  MutableImageView image{};
  MutableImageView mask{};
};

inline ImageView make_hwc_u8_view(const std::uint8_t* data, int width, int height, int channels,
                                  MemorySpace memory = MemorySpace::Host) {
  const auto pixel = static_cast<std::ptrdiff_t>(channels);
  return {data, width, height, channels, pixel, static_cast<std::ptrdiff_t>(width) * pixel,
          1, DataType::UInt8, Layout::HWC, memory};
}

inline MutableImageView make_hwc_u8_view(std::uint8_t* data, int width, int height, int channels,
                                         MemorySpace memory = MemorySpace::Host) {
  const auto pixel = static_cast<std::ptrdiff_t>(channels);
  return {data, width, height, channels, pixel, static_cast<std::ptrdiff_t>(width) * pixel,
          1, DataType::UInt8, Layout::HWC, memory};
}

// These factories make the datatype explicit while retaining the byte-stride
// ABI of ImageView. They are useful for both packed and interleaved views.
template <typename T> constexpr DataType image_data_type() noexcept {
  static_assert(std::is_same<T, std::uint8_t>::value ||
                std::is_same<T, std::uint16_t>::value ||
                std::is_same<T, float>::value,
                "ImageView supports uint8_t, uint16_t, and float");
  return std::is_same<T, std::uint8_t>::value ? DataType::UInt8 :
         (std::is_same<T, std::uint16_t>::value ? DataType::UInt16 : DataType::Float32);
}

template <typename T>
inline ImageView make_hwc_view(const T* data, int width, int height, int channels,
                               MemorySpace memory = MemorySpace::Host) {
  const auto element = static_cast<std::ptrdiff_t>(sizeof(T));
  return {data, width, height, channels,
          static_cast<std::ptrdiff_t>(channels) * element,
          static_cast<std::ptrdiff_t>(width) * channels * element, element,
          image_data_type<T>(), Layout::HWC, memory};
}

template <typename T>
inline MutableImageView make_hwc_view(T* data, int width, int height, int channels,
                                      MemorySpace memory = MemorySpace::Host) {
  const auto element = static_cast<std::ptrdiff_t>(sizeof(T));
  return {data, width, height, channels,
          static_cast<std::ptrdiff_t>(channels) * element,
          static_cast<std::ptrdiff_t>(width) * channels * element, element,
          image_data_type<T>(), Layout::HWC, memory};
}

template <typename T>
inline ImageView make_chw_view(const T* data, int width, int height, int channels,
                               MemorySpace memory = MemorySpace::Host) {
  const auto element = static_cast<std::ptrdiff_t>(sizeof(T));
  return {data, width, height, channels, element,
          static_cast<std::ptrdiff_t>(width) * element,
          static_cast<std::ptrdiff_t>(width) * height * element,
          image_data_type<T>(), Layout::CHW, memory};
}

template <typename T>
inline MutableImageView make_chw_view(T* data, int width, int height, int channels,
                                      MemorySpace memory = MemorySpace::Host) {
  const auto element = static_cast<std::ptrdiff_t>(sizeof(T));
  return {data, width, height, channels, element,
          static_cast<std::ptrdiff_t>(width) * element,
          static_cast<std::ptrdiff_t>(width) * height * element,
          image_data_type<T>(), Layout::CHW, memory};
}

inline ImageView make_hwc_u16_view(const std::uint16_t* data, int width, int height, int channels,
                                    MemorySpace memory = MemorySpace::Host) {
  return make_hwc_view(data, width, height, channels, memory);
}
inline MutableImageView make_hwc_u16_view(std::uint16_t* data, int width, int height, int channels,
                                           MemorySpace memory = MemorySpace::Host) {
  return make_hwc_view(data, width, height, channels, memory);
}
inline ImageView make_hwc_f32_view(const float* data, int width, int height, int channels,
                                   MemorySpace memory = MemorySpace::Host) {
  return make_hwc_view(data, width, height, channels, memory);
}
inline MutableImageView make_hwc_f32_view(float* data, int width, int height, int channels,
                                          MemorySpace memory = MemorySpace::Host) {
  return make_hwc_view(data, width, height, channels, memory);
}
inline ImageView make_chw_u16_view(const std::uint16_t* data, int width, int height, int channels,
                                    MemorySpace memory = MemorySpace::Host) {
  return make_chw_view(data, width, height, channels, memory);
}
inline MutableImageView make_chw_u16_view(std::uint16_t* data, int width, int height, int channels,
                                           MemorySpace memory = MemorySpace::Host) {
  return make_chw_view(data, width, height, channels, memory);
}
}  // namespace augmatch
