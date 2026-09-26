#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace augmatch { namespace detail {

inline int reflect101(int p, int length) {
  if (length <= 1) return 0;
  while (p < 0 || p >= length) p = p < 0 ? -p : 2 * length - p - 2;
  return p;
}

// OpenCV-compatible 8-bit CLAHE core. The caller owns the host buffers.
inline void clahe_u8_host(const std::uint8_t* in, std::uint8_t* out,
                          int width, int height, int channels,
                          float clip_limit, int tiles_x, int tiles_y) {
  if (!in || !out) throw std::invalid_argument("null CLAHE buffer");
  if (width <= 0 || height <= 0 || channels <= 0 || tiles_x <= 0 || tiles_y <= 0)
    throw std::invalid_argument("invalid CLAHE dimensions or tile grid");
  if (!std::isfinite(clip_limit) || clip_limit <= 0.0f)
    throw std::invalid_argument("CLAHE clip limit must be finite and positive");

  // OpenCV pads both right and bottom when either dimension is not divisible.
  // In particular, a divisible dimension receives one full grid-width border
  // in that case; reproducing this detail is required for CLAHE parity.
  const bool needs_padding = width % tiles_x != 0 || height % tiles_y != 0;
  const int tile_width = needs_padding ? (width + tiles_x - width % tiles_x) / tiles_x : width / tiles_x;
  const int tile_height = needs_padding ? (height + tiles_y - height % tiles_y) / tiles_y : height / tiles_y;
  const int histogram_size = 256;
  const std::size_t tile_count = static_cast<std::size_t>(tiles_x) * tiles_y;
  std::vector<std::uint8_t> luts(tile_count * histogram_size);

  for (int channel = 0; channel < channels; ++channel) {
    for (int tile_y = 0; tile_y < tiles_y; ++tile_y) {
      for (int tile_x = 0; tile_x < tiles_x; ++tile_x) {
        int histogram[histogram_size] = {};
        const int x0 = tile_x * tile_width;
        const int y0 = tile_y * tile_height;
        for (int y = 0; y < tile_height; ++y) {
          const int sy = reflect101(y0 + y, height);
          for (int x = 0; x < tile_width; ++x) {
            const int sx = reflect101(x0 + x, width);
            ++histogram[in[(static_cast<std::size_t>(sy) * width + sx) * channels + channel]];
          }
        }
        const int tile_area = tile_width * tile_height;
        const int clip = std::max(1, static_cast<int>(clip_limit * tile_area / histogram_size));
        int clipped = 0;
        for (int value = 0; value < histogram_size; ++value) {
          if (histogram[value] > clip) {
            clipped += histogram[value] - clip;
            histogram[value] = clip;
          }
        }
        const int redist_batch = clipped / histogram_size;
        const int residual = clipped - redist_batch * histogram_size;
        for (int value = 0; value < histogram_size; ++value) histogram[value] += redist_batch;
        if (residual > 0) {
          const int residual_step = std::max(histogram_size / residual, 1);
          int remaining = residual;
          for (int value = 0; value < histogram_size && remaining > 0; value += residual_step, --remaining)
            ++histogram[value];
        }
        int cumulative = 0;
        const std::size_t lut_offset = (static_cast<std::size_t>(tile_y) * tiles_x + tile_x) * histogram_size;
        for (int value = 0; value < histogram_size; ++value) {
          cumulative += histogram[value];
          const double mapped = cumulative * 255.0 / tile_area;
          luts[lut_offset + value] = static_cast<std::uint8_t>(std::max(0L, std::min(255L, std::lrint(mapped))));
        }
      }
    }

    const double inv_tile_width = 1.0 / tile_width;
    const double inv_tile_height = 1.0 / tile_height;
    for (int y = 0; y < height; ++y) {
      double tyf = y * inv_tile_height - 0.5;
      int ty1 = static_cast<int>(std::floor(tyf));
      int ty2 = ty1 + 1;
      double ya = tyf - ty1;
      if (ty1 < 0) { ty1 = ty2 = 0; ya = 0.0; }
      if (ty2 >= tiles_y) { ty1 = ty2 = tiles_y - 1; ya = 0.0; }
      for (int x = 0; x < width; ++x) {
        double txf = x * inv_tile_width - 0.5;
        int tx1 = static_cast<int>(std::floor(txf));
        int tx2 = tx1 + 1;
        double xa = txf - tx1;
        if (tx1 < 0) { tx1 = tx2 = 0; xa = 0.0; }
        if (tx2 >= tiles_x) { tx1 = tx2 = tiles_x - 1; xa = 0.0; }
        const int value = in[(static_cast<std::size_t>(y) * width + x) * channels + channel];
        const auto lut = [&](int tile_x, int tile_y) -> int {
          return luts[(static_cast<std::size_t>(tile_y) * tiles_x + tile_x) * histogram_size + value];
        };
        const double top = lut(tx1, ty1) * (1.0 - xa) + lut(tx2, ty1) * xa;
        const double bottom = lut(tx1, ty2) * (1.0 - xa) + lut(tx2, ty2) * xa;
        const long result = std::lrint(top * (1.0 - ya) + bottom * ya);
        out[(static_cast<std::size_t>(y) * width + x) * channels + channel] =
            static_cast<std::uint8_t>(std::max(0L, std::min(255L, result)));
      }
    }
  }
}

}}  // namespace augmatch::detail
