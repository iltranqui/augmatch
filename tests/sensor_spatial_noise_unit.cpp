#include "augmatch/noise.hpp"
#include <cmath>
#include <vector>

int main() {
  constexpr int w = 5, h = 4, c = 2;
  std::vector<float> input(static_cast<std::size_t>(w) * h * c, 0.5f), output(input.size());
  std::vector<float> rows(static_cast<std::size_t>(h) * c), columns(static_cast<std::size_t>(w) * c);
  for (int y = 0; y < h; ++y) for (int ch = 0; ch < c; ++ch) rows[y * c + ch] = 0.01f * (y + 1) * (ch + 1);
  for (int x = 0; x < w; ++x) for (int ch = 0; ch < c; ++ch) columns[x * c + ch] = -0.02f * (x + 1) * (ch + 1);
  augmatch::RowColumnCorrelatedNoiseConfig spatial{w, h, c, 0.0f, 0.0f, rows.data(), columns.data(), 0.0f, 1.0f, 11};
  augmatch::row_column_correlated_noise_f32(input.data(), output.data(), spatial);
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) for (int ch = 0; ch < c; ++ch) {
    const std::size_t i = (static_cast<std::size_t>(y) * w + x) * c + ch;
    const float expected = 0.5f + rows[y * c + ch] + columns[x * c + ch];
    if (std::abs(output[i] - expected) > 1e-6f) return 1;
  }
  // Generated maps are fixed by coordinate, so pixels in one row/column share
  // their corresponding component while remaining bounded by clipping.
  spatial.row_map = nullptr; spatial.column_map = nullptr; spatial.row_stddev = 0.05f; spatial.column_stddev = 0.0f;
  augmatch::row_column_correlated_noise_f32(input.data(), output.data(), spatial);
  for (float v : output) if (v < 0.0f || v > 1.0f) return 2;
  for (int y = 0; y < h; ++y) for (int x = 1; x < w; ++x) for (int ch = 0; ch < c; ++ch)
    if (output[(y * w + x) * c + ch] != output[(y * w) * c + ch]) return 2;
  std::vector<augmatch::ClusteredDefect> defects{
      {1, 1, 1, augmatch::ClusteredDefectKind::Dead, 0.0f, -1},
      {3, 2, 0, augmatch::ClusteredDefectKind::Stuck, 1.2f, 0},
      {0, 0, 0, augmatch::ClusteredDefectKind::Hot, 0.8f, 1}};
  augmatch::ClusteredDefectivePixelsConfig clustered{w, h, c, defects.data(), defects.size(), 0, 0, 0.0f, 0.0f, 77};
  augmatch::clustered_defective_pixels_f32(input.data(), output.data(), clustered);
  if (output[(1 * w + 1) * c] != 0.0f || output[(1 * w + 1) * c + 1] != 0.0f) return 3;
  if (std::abs(output[(2 * w + 3) * c] - 1.0f) > 1e-6f || output[(2 * w + 3) * c + 1] != 0.5f) return 4;
  if (std::abs(output[1] - 1.0f) > 1e-6f || output[0] != 0.5f) return 5;
  clustered.defects = nullptr; clustered.defect_count = 0; clustered.cluster_count = 5; clustered.cluster_radius = 1; clustered.hot_value = 0.4f; clustered.stuck_value = 0.2f; clustered.seed = 123;
  std::vector<float> repeat(output.size());
  augmatch::clustered_defective_pixels_f32(input.data(), output.data(), clustered);
  augmatch::clustered_defective_pixels_f32(input.data(), repeat.data(), clustered);
  if (output != repeat) return 6;
  for (float v : output) if (v < 0.0f || v > 1.0f) return 7;
  return 0;
}
