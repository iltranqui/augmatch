#include "augmatch/bayer.hpp"
#include "augmatch/derivative.hpp"
#include "augmatch/noise.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
using augmatch::SensorNoiseConfig;
using Clock = std::chrono::steady_clock;

bool close(float a, float b, float tolerance) { return std::abs(a - b) <= tolerance; }

struct Moments {
  double mean = 0.0;
  double variance = 0.0;
};
Moments moments(const std::vector<float>& values, std::size_t begin, std::size_t end) {
  const double n = static_cast<double>(end - begin);
  double mean = 0.0;
  for (std::size_t i = begin; i < end; ++i) mean += values[i];
  mean /= n;
  double variance = 0.0;
  for (std::size_t i = begin; i < end; ++i) {
    const double d = values[i] - mean;
    variance += d * d;
  }
  return {mean, variance / (n - 1.0)};
}

bool poisson_variance_and_read_independence() {
  constexpr int width = 256, height = 128;
  std::vector<float> input(static_cast<std::size_t>(width) * height, 0.2f), output(input.size());
  SensorNoiseConfig poisson;
  poisson.width = width; poisson.height = height; poisson.channels = 1;
  poisson.electrons_per_unit = 100.0f; poisson.exposure = 1.0f;
  poisson.read_noise_electrons = 0.0f; poisson.adc_levels = 65536;
  poisson.seed = 0x11223344;
  augmatch::sensor_noise_f32(input.data(), output.data(), poisson);
  const Moments sample = moments(output, 0, output.size());
  // Poisson(lambda) in normalized electron units has variance lambda/e^2.
  const double expected_mean = 20.0 / 100.0;
  const double expected_variance = 20.0 / (100.0 * 100.0);
  if (!close(static_cast<float>(sample.mean), static_cast<float>(expected_mean), 0.015f) ||
      !close(static_cast<float>(sample.variance), static_cast<float>(expected_variance), 0.0008f)) return false;

  // Turn off the signal path and compare two signal levels. Read noise is an
  // additive electron term, so its normalized variance must not depend on input.
  std::fill(input.begin(), input.end(), 0.15f);
  for (std::size_t i = input.size() / 2; i < input.size(); ++i) input[i] = 0.75f;
  SensorNoiseConfig read = poisson;
  read.exposure = 0.0f; read.electrons_per_unit = 1.0f; read.read_noise_electrons = 0.2f;
  read.seed = 0x55667788;
  augmatch::sensor_noise_f32(input.data(), output.data(), read);
  const Moments low = moments(output, 0, output.size() / 2);
  const Moments high = moments(output, output.size() / 2, output.size());
  return std::abs(low.variance - high.variance) < 0.0002 && low.variance > 0.00001;
}

bool row_column_constancy() {
  constexpr int width = 9, height = 7, channels = 2;
  const std::size_t n = static_cast<std::size_t>(width) * height * channels;
  std::vector<float> input(n, 0.4f), output(n), rows(static_cast<std::size_t>(height) * channels), cols(static_cast<std::size_t>(width) * channels);
  for (int y = 0; y < height; ++y) for (int c = 0; c < channels; ++c) rows[y * channels + c] = 0.01f * (y + 1) * (c + 1);
  for (int x = 0; x < width; ++x) for (int c = 0; c < channels; ++c) cols[x * channels + c] = -0.005f * (x + 1) * (c + 1);
  augmatch::RowColumnCorrelatedNoiseConfig config;
  config.width = width; config.height = height; config.channels = channels;
  config.row_map = rows.data(); config.column_map = cols.data(); config.clip_min = 0.0f; config.clip_max = 1.0f;
  augmatch::row_column_correlated_noise_f32(input.data(), output.data(), config);
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < channels; ++c) {
    const std::size_t i = (static_cast<std::size_t>(y) * width + x) * channels + c;
    if (!close(output[i], input[i] + rows[y * channels + c] + cols[x * channels + c], 1e-6f)) return false;
  }
  return true;
}

bool defect_probabilities_and_bounds() {
  constexpr int width = 192, height = 128;
  const std::size_t n = static_cast<std::size_t>(width) * height;
  std::vector<float> input(n, 0.0f), output(n);
  SensorNoiseConfig hot;
  hot.width = width; hot.height = height; hot.channels = 1; hot.exposure = 0.0f;
  hot.electrons_per_unit = 1.0f; hot.read_noise_electrons = 0.0f; hot.amplifier_noise_electrons = 0.0f;
  hot.reset_noise_electrons = 0.0f; hot.hot_pixel_probability = 0.08f; hot.hot_pixel_electrons = 1.0f;
  hot.adc_levels = 65536; hot.seed = 91;
  augmatch::sensor_noise_f32(input.data(), output.data(), hot);
  const std::size_t hot_count = static_cast<std::size_t>(std::count_if(output.begin(), output.end(), [](float v) { return v > 0.5f; }));
  const double p = static_cast<double>(hot_count) / n;
  if (std::abs(p - 0.08) > 0.025) { std::cerr << "hot probability=" << p << '\n'; return false; }

  std::fill(input.begin(), input.end(), 1.0f);
  SensorNoiseConfig dead = hot;
  dead.hot_pixel_probability = 0.0f; dead.dead_pixel_probability = 0.08f;
  dead.electrons_per_unit = 100.0f; dead.exposure = 1.0f; dead.seed = 92;
  augmatch::sensor_noise_f32(input.data(), output.data(), dead);
  const std::size_t dead_count = static_cast<std::size_t>(std::count_if(output.begin(), output.end(), [](float v) { return v < 1e-6f; }));
  const double dead_p = static_cast<double>(dead_count) / n;
  if (std::abs(dead_p - 0.08) >= 0.025) { std::cerr << "dead probability=" << dead_p << '\n'; return false; }
  return std::all_of(output.begin(), output.end(), [](float v) { return v >= 0.0f && v <= 1.0f; });
}

bool clipping_black_level_and_dtype() {
  constexpr int width = 5, height = 1;
  std::vector<float> input{-1.0f, 0.0f, 0.25f, 1.0f, 2.0f}, output(input.size());
  SensorNoiseConfig config;
  config.width = width; config.height = height; config.channels = 1;
  config.exposure = 1.0f; config.electrons_per_unit = 100.0f; config.read_noise_electrons = 0.0f;
  config.black_level = 0.1f; config.white_level = 0.9f; config.adc_levels = 5; config.seed = 12;
  augmatch::sensor_noise_f32(input.data(), output.data(), config);
  // The public sensor contract is normalized after black-level subtraction.
  return output[0] == 0.0f && output[1] == 0.0f && output[4] == 1.0f &&
         std::all_of(output.begin(), output.end(), [](float v) { return v >= 0.0f && v <= 1.0f; });
}

bool bayer_and_derivative_golden_vectors() {
  constexpr int width = 4, height = 4;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3, 0), raw(width * height), reconstructed(rgb.size());
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
    const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 3;
    rgb[i] = 31; rgb[i + 1] = 97; rgb[i + 2] = 211;
  }
  augmatch::BayerConfig bayer{width, height, augmatch::BayerPattern::RGGB};
  augmatch::bayer_sample_u8(rgb.data(), raw.data(), bayer);
  const std::vector<std::uint8_t> expected_raw{31, 97, 31, 97, 97, 211, 97, 211, 31, 97, 31, 97, 97, 211, 97, 211};
  if (raw != expected_raw) return false;
  augmatch::bayer_demosaic_nearest_u8(raw.data(), reconstructed.data(), bayer);
  if (reconstructed != rgb) return false;

  std::vector<std::uint8_t> ramp(static_cast<std::size_t>(width) * height), gradient(ramp.size());
  for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) ramp[y * width + x] = static_cast<std::uint8_t>(40 + 10 * x);
  augmatch::DerivativeConfig derivative{width, height, 1, 1.0f, augmatch::BorderPolicy::Clamp, 0};
  augmatch::sobel_x_u8(ramp.data(), gradient.data(), derivative);
  // Interior horizontal ramp: [-1,0,1;-2,0,2;-1,0,1] gives 80.
  return gradient[1 * width + 1] == 80 && gradient[2 * width + 2] == 80;
}

bool deterministic_and_independent() {
  constexpr int width = 32, height = 32;
  std::vector<float> input(static_cast<std::size_t>(width) * height, 0.3f), a(input.size()), b(input.size()), c(input.size());
  SensorNoiseConfig config;
  config.width = width; config.height = height; config.channels = 1; config.electrons_per_unit = 80.0f;
  config.read_noise_electrons = 2.0f; config.adc_levels = 65536; config.seed = 101;
  augmatch::sensor_noise_f32(input.data(), a.data(), config);
  augmatch::sensor_noise_f32(input.data(), b.data(), config);
  config.seed = 102; augmatch::sensor_noise_f32(input.data(), c.data(), config);
  return a == b && std::inner_product(a.begin(), a.end(), c.begin(), 0.0, std::plus<float>(),
                                      [](float x, float y) { return std::abs(x - y) > 1e-7f ? 1.0f : 0.0f; }) > 0.0f;
}

bool run_validation() {
  const char* names[] = {"poisson/read", "row/column", "defect probabilities", "clipping", "bayer/derivative", "seed independence"};
  const bool checks[] = {
      poisson_variance_and_read_independence(), row_column_constancy(), defect_probabilities_and_bounds(),
      clipping_black_level_and_dtype(), bayer_and_derivative_golden_vectors(), deterministic_and_independent()};
  bool passed = true;
  for (std::size_t i = 0; i < std::size(checks); ++i) {
    if (!checks[i]) { std::cerr << "FAIL: " << names[i] << '\n'; passed = false; }
  }
  return passed;
}

void benchmark() {
  constexpr int channels = 1;
  const int sizes[][2] = {{512, 512}, {1920, 1080}, {3840, 2160}};
  for (const auto& size : sizes) {
    const int width = size[0], height = size[1];
    const std::size_t count = static_cast<std::size_t>(width) * height * channels;
    std::vector<float> input(count, 0.35f), output(count), copied(count);
    SensorNoiseConfig config;
    config.width = width; config.height = height; config.channels = channels; config.electrons_per_unit = 100.0f;
    config.read_noise_electrons = 2.0f; config.adc_levels = 256; config.seed = 7;
    const auto start = Clock::now();
    augmatch::sensor_noise_f32(input.data(), output.data(), config);
    const double transform_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const auto copy_start = Clock::now();
    std::memcpy(copied.data(), output.data(), count * sizeof(float));
    const double copy_ms = std::chrono::duration<double, std::milli>(Clock::now() - copy_start).count();
    std::cout << "BENCHMARK " << width << "x" << height << " transform_ms=" << transform_ms
              << " memcpy_ms=" << copy_ms << " bytes=" << count * sizeof(float) << '\n';
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--benchmark") { benchmark(); return 0; }
  if (!run_validation()) { std::cerr << "noise catalog validation failed\n"; return 1; }
  std::cout << "PASS: noise catalog validation (rows 591-610 CPU subset)\n";
  return 0;
}
