// 06_weather_rain — overlay rain streaks, generated from a seed.
//
// What it shows:
//   * Weather effects are explicit records (RainStreak) that can be generated from a seed
//     or supplied by you; here we let augmatch generate them.
//   * make_random_rain_streaks shows exactly which streaks the seed produces (for logging/replay).
// API used: RandomRainConfig, make_random_rain_streaks, easy::rain.
// Output: 06_weather_rain.ppm — the test image with 40 slanted white streaks.

#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"
#include "example_utils.hpp"

int main() {
  const int width = 64, height = 48, channels = 3;
  std::vector<std::uint8_t> input = example::make_test_image(width, height);
  std::vector<std::uint8_t> output(input.size());

  augmatch::RandomRainConfig rain;
  rain.width = width;             // make_random_rain_streaks needs the size to place streaks
  rain.height = height;           // (easy::rain fills these in automatically)
  rain.channels = channels;
  rain.streak_count = 40;         // how many streaks to draw
  rain.alpha = 0.6f;              // 0 = invisible, 1 = solid white
  rain.length_min = 6.0f;         // streak length range, in pixels
  rain.length_max = 14.0f;
  rain.angle_min = 70.0f;         // angle range in degrees from +x (90 = vertical)
  rain.angle_max = 80.0f;
  rain.seed = 7;                  // streaks == nullptr, so placement comes from this seed

  // Optional: look at the generated streaks. Passing these back via rain.streaks gives the same image.
  std::vector<augmatch::RainStreak> streaks(rain.streak_count);
  augmatch::make_random_rain_streaks(streaks.data(), rain);
  std::cout << "first streak: (" << streaks[0].x0 << "," << streaks[0].y0 << ") -> ("
            << streaks[0].x1 << "," << streaks[0].y1 << ")\n";

  const auto in = augmatch::make_hwc_u8_view(input.data(), width, height, channels);
  const auto out = augmatch::make_hwc_u8_view(output.data(), width, height, channels);
  const augmatch::Status status = augmatch::easy::rain(in, out, rain);
  if (!status) { std::cerr << "rain: " << status.message << '\n'; return 1; }

  example::write_ppm("06_weather_rain.ppm", output, width, height);
  std::cout << "wrote 06_weather_rain.ppm\n";
  return 0;
}
