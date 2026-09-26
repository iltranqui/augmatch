#include "augmatch/weather.hpp"
#include "augmatch/imgcorruptlike.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  const int w = 4, h = 2, c = 3;
  std::vector<std::uint8_t> image(static_cast<std::size_t>(w) * h * c, 40), output(image.size());
  std::vector<std::uint8_t> bright(image.size(), 255);
  augmatch::fast_snowy_landscape_u8(bright.data(), output.data(), {w, h, c, 0.0f, 1.0f});
  for (std::uint8_t value : output) assert(value == 255);

  augmatch::CloudsConfig clouds{w, h, c, 0.7f, 1.0f, nullptr, 42};
  std::vector<float> field(static_cast<std::size_t>(w) * h), field_again(field.size());
  augmatch::make_cloud_field(field.data(), clouds);
  augmatch::make_cloud_field(field_again.data(), clouds);
  assert(field == field_again);
  augmatch::clouds_u8(image.data(), output.data(), clouds);
  assert(output != image);

  augmatch::FogConfig fog{w, h, c, 1.0f, 0.0f, nullptr, 7};
  augmatch::fog_u8(image.data(), output.data(), fog);
  assert(output == image);

  std::vector<float> layer(static_cast<std::size_t>(w) * h, 0.0f);
  layer[0] = 1.0f;
  augmatch::WeatherLayerConfig layer_config{w, h, c, layer.data(), 1.0f};
  augmatch::cloud_layer_u8(image.data(), output.data(), layer_config);
  for (int channel = 0; channel < c; ++channel) assert(output[channel] == 255);
  for (std::size_t i = c; i < output.size(); ++i) assert(output[i] == image[i]);

  augmatch::Snowflake flake{1.5f, 0.5f, 0.75f, 1.0f};
  augmatch::SnowflakesConfig snow{w, h, c, &flake, 1, 1.0f, 1.0f, 1.0f, 0};
  augmatch::snowflakes_u8(image.data(), output.data(), snow);
  assert(output[c] == 255);

  augmatch::RainStreak streak{2.5f, 0.5f, 2.5f, 1.5f, 1.0f, 1.0f};
  augmatch::RainConfig rain{w, h, c, &streak, 1, 1.0f, 1.0f, 1.0f, 90.0f, 90.0f, 1.0f, 0};
  augmatch::rain_u8(image.data(), output.data(), rain);
  assert(output[2 * c + 0] == 255);
  return 0;
}
