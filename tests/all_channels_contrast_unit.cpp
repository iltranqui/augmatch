#include "augmatch/color/tone.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  try {
    const int width = 11, height = 7, channels = 4;
    const std::size_t elements = static_cast<std::size_t>(width) * height * channels;
    std::vector<std::uint8_t> input(elements), clahe_alias(elements), clahe_native(elements);
    std::vector<std::uint8_t> equalize_alias(elements), equalize_native(elements);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        for (int channel = 0; channel < channels; ++channel)
          input[(static_cast<std::size_t>(y) * width + x) * channels + channel] =
              static_cast<std::uint8_t>((13 * x + 29 * y + 47 * channel + x * y) & 255);

    const augmatch::AllChannelsCLAHEConfig clahe_config{width, height, channels, 3.5f, 4, 3};
    augmatch::all_channels_clahe_u8(input.data(), clahe_alias.data(), clahe_config);
    augmatch::clahe_u8(input.data(), clahe_native.data(),
                       {width, height, channels, clahe_config.clip_limit, clahe_config.tiles_x, clahe_config.tiles_y});
    if (clahe_alias != clahe_native) throw std::runtime_error("AllChannelsCLAHE differs from native CLAHE");

    const augmatch::AllChannelsHistogramEqualizationConfig equalize_config{width, height, channels};
    augmatch::all_channels_histogram_equalization_u8(input.data(), equalize_alias.data(), equalize_config);
    augmatch::equalize_u8(input.data(), equalize_native.data(), {width, height, channels});
    if (equalize_alias != equalize_native)
      throw std::runtime_error("AllChannelsHistogramEqualization differs from native equalize");
    bool fourth_channel_changed = false;
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(width) * height; ++pixel)
      fourth_channel_changed = fourth_channel_changed ||
          equalize_alias[pixel * channels + 3] != input[pixel * channels + 3];
    if (!fourth_channel_changed)
      throw std::runtime_error("all channels policy did not process the fourth channel");

    std::cout << "all-channel contrast aliases passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "all-channel contrast aliases failed: " << error.what() << '\n';
    return 1;
  }
}
