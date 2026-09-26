#include "augmatch/pipeline/composition.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace augmatch {
namespace {
void validate_cpu_image(const MutableImageView& image) {
  if (!image.valid()) throw std::invalid_argument("composition requires a valid image view");
  if (image.memory != MemorySpace::Host)
    throw std::invalid_argument("CPU composition requires a host image view");
}

void validate_device_image(const MutableImageView& image) {
  if (!image.valid()) throw std::invalid_argument("composition requires a valid image view");
  if (image.memory != MemorySpace::Device)
    throw std::invalid_argument("device composition requires a device image view");
}

ChannelRange validate_range(const MutableImageView& image, ChannelRange range) {
  if (range.count < -1 || range.first < 0 || range.first > image.channels ||
      (range.count >= 0 && range.count > image.channels - range.first))
    throw std::invalid_argument("composition channel range is outside the image");
  if (range.count < 0) range.count = image.channels - range.first;
  return range;
}

MutableImageView channel_view(MutableImageView image, ChannelRange range) {
  range = validate_range(image, range);
  const auto element = static_cast<std::ptrdiff_t>(bytes_per_element(image.type));
  auto* base = static_cast<std::uint8_t*>(image.data);
  const std::ptrdiff_t offset = image.layout == Layout::HWC
                                    ? range.first * element
                                    : range.first * image.stride_c;
  image.data = base + offset;
  image.channels = range.count;
  return image;
}

ChannelRange intersect_ranges(ChannelRange outer, ChannelRange inner,
                              const MutableImageView& image) {
  outer = validate_range(image, outer);
  inner = validate_range(image, inner);
  const int begin = std::max(outer.first, inner.first);
  const int end = std::min(outer.first + outer.count, inner.first + inner.count);
  return {begin, std::max(0, end - begin)};
}

bool selection_is_unset(const ChannelSelection& selection) {
  return selection.indices.empty() && selection.ranges.empty();
}

std::vector<ChannelRange> normalize_selection(const MutableImageView& image,
                                              const ChannelSelection& selection) {
  std::vector<ChannelRange> ranges;
  if (selection_is_unset(selection)) {
    ranges.push_back(validate_range(image, ChannelRange{}));
    return ranges;
  }
  for (const int index : selection.indices) {
    if (index < 0 || index >= image.channels)
      throw std::invalid_argument("composition channel index is outside the image");
    ranges.push_back({index, 1});
  }
  for (const ChannelRange range : selection.ranges) {
    const ChannelRange valid = validate_range(image, range);
    if (valid.count > 0) ranges.push_back(valid);
  }
  std::sort(ranges.begin(), ranges.end(), [](const ChannelRange lhs, const ChannelRange rhs) {
    return lhs.first < rhs.first;
  });
  std::vector<ChannelRange> merged;
  for (const ChannelRange range : ranges) {
    if (merged.empty() || range.first > merged.back().first + merged.back().count) {
      merged.push_back(range);
    } else {
      const int end = std::max(merged.back().first + merged.back().count,
                               range.first + range.count);
      merged.back().count = end - merged.back().first;
    }
  }
  return merged;
}

std::vector<ChannelRange> stage_ranges(const MutableImageView& image,
                                       const CpuStage& stage) {
  std::vector<ChannelRange> ranges;
  if (selection_is_unset(stage.channel_selection)) {
    ranges.push_back(validate_range(image, stage.channel_range));
  } else {
    for (const ChannelRange selected : normalize_selection(image, stage.channel_selection)) {
      const ChannelRange range = intersect_ranges(selected, stage.channel_range, image);
      if (range.count > 0) ranges.push_back(range);
    }
  }
  return ranges;
}

void apply_cpu(MutableImageView image, const CpuPipeline& pipeline,
               const ExecutionContext& execution,
               const ChannelSelection& selected = {}) {
  validate_cpu_image(image);
  const auto selected_ranges = normalize_selection(image, selected);
  for (const CpuStage& stage : pipeline.stages) {
    if (stage.callback == nullptr)
      throw std::invalid_argument("CPU composition contains a null stage callback");
    for (const ChannelRange selected_range : selected_ranges) {
      for (const ChannelRange stage_range : stage_ranges(image, stage)) {
        const ChannelRange range = intersect_ranges(selected_range, stage_range, image);
        if (range.count > 0)
          stage.callback(channel_view(image, range), execution, stage.context);
      }
    }
  }
}

void apply_device_fallback(MutableImageView image, const DevicePipeline& pipeline,
                           const ExecutionContext& execution) {
  validate_device_image(image);
  (void)pipeline;
  (void)execution;
  throw std::invalid_argument("device composition requires the CUDA backend");
}
std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

StageSelectionRecord choose(const ExecutionContext& execution, SelectionKind kind,
                            std::size_t option_count, double probability = 1.0) {
  if (option_count == 0) throw std::invalid_argument("OneOf requires at least one choice");
  if (execution.record != nullptr && execution.replay != nullptr)
    throw std::invalid_argument("composition record and replay cannot both be set");
  const std::size_t position = execution.replay_index++;
  if (execution.replay != nullptr) {
    if (position >= execution.replay->selections.size())
      throw std::invalid_argument("composition replay is missing a stage selection");
    const StageSelectionRecord entry = execution.replay->selections[position];
    if (entry.kind != kind || entry.option_count != option_count ||
        entry.selected >= option_count)
      throw std::invalid_argument("composition replay stage selection does not match pipeline");
    return entry;
  }
  StageSelectionRecord entry;
  entry.kind = kind;
  entry.option_count = option_count;
  const std::uint64_t random_bits = splitmix64(execution.seed + position);
  if (kind == SelectionKind::RandomApply) {
    entry.selected = 0;
    entry.applied = probability >= 1.0 ||
                    (probability > 0.0 &&
                     static_cast<double>(random_bits >> 11U) / 9007199254740992.0 < probability);
  } else {
    entry.selected = static_cast<std::size_t>(random_bits % option_count);
    entry.applied = true;
  }
  if (execution.record != nullptr) execution.record->selections.push_back(entry);
  return entry;
}

void validate_probability(double probability) {
  if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
    throw std::invalid_argument("RandomApply probability must be finite in [0,1]");
}

StageSelectionRecord choose_indices(const ExecutionContext& execution, SelectionKind kind,
                                    std::size_t option_count, std::size_t count,
                                    bool replace) {
  if (option_count == 0 || (!replace && count > option_count))
    throw std::invalid_argument("SomeOf selection count is outside the choices");
  if (execution.record != nullptr && execution.replay != nullptr)
    throw std::invalid_argument("composition record and replay cannot both be set");
  const std::size_t position = execution.replay_index++;
  if (execution.replay != nullptr) {
    if (position >= execution.replay->selections.size())
      throw std::invalid_argument("composition replay is missing a stage selection");
    const StageSelectionRecord entry = execution.replay->selections[position];
    if (entry.kind != kind || entry.option_count != option_count ||
        entry.selected != count || entry.with_replacement != replace ||
        entry.selected_indices.size() != count)
      throw std::invalid_argument("composition replay SomeOf selection does not match pipeline");
    for (const auto index : entry.selected_indices)
      if (index >= option_count)
        throw std::invalid_argument("composition replay selected index is outside choices");
    if (!replace) {
      std::vector<std::size_t> check = entry.selected_indices;
      std::sort(check.begin(), check.end());
      if (std::adjacent_find(check.begin(), check.end()) != check.end())
        throw std::invalid_argument("composition replay repeats a non-replaceable choice");
    }
    return entry;
  }
  StageSelectionRecord entry;
  entry.kind = kind;
  entry.option_count = option_count;
  entry.selected = count;
  entry.applied = true;
  entry.with_replacement = replace;
  if (replace) {
    for (std::size_t i = 0; i < count; ++i)
      entry.selected_indices.push_back(
          static_cast<std::size_t>(splitmix64(execution.seed + position + i) % option_count));
  } else {
    std::vector<std::size_t> available(option_count);
    for (std::size_t i = 0; i < option_count; ++i) available[i] = i;
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t remaining = option_count - i;
      const std::size_t offset =
          static_cast<std::size_t>(splitmix64(execution.seed + position + i) % remaining);
      entry.selected_indices.push_back(available[offset]);
      available[offset] = available[remaining - 1];
    }
  }
  if (execution.record != nullptr) execution.record->selections.push_back(entry);
  return entry;
}

StageSelectionRecord choose_order(const ExecutionContext& execution,
                                  std::size_t option_count) {
  if (execution.record != nullptr && execution.replay != nullptr)
    throw std::invalid_argument("composition record and replay cannot both be set");
  const std::size_t position = execution.replay_index++;
  if (execution.replay != nullptr) {
    if (position >= execution.replay->selections.size())
      throw std::invalid_argument("composition replay is missing a stage order");
    const StageSelectionRecord entry = execution.replay->selections[position];
    if (entry.kind != SelectionKind::RandomOrder || entry.option_count != option_count ||
        entry.order.size() != option_count)
      throw std::invalid_argument("composition replay stage order does not match pipeline");
    std::vector<std::size_t> check = entry.order;
    std::sort(check.begin(), check.end());
    for (std::size_t i = 0; i < option_count; ++i)
      if (check[i] != i)
        throw std::invalid_argument("composition replay stage order is not a permutation");
    return entry;
  }
  StageSelectionRecord entry;
  entry.kind = SelectionKind::RandomOrder;
  entry.option_count = option_count;
  entry.applied = true;
  entry.order.resize(option_count);
  for (std::size_t i = 0; i < option_count; ++i) entry.order[i] = i;
  for (std::size_t i = option_count; i > 1; --i) {
    const std::size_t offset =
        static_cast<std::size_t>(splitmix64(execution.seed + position + i) % i);
    std::swap(entry.order[i - 1], entry.order[offset]);
  }
  if (execution.record != nullptr) execution.record->selections.push_back(entry);
  return entry;
}
}  // namespace

CpuPipeline make_compose(std::vector<CpuStage> stages) {
  return CpuPipeline{std::move(stages)};
}

CpuPipeline make_sequential(std::vector<CpuStage> stages) {
  return CpuPipeline{std::move(stages)};
}

DevicePipeline make_compose(std::vector<DeviceStage> stages) {
  return DevicePipeline{std::move(stages)};
}

DevicePipeline make_sequential(std::vector<DeviceStage> stages) {
  return DevicePipeline{std::move(stages)};
}

void compose(MutableImageView image, const CpuPipeline& pipeline,
             const ExecutionContext& execution) {
  if (execution.record != nullptr && execution.replay != nullptr)
    throw std::invalid_argument("composition record and replay cannot both be set");
  execution.replay_index = 0;
  apply_cpu(image, pipeline, execution);
}

void sequential(MutableImageView image, const CpuPipeline& pipeline,
                const ExecutionContext& execution) {
  if (execution.record != nullptr && execution.replay != nullptr)
    throw std::invalid_argument("composition record and replay cannot both be set");
  execution.replay_index = 0;
  apply_cpu(image, pipeline, execution);
}

void compose(MutableImageView image, const DevicePipeline& pipeline,
             const ExecutionContext& execution) {
  if (execution.record != nullptr && execution.replay != nullptr)
    throw std::invalid_argument("composition record and replay cannot both be set");
  execution.replay_index = 0;
  apply_device_fallback(image, pipeline, execution);
}

void sequential(MutableImageView image, const DevicePipeline& pipeline,
                const ExecutionContext& execution) {
  if (execution.record != nullptr && execution.replay != nullptr)
    throw std::invalid_argument("composition record and replay cannot both be set");
  execution.replay_index = 0;
  apply_device_fallback(image, pipeline, execution);
}

void identity(MutableImageView image, const ExecutionContext&) {
  // Identity/Noop touch no bytes, so they only require valid metadata.
  if (!image.valid()) throw std::invalid_argument("composition requires a valid image view");
}

void noop(MutableImageView image, const ExecutionContext& execution) {
  identity(image, execution);
}

void replay_compose(MutableImageView image, const CpuPipeline& pipeline,
                    const ExecutionContext& execution) {
  compose(image, pipeline, execution);
}

void replay_compose(MutableImageView image, const DevicePipeline& pipeline,
                    const ExecutionContext& execution) {
  compose(image, pipeline, execution);
}

void one_of(MutableImageView image, const std::vector<CpuPipeline>& choices,
            const ExecutionContext& execution) {
  const auto entry = choose(execution, SelectionKind::OneOf, choices.size());
  apply_cpu(image, choices[entry.selected], execution);
}

void one_of(MutableImageView image, const std::vector<DevicePipeline>& choices,
            const ExecutionContext& execution) {
  (void)image;
  (void)choices;
  (void)execution;
  throw std::invalid_argument("device composition requires the CUDA backend");
}

void one_or_other(MutableImageView image, const CpuPipeline& first,
                  const CpuPipeline& second, const ExecutionContext& execution) {
  const auto entry = choose(execution, SelectionKind::OneOrOther, 2);
  apply_cpu(image, entry.selected == 0 ? first : second, execution);
}

void one_or_other(MutableImageView image, const DevicePipeline& first,
                  const DevicePipeline& second, const ExecutionContext& execution) {
  (void)image;
  (void)first;
  (void)second;
  (void)execution;
  throw std::invalid_argument("device composition requires the CUDA backend");
}

void random_apply(MutableImageView image, const CpuPipeline& pipeline, double probability,
                  const ExecutionContext& execution) {
  validate_probability(probability);
  const auto entry = choose(execution, SelectionKind::RandomApply, 1, probability);
  if (entry.applied) apply_cpu(image, pipeline, execution);
}

void random_apply(MutableImageView image, const DevicePipeline& pipeline, double probability,
                  const ExecutionContext& execution) {
  validate_probability(probability);
  (void)choose(execution, SelectionKind::RandomApply, 1, probability);
  (void)image;
  (void)pipeline;
  (void)execution;
  throw std::invalid_argument("device composition requires the CUDA backend");
}

void some_of(MutableImageView image, const std::vector<CpuPipeline>& choices,
             std::size_t count, const ExecutionContext& execution, bool replace) {
  const auto entry = choose_indices(execution, SelectionKind::SomeOf, choices.size(), count, replace);
  for (const auto index : entry.selected_indices) apply_cpu(image, choices[index], execution);
}

void some_of(MutableImageView image, const std::vector<CpuPipeline>& choices,
             std::size_t count, bool replace, const ExecutionContext& execution) {
  some_of(image, choices, count, execution, replace);
}

void some_of(MutableImageView image, const std::vector<DevicePipeline>& choices,
             std::size_t count, const ExecutionContext& execution, bool replace) {
  (void)image;
  (void)choices;
  (void)count;
  (void)execution;
  (void)replace;
  throw std::invalid_argument("device composition requires the CUDA backend");
}

void some_of(MutableImageView image, const std::vector<DevicePipeline>& choices,
             std::size_t count, bool replace, const ExecutionContext& execution) {
  some_of(image, choices, count, execution, replace);
}

void random_order(MutableImageView image, const std::vector<CpuPipeline>& choices,
                  const ExecutionContext& execution) {
  const auto entry = choose_order(execution, choices.size());
  for (const auto index : entry.order) apply_cpu(image, choices[index], execution);
}

void random_order(MutableImageView image, const std::vector<DevicePipeline>& choices,
                  const ExecutionContext& execution) {
  (void)image;
  (void)choices;
  (void)execution;
  throw std::invalid_argument("device composition requires the CUDA backend");
}

void selective_channel_transform(MutableImageView image, const CpuPipeline& pipeline,
                                 ChannelSelection channels,
                                 const ExecutionContext& execution) {
  validate_cpu_image(image);
  apply_cpu(image, pipeline, execution, channels);
}

void selective_channel_transform(MutableImageView image, const DevicePipeline& pipeline,
                                 ChannelSelection channels,
                                 const ExecutionContext& execution) {
  (void)image;
  (void)pipeline;
  (void)channels;
  (void)execution;
  throw std::invalid_argument("device composition requires the CUDA backend");
}

void with_channels(MutableImageView image, const CpuPipeline& pipeline,
                   const std::vector<int>& channels,
                   const ExecutionContext& execution) {
  if (channels.empty()) {
    validate_cpu_image(image);
    return;
  }
  selective_channel_transform(image, pipeline, ChannelSelection{channels, {}}, execution);
}

void with_channels(MutableImageView image, const DevicePipeline& pipeline,
                   const std::vector<int>& channels,
                   const ExecutionContext& execution) {
  if (channels.empty()) {
    validate_device_image(image);
    return;
  }
  selective_channel_transform(image, pipeline, ChannelSelection{channels, {}}, execution);
}

void with_channels(MutableImageView image, const CpuPipeline& pipeline,
                   ChannelRange channels,
                   const ExecutionContext& execution) {
  selective_channel_transform(image, pipeline, ChannelSelection{{}, {channels}}, execution);
}

void with_channels(MutableImageView image, const DevicePipeline& pipeline,
                   ChannelRange channels,
                   const ExecutionContext& execution) {
  selective_channel_transform(image, pipeline, ChannelSelection{{}, {channels}}, execution);
}

void with_channels(MutableImageView image, const CpuPipeline& pipeline,
                   const ChannelSelection& channels,
                   const ExecutionContext& execution) {
  selective_channel_transform(image, pipeline, channels, execution);
}

void with_channels(MutableImageView image, const DevicePipeline& pipeline,
                   const ChannelSelection& channels,
                   const ExecutionContext& execution) {
  selective_channel_transform(image, pipeline, channels, execution);
}

}  // namespace augmatch
