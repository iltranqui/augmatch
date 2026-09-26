#include "augmatch/composition.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
void add_one(augmatch::MutableImageView image, const augmatch::ExecutionContext&, void*) {
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  for (int i = 0; i < image.width * image.height * image.channels; ++i) ++bytes[i];
}
void add_ten(augmatch::MutableImageView image, const augmatch::ExecutionContext&, void*) {
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  for (int i = 0; i < image.width * image.height * image.channels; ++i) bytes[i] += 10;
}
void multiply_two(augmatch::MutableImageView image, const augmatch::ExecutionContext&, void*) {
  auto* bytes = static_cast<std::uint8_t*>(image.data);
  for (int i = 0; i < image.width * image.height * image.channels; ++i) bytes[i] *= 2;
}

augmatch::CpuPipeline pipeline(augmatch::CpuStageCallback callback) {
  return augmatch::make_compose(std::vector<augmatch::CpuStage>{{callback, nullptr}});
}
}  // namespace

int main() {
  constexpr int width = 4;
  std::vector<std::uint8_t> source{1, 2, 3, 4};
  const auto view = [&](std::vector<std::uint8_t>& pixels) {
    return augmatch::make_hwc_u8_view(pixels.data(), width, 1, 1);
  };
  const auto choices = std::vector<augmatch::CpuPipeline>{pipeline(add_one), pipeline(add_ten),
                                                            pipeline(multiply_two)};

  augmatch::ReplayRecord record;
  augmatch::ExecutionContext recording{nullptr, 91, true, &record, nullptr};
  auto recorded_image = source;
  augmatch::one_of(view(recorded_image), choices, recording);
  augmatch::random_apply(view(recorded_image), pipeline(add_one), 1.0, recording);
  if (record.selections.size() != 2 || record.selections[0].kind != augmatch::SelectionKind::OneOf ||
      record.selections[1].kind != augmatch::SelectionKind::RandomApply)
    return 1;

  auto replayed_image = source;
  augmatch::ExecutionContext replaying{nullptr, 999, true, nullptr, &record};
  augmatch::replay_compose(view(replayed_image), augmatch::CpuPipeline{}, replaying);
  // ReplayCompose itself has no implicit random decision; replaying the two
  // recorded control nodes explicitly must reproduce the recorded result.
  replaying.replay_index = 0;
  augmatch::one_of(view(replayed_image), choices, replaying);
  augmatch::random_apply(view(replayed_image), pipeline(add_one), 0.0, replaying);
  if (replayed_image != recorded_image) return 2;

  auto first = source;
  auto second = source;
  augmatch::ExecutionContext deterministic_first{nullptr, 17, true};
  augmatch::ExecutionContext deterministic_second{nullptr, 17, true};
  augmatch::one_or_other(view(first), pipeline(add_one), pipeline(add_ten), deterministic_first);
  augmatch::one_or_other(view(second), pipeline(add_one), pipeline(add_ten), deterministic_second);
  if (first != second) return 3;

  // Property-style check: every explicit seed produces a replayable result,
  // including both branches and both random-apply outcomes.
  for (std::uint64_t seed = 0; seed < 64; ++seed) {
    augmatch::ReplayRecord property_record;
    augmatch::ExecutionContext property_recording{nullptr, seed, true, &property_record, nullptr};
    auto property_source = source;
    augmatch::one_of(view(property_source), choices, property_recording);
    augmatch::random_apply(view(property_source), pipeline(add_one), 0.5, property_recording);
    auto property_replay = source;
    augmatch::ExecutionContext property_replaying{nullptr, seed + 1, true, nullptr,
                                                   &property_record};
    augmatch::one_of(view(property_replay), choices, property_replaying);
    augmatch::random_apply(view(property_replay), pipeline(add_one), 0.0, property_replaying);
    if (property_source != property_replay || property_record.selections.size() != 2) return 6;
  }

  auto skipped = source;
  augmatch::random_apply(view(skipped), pipeline(add_ten), 0.0, deterministic_first);
  if (skipped != source) return 4;

  bool rejected_empty = false;
  try {
    augmatch::one_of(view(skipped), std::vector<augmatch::CpuPipeline>{}, deterministic_first);
  } catch (const std::invalid_argument&) {
    rejected_empty = true;
  }
  if (!rejected_empty) return 5;
  return 0;
}
