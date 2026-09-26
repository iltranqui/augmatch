# Pipeline and composition

Headers: `include/augmatch/pipeline/`. Back to the [docs index](../README.md).

## Contents

- [Composition and control contract](#composition-and-control-contract)
- [`augmenters.meta` target-aware operations](#augmentersmeta-target-aware-operations)

## Composition and control contract

`composition.hpp` provides deterministic in-place `compose` and `sequential`
pipelines plus validation-preserving `identity` and `noop` operations. CPU
stages are borrowed synchronous callbacks receiving a mutable image view, a
read-only `ExecutionContext`, and caller-owned `void*` context; callbacks must
not retain either pointer. Stage order is preserved and the context seed is
never changed by the pipeline.

CUDA does not accept CPU callbacks for device images. `DeviceStage` is the
explicit boundary: its host launch function enqueues device work on the supplied
stream, and its context remains caller-owned until the call returns. A null
launch is an identity/no-op stage. Pipelines do not synchronize CUDA streams;
callers synchronize when completion is required.

`one_of`, `one_or_other`, `random_apply`, `some_of`, and `random_order` operate
on vectors or pipelines of `CpuStage`/`DeviceStage` representations. Each
decision uses `SplitMix64(seed + traversal_index)` and never consults a global
RNG. `SomeOf` records selected child indices and `RandomOrder` records a full
permutation. Set `ExecutionContext::record` to append explicit
`StageSelectionRecord` entries; set `ExecutionContext::replay` to validate and
reuse those entries. `replay_compose` is the explicit top-level replay boundary
and resets traversal at the start of a call. `ChannelRange`,
`ChannelSelection`, and `selective_channel_transform` pass borrowed channel
subviews to CPU callbacks or CUDA launch callbacks, retaining original strides
and avoiding copies. `with_channels` applies a child pipeline to explicit
imgaug-style channel indices or ranges. Discontiguous HWC selections invoke a
stage once per contiguous run. CPU and CUDA controls share the record format,
while device stages remain host launch callbacks.

## `augmenters.meta` target-aware operations

`meta.hpp` adds native replacements for Lambda, AssertLambda, AssertShape,
RemoveCBAsByOutOfImageFraction, and ClipCBAsToImagePlanes. `LambdaCallback`
and `AssertLambdaPredicate` are plain synchronous host function pointers, not
`std::function` objects: callback/predicate and `void*` context are borrowed,
owned by the caller, and must not be retained. A callback may mutate the image
and caller-owned box/keypoint arrays; a predicate receives const views. Null
callbacks, invalid views, and null arrays with nonzero counts throw
`std::invalid_argument`. Device image views are rejected explicitly. This is
intentional: there is no safe ABI for arbitrary host callbacks or predicates in
a CUDA kernel. Device pipelines must use `DeviceStage` launch functions.

`AssertShapeConfig` uses `-1` as a wildcard dimension and can independently
check datatype and layout. It never changes image bytes. Target metadata is
represented by borrowed `MutableTargetAnnotations`/`TargetAnnotations`; the
library never allocates, frees, or resizes those arrays.

`remove_cbas_by_out_of_image_fraction` compacts boxes and keypoints in place
and returns both retained counts. A box's out-of-image fraction is one minus
its clipped-area/original-area ratio; malformed boxes count as fully outside.
A keypoint is inside when its finite pixel centre is in `[0,width) x
[0,height)`, otherwise it has fraction one. Optional source-index arrays let a
caller compact labels in lockstep. The threshold is finite and in `[0,1]`.
`clip_cbas_to_image_planes` clips box edges to `[0,width] x [0,height]` and
keypoint centres to `[0,width-1] x [0,height-1]`; non-finite keypoints are an
error. Metadata operations are host-only in CPU and CUDA builds.

The CLI metadata forms are:

```text
augmatch_cli meta_clip BOXES_IN BOXES_OUT KEYPOINTS_IN KEYPOINTS_OUT WIDTH HEIGHT
augmatch_cli meta_remove BOXES_IN BOXES_OUT KEYPOINTS_IN KEYPOINTS_OUT WIDTH HEIGHT MAX_FRACTION
```

The files contain whitespace-separated `x1 y1 x2 y2` boxes and `x y`
keypoints. `examples/meta_example.cpp` demonstrates callback ownership and
annotation clipping.
