# Albumentations and imgaug Transform Catalog

This is the implementation checklist for Augmatch. Names are grouped by upstream library and category. `[ ]` means not yet ported, `[~]` means partially implemented, and `[x]` means implemented.

Reference targets:

- Albumentations `2.0.8`
- imgaug `0.4.0`

## Albumentations

### Composition and control

- [x] `Compose`
- [x] `ReplayCompose`
- [x] `OneOf`
- [x] `OneOrOther`
- [x] `SomeOf`
- [x] `RandomOrder`
- [x] `Sequential`
- [x] `RandomApply`
- [x] `SelectiveChannelTransform`

### Cropping, padding, and resizing

- [~] `CenterCrop`
- [~] `Crop`
- [~] `CropAndPad`
- [x] `CropNonEmptyMaskIfExists` (CPU/CUDA image+mask crop; deterministic target selection and explicit fallback)
- [~] `RandomCrop`
- [x] `RandomCropFromBorders`
- [x] `RandomResizedCrop`
- [x] `RandomSizedCrop`
- [x] `RandomSizedBBoxSafeCrop` (CPU/CUDA; deterministic explicit Pascal-VOC box arrays, safe source metadata, resize, and explicit annotation ownership)
- [x] `BBoxSafeRandomCrop` (CPU/CUDA; deterministic all-box envelope, empty-box fallback, source metadata, and explicit annotation ownership)
- [x] `AtLeastOneBBoxRandomCrop` (CPU/CUDA; deterministic selected-box intersection, empty-box fallback, source metadata, and explicit annotation ownership)
- [~] `Pad`
- [~] `PadIfNeeded`
- [~] `LongestMaxSize`
- [~] `SmallestMaxSize`
- [~] `Resize`
- [~] `SquareSymmetricPad`

### Geometric transforms

- [~] `Affine`
- [~] `ElasticTransform`
- [~] `GridDistortion`
- [~] `PiecewiseAffine`
- [~] `Perspective`
- [~] `OpticalDistortion`
- [~] `RandomGridShuffle`
- [~] `RandomScale`
- [~] `Rotate`
- [~] `SafeRotate`
- [~] `ShiftScaleRotate`
- [~] `Transpose`
- [~] `HorizontalFlip`
- [~] `VerticalFlip`
- [~] `Flip`

### Blur and convolution

- [~] `Blur`
- [~] `GaussianBlur`
- [~] `MedianBlur`
- [~] `MotionBlur`
- [~] `AdvancedBlur`
- [~] `Defocus`
- [~] `GlassBlur`
- [~] `MedianBlur`
- [~] `ZoomBlur`
- [~] `UnsharpMask`
- [~] `Sharpen`
- [~] `RingingOvershoot`
- [~] `Superpixels`
- [~] `NonLocalMeansDenoising`

### Noise and dropout

- [~] `GaussNoise`
- [x] `ISONoise`
- [~] `MultiplicativeNoise`
- [~] `ShotNoise`
- [x] `SaltAndPepper`
- [~] `CoarseDropout`
- [~] `GridDropout`
- [x] `MaskDropout`
- [x] `XYMasking`
- [~] `ChannelDropout`
- [~] `PixelDropout`

### Color and intensity

- [~] `RandomBrightnessContrast`
- [~] `RandomGamma`
- [~] `RandomToneCurve`
- [~] `ColorJitter`
- [~] `HueSaturationValue`
- [~] `RGBShift`
- [~] `ToGray`
- [~] `ToRGB`
- [~] `ToSepia`
- [~] `PlanckianJitter`
- [~] `ChromaticAberration`
- [~] `FancyPCA`
- [x] `CLAHE`
- [~] `Equalize`
- [~] `AutoContrast`
- [~] `Posterize`
- [~] `Solarize`
- [~] `InvertImg`
- [~] `ChannelShuffle`
- [~] `RandomBrightnessContrast`
- [~] `RandomColorJitter`
- [x] `Dithering`
- [~] `PlasmaBrightnessContrast`
- [~] `PlasmaContrast`
- [~] `PlasmaShadow`

### Weather and atmosphere

- [~] `RandomRain` (CPU/CUDA; explicit deterministic streak records or seed-derived placement; sequential white alpha blend)
- [~] `RandomSnow` (CPU/CUDA; explicit deterministic snowflake records or seed-derived placement; pixel-centre disc rasterization and sequential white alpha blend)
- [x] `RandomFog` (CPU/CUDA; explicit width*height density field or deterministic SplitMix64 seed; depth-independent white veil blend)
- [~] `RandomSunFlare` (CPU/CUDA; explicit source and ray records or deterministic SplitMix64 ray placement; sequential white disc/capsule alpha blend)
- [~] `RandomShadow` (partial; CPU/CUDA explicit polygon and rectangle masks with deterministic pixel-centre alpha blending)
- [x] `RandomGravel` (CPU/CUDA; explicit deterministic filled-disc particle records or SplitMix64 seed placement; sequential scalar fill alpha blend)
- [x] `Spatter` (CPU/CUDA; explicit deterministic colored droplet records or SplitMix64 seed placement; pixel-centre disc rasterization; sequential per-channel color alpha blend)
- [x] `SnowStamp` (CPU/CUDA; explicit uint8 stamp image/mask and ordered integer placements; clipped sequential alpha/fill blending)

### Compression and image corruption

- [~] `ImageCompression` (libjpeg-turbo CPU codec; CUDA uses documented host codec fallback)
- [~] `JpegCompression` (libjpeg-turbo CPU codec; CUDA uses documented host codec fallback)
- [~] `WebPCompression` (libwebp native CPU codec with explicit lossy/lossless entry points; CUDA uses documented host codec fallback; conditional on libwebp with clear unavailable fallback)
- [x] `Downscale` (CPU/CUDA deterministic resize round trip; no codec-dependent compression)
- [x] `TemplateTransform` (shared weighted CPU/CUDA mixing API)
- [~] `FDA` (host-only deterministic low-frequency color-statistics approximation)
- [~] `FourierDomainAdaptation` (host-only alias of FDA approximation)
- [~] `RingingOvershoot`

### Segmentation and region transforms

- [~] `Superpixels`
- [x] `RandomCropNearBBox` (CPU/CUDA; explicit Pascal VOC box, deterministic margins, and explicit crop metadata; no hidden Python targets)
- [x] `CropNonEmptyMaskIfExists` (CPU/CUDA image+mask crop; deterministic target selection and explicit fallback)
- [x] `MaskDropout`
- [x] `Mosaic` (CPU/CUDA; four explicit HWC uint8 sources and deterministic split)
- [x] `OverlayElements` (CPU/CUDA; explicit clipped source records and alpha masks)

### Mixing and multi-image transforms

- [x] `MixUp` (CPU/CUDA; explicit weight or seed-derived weight)
- [x] `CutMix` (CPU/CUDA; explicit half-open box and target clipping)
- [x] `Mosaic` (CPU/CUDA; four explicit HWC uint8 sources and deterministic split)
- [x] `TemplateTransform` (CPU/CUDA; weighted template records)
- [x] `OverlayElements` (CPU/CUDA; explicit clipped source records and alpha masks)

### Normalization and datatype transforms

- [~] `Normalize`
- [~] `FromFloat`
- [~] `ToFloat`
- [x] `ToTensorV2` (native contiguous HWC uint8 to CHW float32 tensor-buffer conversion; optional normalization; no PyTorch dependency)
- [x] `ToTensor3D` (native contiguous DHWC uint8 to CDHW float32 tensor-buffer conversion; optional normalization; no PyTorch dependency)

### Target-aware behavior to implement

- [~] Images (Geometry crop/flip helper preserves existing `transform_u8`)
- [~] Masks (`transform_mask_u8`, explicit nearest/linear policy)
- [~] Bounding boxes (`transform_boxes`, clipping and filtering)
- [~] Keypoints (`transform_points`, continuous centre coordinates)
- [~] Additional image targets (host strided view pairs, shared Geometry)
- [~] Additional mask targets (host strided view pairs, explicit interpolation policy)
- [~] Pascal VOC boxes (absolute continuous half-open edges)
- [~] COCO boxes (absolute x/y/width/height conversion and transform)
- [~] YOLO boxes (normalized center/width/height conversion and transform)
- [~] Albumentations normalized boxes (normalized xyxy conversion and transform)
- [~] Keypoint visibility (normalized [0,1] visibility with out-of-frame policy)
- [~] Box clipping and filtering
- [~] Mask interpolation policies

## imgaug

### `augmenters.meta`

- [x] `Sequential`
- [x] `SomeOf`
- [x] `OneOf`
- [x] `Sometimes`
- [x] `WithChannels` (native CPU/CUDA channel-index/range stage contract)
- [x] `Identity`
- [x] `Noop`
- [~] `Lambda` (borrowed synchronous host callback with target-aware box/keypoint views)
- [~] `AssertLambda` (borrowed synchronous host predicate with explicit failure)
- [~] `AssertShape` (wildcard dimension/type/layout validation)
- [~] `ChannelShuffle`
- [~] `RemoveCBAsByOutOfImageFraction` (in-place box/keypoint compaction and source indices)
- [~] `ClipCBAsToImagePlanes` (continuous box-edge and pixel-centre clipping)

### `augmenters.arithmetic`

- [~] `Add`
- [~] `AddElementwise` (explicit per-element values or deterministic seeded ranges)
- [~] `AdditiveGaussianNoise`
- [~] `AdditiveLaplaceNoise` (deterministic host-only HWC uint8 API)
- [~] `AdditivePoissonNoise` (deterministic host-only HWC uint8 API)
- [~] `Multiply`
- [~] `MultiplyElementwise` (explicit per-element values or deterministic seeded ranges)
- [x] `Cutout` (explicit ordered half-open rectangle list and scalar fill)
- [~] `Dropout`
- [~] `CoarseDropout`
- [~] `Dropout2D`
- [x] `TotalDropout` (explicit one-byte mask or deterministic seed/probability decision and scalar fill)
- [~] `ReplaceElementwise` (explicit mask/value arrays or deterministic seeded Bernoulli replacement)
- [~] `ImpulseNoise` (explicit mask/value arrays or deterministic seeded impulse parameters)
- [x] `SaltAndPepper` (CPU/CUDA deterministic pixel masks or seeded Bernoulli blocks)
- [x] `CoarseSaltAndPepper` (CPU/CUDA deterministic rectangle masks or seeded block decisions)
- [x] `Salt` (CPU/CUDA deterministic pixel masks or seeded Bernoulli decisions)
- [x] `CoarseSalt` (CPU/CUDA deterministic rectangle masks or seeded block decisions)
- [x] `Pepper` (CPU/CUDA deterministic pixel masks or seeded Bernoulli decisions)
- [x] `CoarsePepper` (CPU/CUDA deterministic rectangle masks or seeded block decisions)
- [~] `Invert`
- [~] `Solarize`
- [~] `JpegCompression` (libjpeg-turbo CPU codec; CUDA uses documented host codec fallback)

### `augmenters.artistic`

- [x] `Cartoon` (deterministic mean-smoothing and edge-mask approximation)

### `augmenters.blend`

- [x] `BlendAlpha`
- [x] `BlendAlphaMask`
- [x] `BlendAlphaElementwise`
- [x] `BlendAlphaSimplexNoise`
- [x] `BlendAlphaFrequencyNoise`
- [x] `BlendAlphaSomeColors`
- [x] `BlendAlphaHorizontalLinearGradient`
- [x] `BlendAlphaVerticalLinearGradient`
- [x] `BlendAlphaRegularGrid`
- [x] `BlendAlphaCheckerboard`
- [x] `BlendAlphaSegMapClassIds`
- [x] `BlendAlphaBoundingBoxes`

### `augmenters.blur`

- [~] `GaussianBlur`
- [~] `AverageBlur`
- [~] `MedianBlur`
- [x] `BilateralBlur`
- [~] `MotionBlur`
- [x] `MeanShiftBlur`

### `augmenters.collections`

- [x] `RandAugment` (deterministic fixed operation set and magnitude)

### `augmenters.color`

- [~] `WithColorspace`
- [~] `WithBrightnessChannels`
- [~] `MultiplyAndAddToBrightness`
- [~] `MultiplyBrightness`
- [~] `AddToBrightness`
- [~] `WithHueAndSaturation`
- [~] `MultiplyHueAndSaturation`
- [~] `MultiplyHue`
- [~] `MultiplySaturation`
- [~] `RemoveSaturation`
- [~] `AddToHueAndSaturation`
- [~] `AddToHue`
- [~] `AddToSaturation`
- [x] `ChangeColorspace` (RGB/HSV native conversion; LAB is explicit copy approximation)
- [~] `Grayscale`
- [~] `ChangeColorTemperature`
- [x] `KMeansColorQuantization` (deterministic seeded RGB k-means)
- [x] `UniformColorQuantization`
- [x] `UniformColorQuantizationToNBits`
- [~] `Posterize`

### `augmenters.contrast`

- [~] `GammaContrast`
- [~] `SigmoidContrast`
- [~] `LogContrast`
- [~] `LinearContrast`
- [x] `AllChannelsCLAHE`
- [x] `CLAHE`
- [x] `AllChannelsHistogramEqualization`
- [~] `HistogramEqualization`

### `augmenters.convolutional`

- [x] `Convolve` (borrowed host kernel, reflected border)
- [~] `Sharpen`
- [~] `Emboss` (deterministic explicit alpha/strength; CPU/CUDA)
- [~] `EdgeDetect` (deterministic explicit alpha; CPU/CUDA)
- [~] `DirectedEdgeDetect` (deterministic explicit alpha/direction; CPU/CUDA)

### `augmenters.debug`

- [x] `SaveDebugImageEveryNBatches` (synchronous PGM/PPM host file contract; callbacks unsupported)

### `augmenters.edges`

- [x] `Canny`

### `augmenters.flip`

- [~] `HorizontalFlip`
- [~] `VerticalFlip`
- [~] `Fliplr`
- [~] `Flipud`

### `augmenters.geometric`

- [~] `Affine`
- [~] `ScaleX`
- [~] `ScaleY`
- [~] `TranslateX`
- [~] `TranslateY`
- [~] `Rotate`
- [~] `ShearX`
- [~] `ShearY`
- [~] `PiecewiseAffine`
- [~] `PerspectiveTransform`
- [~] `ElasticTransformation`
- [~] `Rot90`
- [x] `WithPolarWarping` (deterministic nearest-neighbour polar remap approximation)
- [x] `Jigsaw` (explicit or seeded bijective tile permutation)

### `augmenters.imgcorruptlike`

- [~] `GaussianNoise`
- [~] `ShotNoise`
- [~] `ImpulseNoise` (explicit mask/value arrays or deterministic seeded impulse parameters)
- [~] `SpeckleNoise` (deterministic native multiplicative Gaussian approximation)
- [~] `GaussianBlur`
- [~] `GlassBlur`
- [~] `DefocusBlur`
- [~] `MotionBlur`
- [~] `ZoomBlur`
- [~] `Fog` (deterministic explicit/seeded white veil approximation)
- [~] `Frost` (deterministic cold white/blue veil approximation)
- [~] `Snow` (deterministic seeded sparse white accumulation)
- [~] `Spatter`
- [~] `Contrast` (deterministic midpoint contrast approximation)
- [~] `Brightness` (deterministic multiplicative brightness approximation)
- [~] `Saturate` (deterministic HSV saturation approximation)
- [~] `JpegCompression` (libjpeg-turbo CPU codec; CUDA uses documented host codec fallback)
- [~] `Pixelate` (deterministic nearest block-centre sampling)
- [~] `ElasticTransform`

### `augmenters.pillike`

- [~] `Solarize`
- [~] `Posterize`
- [~] `Equalize`
- [~] `Autocontrast`
- [x] `EnhanceColor` (deterministic explicit factor; Pillow parity)
- [x] `EnhanceContrast` (deterministic explicit factor; Pillow parity)
- [x] `EnhanceBrightness` (deterministic explicit factor; Pillow parity)
- [~] `EnhanceSharpness` (deterministic explicit factor; Pillow parity)
- [x] `FilterBlur` (Pillow 5x5 kernel and unchanged borders)
- [x] `FilterSmooth` (Pillow 3x3 kernel and unchanged borders)
- [x] `FilterSmoothMore` (Pillow 5x5 kernel and unchanged borders)
- [x] `FilterEdgeEnhance` (Pillow integer kernel, unchanged borders, alpha filtered)
- [x] `FilterEdgeEnhanceMore` (Pillow integer kernel, unchanged borders, alpha filtered)
- [x] `FilterFindEdges` (Pillow integer kernel, unchanged borders, alpha filtered)
- [x] `FilterContour` (Pillow integer kernel/offset, unchanged borders, alpha filtered)
- [x] `FilterEmboss` (Pillow integer kernel/offset, unchanged borders, alpha filtered)
- [x] `FilterSharpen` (Pillow integer kernel, unchanged borders, alpha filtered)
- [x] `FilterDetail` (Pillow integer kernel, unchanged borders, alpha filtered)
- [~] `Affine`

### `augmenters.pooling`

- [~] `AveragePooling`
- [~] `MaxPooling`
- [~] `MinPooling`
- [~] `MedianPooling`

### `augmenters.segmentation`

- [~] `Superpixels`
- [x] `Voronoi` (seeded nearest-site target-label map)
- [x] `UniformVoronoi` (seeded jittered stratified sites)
- [x] `RegularGridVoronoi` (explicit grid-count sites)
- [x] `RelativeRegularGridVoronoi` (fractional cell-size grid sites)

### `augmenters.size`

- [~] `Resize`
- [~] `CropAndPad`
- [~] `Pad`
- [~] `Crop`
- [~] `PadToFixedSize`
- [~] `CropToFixedSize`
- [x] `PadToMultiplesOf`
- [x] `CropToMultiplesOf`
- [x] `CropToPowersOf`
- [x] `PadToPowersOf`
- [x] `CropToAspectRatio`
- [x] `PadToAspectRatio`
- [~] `CropToSquare`
- [~] `PadToSquare`
- [~] `CenterPadToFixedSize`
- [~] `CenterCropToFixedSize`
- [x] `CenterCropToMultiplesOf`
- [x] `CenterPadToMultiplesOf`
- [x] `CenterCropToPowersOf`
- [x] `CenterPadToPowersOf`
- [x] `CenterCropToAspectRatio`
- [x] `CenterPadToAspectRatio`
- [~] `CenterCropToSquare`
- [~] `CenterPadToSquare`
- [x] `KeepSizeByResize`

### `augmenters.weather`

- [x] `FastSnowyLandscape` (CPU/CUDA, brightness-threshold white approximation)
- [x] `Clouds` (CPU/CUDA, seeded counter-field approximation)
- [x] `Fog` (CPU/CUDA, depth-independent white veil)
- [x] `CloudLayer` (CPU/CUDA, explicit borrowed HxW alpha map)
- [x] `Snowflakes` (CPU/CUDA, seeded/explicit disc records)
- [x] `SnowflakesLayer` (CPU/CUDA, explicit borrowed HxW alpha map)
- [x] `Rain` (CPU/CUDA, seeded/explicit streak records)
- [x] `RainLayer` (CPU/CUDA, explicit borrowed HxW alpha map)

## Cross-library implementation requirements

- [x] Common transform parameter representation.
- [x] Shared seeded RNG.
- [x] CPU reference implementation.
- [x] CUDA implementation.
- [x] C++17 API.
- [x] C++23 compatibility.
- [x] HWC and CHW layouts.
- [x] `uint8`, `uint16`, and `float32` images.
- [x] Masks and segmentation maps.
- [x] Bounding boxes.
- [x] Keypoints.
- [~] Polygons (blocked: no public polygon record/ring API).
- [~] Line strings (blocked: no ordered line-string API).
- [~] Heatmaps (blocked: no heatmap target semantics/API).
- [x] Exact parity vectors.
- [x] Statistical tests for stochastic transforms.
- [x] Property tests for geometry.
- [x] CPU/CUDA tolerance definitions.
- [x] CMake install/export targets.
- [x] Benchmarks.
- [x] Documentation and examples.
