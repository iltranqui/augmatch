#pragma once
#include <cstddef>
#include <cstdint>
#include "augmatch/canny.hpp"
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
enum class BlurMode : int { Average=0, Gaussian=1, Median=2 };
struct BlurConfig { int width=0,height=0,channels=0,kernel_size=3; float sigma=1.0f; BlurMode mode=BlurMode::Average; };
void blur_u8(const std::uint8_t*,std::uint8_t*,const BlurConfig&,cudaStream_t stream=nullptr);
// Deterministic bilateral blur. For each output pixel, the spatial Gaussian
// uses integer offsets in [-radius,radius]^2 and REFLECT_101 border samples.
// The range Gaussian uses the squared Euclidean difference of the complete
// interleaved pixel (all channels). The normalized weighted sum is rounded
// half-up and clamped to [0,255]. Radius is limited to 15 by both backends.
struct BilateralBlurConfig { int width=0,height=0,channels=0,radius=2; float sigma_space=1.0f,sigma_color=25.0f; };
void bilateral_blur_u8(const std::uint8_t*,std::uint8_t*,const BilateralBlurConfig&,cudaStream_t stream=nullptr);
// Deterministic fixed-iteration mean-shift blur. Each iteration examines the
// finite square spatial window around the original pixel, keeps samples whose
// complete-pixel Euclidean color distance from the current color is at most
// color_radius, and replaces the color with their arithmetic mean. Borders use
// REFLECT_101 and output uses clipped half-up uint8 rounding.
struct MeanShiftBlurConfig { int width=0,height=0,channels=0,spatial_radius=3; float color_radius=10.0f; int iterations=1; };
void mean_shift_blur_u8(const std::uint8_t*,std::uint8_t*,const MeanShiftBlurConfig&,cudaStream_t stream=nullptr);
// Deterministic anisotropic Gaussian blur. The normalized kernel is
// exp(-0.5 * ((x'/sigma_x)^2 + (y'/sigma_y)^2)), where (x', y') is the
// sample offset rotated by angle_degrees. Border samples use REFLECT_101.
struct AdvancedBlurConfig { int width=0,height=0,channels=0,kernel_size=5; float sigma_x=1.0f,sigma_y=1.0f,angle_degrees=0.0f; };
void advanced_blur_u8(const std::uint8_t*,std::uint8_t*,const AdvancedBlurConfig&,cudaStream_t stream=nullptr);
struct UnsharpConfig { int width=0,height=0,channels=0,kernel_size=3; float sigma=1,amount=0.5f; int threshold=0; };
void unsharp_mask_u8(const std::uint8_t*,std::uint8_t*,const UnsharpConfig&,cudaStream_t stream=nullptr);
struct SharpenConfig { int width=0,height=0,channels=0,kernel_size=3; float sigma=1,alpha=0.5f,lightness=1; };
void sharpen_u8(const std::uint8_t*,std::uint8_t*,const SharpenConfig&,cudaStream_t stream=nullptr);
// Deterministic Gaussian high-boost ringing/overshoot filter. It computes a
// normalized Gaussian blur with REFLECT_101 borders, then returns
// clamp(round(input + amount * (input - blur))). The finite odd kernel is
// applied independently to every channel; no random state is used.
struct RingingOvershootConfig { int width=0,height=0,channels=0,kernel_size=5; float sigma=1.0f,amount=0.5f; };
void ringing_overshoot_u8(const std::uint8_t*,std::uint8_t*,const RingingOvershootConfig&,cudaStream_t stream=nullptr);
struct MotionBlurConfig { int width=0,height=0,channels=0,kernel_size=3; float angle_degrees=0,direction=0; bool allow_shifted=true; };
void motion_blur_u8(const std::uint8_t*,std::uint8_t*,const MotionBlurConfig&,cudaStream_t stream=nullptr);
// NOISE_CATALOG optical names are separate from augmenter/filter names while
// reusing the same deterministic finite kernels and HWC uint8 contract.
struct OpticalMotionBlurConfig { int width=0,height=0,channels=0,kernel_size=3; float angle_degrees=0.0f,direction=0.0f; };
void optical_motion_blur_u8(const std::uint8_t*,std::uint8_t*,const OpticalMotionBlurConfig&,cudaStream_t stream=nullptr);
struct DefocusConfig { int width=0,height=0,channels=0,radius=3; float alias_blur=0; };
void defocus_u8(const std::uint8_t*,std::uint8_t*,const DefocusConfig&,cudaStream_t stream=nullptr);
struct OpticalDefocusConfig { int width=0,height=0,channels=0,radius=3; float alias_blur=0.0f; };
void optical_defocus_u8(const std::uint8_t*,std::uint8_t*,const OpticalDefocusConfig&,cudaStream_t stream=nullptr);
// Optical-artifact contracts use contiguous HWC uint8 input/output.  Borders
// use REFLECT_101, samples are uniform and output is saturating half-up uint8.
// Depth is borrowed H*W float32 (host for CPU, device for CUDA).  A pixel's
// disk radius is round(abs(depth-focus_depth)*blur_scale), clamped to
// [0,max_radius]; radius zero copies the source.  This is an explicit thin-
// lens surrogate, not a physically complete aperture/depth renderer.
struct DepthDependentDefocusConfig {
  int width=0,height=0,channels=0; const float* depth=nullptr;
  float focus_depth=0.5f,blur_scale=4.0f; int max_radius=16;
};
void depth_dependent_defocus_u8(const std::uint8_t*,std::uint8_t*,const DepthDependentDefocusConfig&,cudaStream_t stream=nullptr);
// offsets_x/y are borrowed camera translations in pixels, one pair per sample.
// The samples are averaged in listed order; zero is a valid offset and at least
// one sample is required.  Nearest sampling makes CPU/CUDA and clipping exact.
struct CameraShakeBlurConfig {
  int width=0,height=0,channels=0,sample_count=1;
  const float* offsets_x=nullptr; const float* offsets_y=nullptr;
};
void camera_shake_blur_u8(const std::uint8_t*,std::uint8_t*,const CameraShakeBlurConfig&,cudaStream_t stream=nullptr);
// Linear directional blur samples uniformly on the centered segment of the
// supplied length.  length is a pixel span and sample_count includes both
// endpoints; angle is degrees counter-clockwise from +x.
struct LinearDirectionalBlurConfig {
  int width=0,height=0,channels=0,sample_count=3;
  float length=3.0f,angle_degrees=0.0f;
};
void linear_directional_blur_u8(const std::uint8_t*,std::uint8_t*,const LinearDirectionalBlurConfig&,cudaStream_t stream=nullptr);
// Rotational blur averages samples rotated around (center_x,center_y) from
// -angle/2 through +angle/2.  The center defaults are not implicit: callers
// should supply image coordinates.  Nearest border-reflected sampling is the
// documented approximation to a spatially varying rotational PSF.
struct RotationalMotionBlurConfig {
  int width=0,height=0,channels=0,sample_count=3;
  float center_x=0.0f,center_y=0.0f,angle_degrees=0.0f;
};
void rotational_motion_blur_u8(const std::uint8_t*,std::uint8_t*,const RotationalMotionBlurConfig&,cudaStream_t stream=nullptr);
// Each row offset pair is a total pixel displacement during that row's readout
// interval.  samples uniformly cover -0.5..+0.5 of that displacement.  The
// H-element maps are borrowed (host CPU/device CUDA); endpoint samples are
// reflect-101 clamped and output is saturating half-up.
struct RollingShutterMotionBlurConfig {
  int width=0,height=0,channels=0,sample_count=3;
  const float* row_offsets_x=nullptr; const float* row_offsets_y=nullptr;
};
void rolling_shutter_motion_blur_u8(const std::uint8_t*,std::uint8_t*,const RollingShutterMotionBlurConfig&,cudaStream_t stream=nullptr);
// Deterministic finite PSF contracts for optical aperture artifacts. All four
// APIs consume contiguous HWC uint8 buffers, use REFLECT_101 borders, normalize
// the per-pixel PSF, then saturating half-up round to uint8. radius is an
// integer pixel support in [0,32]; radius zero is an exact copy. The models
// are discrete, wavelength-independent surrogates rather than calibrated lens
// simulations.
struct DiffractionBlurConfig {
  int width=0,height=0,channels=0,radius=3;
  float wavelength_nm=550.0f,aperture_diameter_mm=2.0f,focal_length_mm=50.0f;
};
void diffraction_blur_u8(const std::uint8_t*,std::uint8_t*,const DiffractionBlurConfig&,cudaStream_t stream=nullptr);
// Bokeh uses a uniform disk with optional linear edge apodization. edge_softness
// is the fractional radial transition [0,1], where zero is a hard disk.
struct BokehBlurConfig { int width=0,height=0,channels=0,radius=3; float edge_softness=0.0f; };
void bokeh_blur_u8(const std::uint8_t*,std::uint8_t*,const BokehBlurConfig&,cudaStream_t stream=nullptr);
// Cat-eye bokeh shrinks the horizontal semi-axis toward the normalized image
// boundary. cat_eye_strength [0,1] controls shrinkage; center coordinates are
// normalized [0,1] and are explicit to avoid an implicit optical centre.
struct CatEyeBokehConfig { int width=0,height=0,channels=0,radius=3; float cat_eye_strength=0.5f,center_x=0.5f,center_y=0.5f; };
void cat_eye_bokeh_u8(const std::uint8_t*,std::uint8_t*,const CatEyeBokehConfig&,cudaStream_t stream=nullptr);
// Aperture-shape blur uses a regular polygon with blades sides. rotation is in
// degrees and roundness [0,1] linearly blends polygon boundary with a circle.
struct ApertureShapeBlurConfig { int width=0,height=0,channels=0,radius=3,blades=6; float rotation_degrees=0.0f,roundness=0.0f; };
void aperture_shape_blur_u8(const std::uint8_t*,std::uint8_t*,const ApertureShapeBlurConfig&,cudaStream_t stream=nullptr);
// Deterministic imgaug convolutional filters. Each operation applies its fixed
// 3x3 stencil independently to every HWC channel and blends the filtered
// result with the source using alpha. Borders are OpenCV BORDER_REFLECT_101
// (the default used by cv::filter2D). Integer output uses OpenCV-compatible
// saturating half-up nearest-integer conversion, not truncation. Alpha is in [0,1].
// Emboss uses the imgaug stencil with the supplied nonnegative strength:
// [[-1-strength,-strength,0],[-strength,1,strength],[0,strength,1+strength]].
struct EmbossConfig { int width=0,height=0,channels=0; float alpha=1.0f,strength=1.0f; };
void emboss_u8(const std::uint8_t*,std::uint8_t*,const EmbossConfig&,cudaStream_t stream=nullptr);
// EdgeDetect uses [[0,1,0],[1,-4,1],[0,1,0]], alpha-blended with identity.
struct EdgeDetectConfig { int width=0,height=0,channels=0; float alpha=1.0f; };
void edge_detect_u8(const std::uint8_t*,std::uint8_t*,const EdgeDetectConfig&,cudaStream_t stream=nullptr);
// DirectedEdgeDetect follows imgaug's normalized direction convention:
// direction*360 degrees clockwise from the top. The directional 3x3 stencil
// is computed by the same angle-similarity formula as imgaug; direction is
// quantized to integer degrees before construction.
struct DirectedEdgeDetectConfig { int width=0,height=0,channels=0; float alpha=1.0f,direction=0.0f; };
void directed_edge_detect_u8(const std::uint8_t*,std::uint8_t*,const DirectedEdgeDetectConfig&,cudaStream_t stream=nullptr);
// ZoomBlur averages bilinear, center-preserving zoom samples. The deterministic
// factor set is min_factor + i * (max_factor - min_factor) / steps for
// i = 0..steps (uniform weights, including factor 1 when min_factor is 1).
// Borders use REFLECT_101. Unlike randomized Albumentations/imgaug contracts,
// this API never samples a range: callers choose the complete factor interval
// and number of equal-weight intervals explicitly.
struct ZoomBlurConfig {
  int width=0,height=0,channels=0;
  float min_factor=1.0f,max_factor=1.25f;
  int steps=5;
};
int zoom_blur_factor_count(const ZoomBlurConfig& config);
void zoom_blur_u8(const std::uint8_t*,std::uint8_t*,const ZoomBlurConfig&,cudaStream_t stream=nullptr);
struct OpticalZoomBlurConfig { int width=0,height=0,channels=0; float min_factor=1.0f,max_factor=1.25f; int steps=5; };
void optical_zoom_blur_u8(const std::uint8_t*,std::uint8_t*,const OpticalZoomBlurConfig&,cudaStream_t stream=nullptr);
// One GlassBlur swap offset in the deterministic sequence. The sequence is
// ordered by iteration, then y, then x over the interior [max_delta, size-
// max_delta) rectangle. Each entry swaps (x,y) with (x+dx,y+dy), for every
// channel. A null sequence is generated with the documented SplitMix64 seed.
struct GlassBlurSwap { std::int8_t dx=0,dy=0; };
static_assert(sizeof(GlassBlurSwap)==2,"GlassBlurSwap must remain a packed two-byte sequence entry");
struct GlassBlurConfig {
  int width=0,height=0,channels=0;
  float sigma=0.7f;
  int max_delta=1,iterations=1;
  const GlassBlurSwap* swap_sequence=nullptr;
  std::uint64_t seed=0;
};
// Number of entries required by GlassBlurConfig::swap_sequence.
std::size_t glass_blur_swap_count(int width,int height,int max_delta,int iterations);
// Fills a caller-owned host sequence. For each entry k, SplitMix64(seed+2*k)
// selects dx and SplitMix64(seed+2*k+1) selects dy uniformly from
// [-max_delta,max_delta].
void make_glass_blur_swap_sequence(GlassBlurSwap* sequence,int width,int height,int max_delta,int iterations,std::uint64_t seed);
void glass_blur_u8(const std::uint8_t*,std::uint8_t*,const GlassBlurConfig&,cudaStream_t stream=nullptr);
// Deterministic Non-Local Means denoising. For each output pixel, candidates
// are every pixel in the bounded square search window [-search_radius,
// search_radius]^2. Candidate weights are exp(-d/(h*h)), where d is the
// mean squared difference over the complete square patch of radius
// patch_radius and all channels. The center candidate is included, borders
// use REFLECT_101, and the normalized result is rounded to nearest uint8.
// The explicit bounds (patch_radius <= 4 and search_radius <= 8) keep work
// and CUDA register use bounded; h must be finite and strictly positive.
struct NonLocalMeansDenoisingConfig {
  int width=0,height=0,channels=0;
  int patch_radius=1,search_radius=3;
  float h=10.0f;
};
void non_local_means_denoising_u8(const std::uint8_t*,std::uint8_t*,const NonLocalMeansDenoisingConfig&,cudaStream_t stream=nullptr);
}
