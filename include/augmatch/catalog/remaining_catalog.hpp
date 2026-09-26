#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "augmatch/color/colorspace.hpp"
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
// The remaining imgaug catalog operations use contiguous interleaved HWC uint8.
// These APIs are deterministic for a fixed config and seed.  They are host-only
// even in a CUDA build: pointers must be host pointers and stream is ignored.
// This explicit contract is used for callback/file and borrowed-record APIs
// where silently dereferencing device memory would be unsafe.

// Adds Laplace-distributed noise with the given scale, keyed by seed.
struct AdditiveLaplaceNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float scale = 0.0f;
  std::uint64_t seed = 0;
};
void additive_laplace_noise_u8(const std::uint8_t* input, std::uint8_t* output, const AdditiveLaplaceNoiseConfig& config, cudaStream_t stream = nullptr);

// Adds Poisson-distributed noise with the given scale, keyed by seed.
struct AdditivePoissonNoiseConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float scale = 1.0f;
  std::uint64_t seed = 0;
};
void additive_poisson_noise_u8(const std::uint8_t* input, std::uint8_t* output, const AdditivePoissonNoiseConfig& config, cudaStream_t stream = nullptr);

// Cartoon effect: blurs by blur_radius, then darkens edges whose gradient
// exceeds edge_threshold to edge_value.
struct CartoonConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int blur_radius = 1;
  float edge_threshold = 24.0f;
  std::uint8_t edge_value = 0;
};
void cartoon_u8(const std::uint8_t* input, std::uint8_t* output, const CartoonConfig& config, cudaStream_t stream = nullptr);

// Applies count RandAugment-style operations at the given magnitude, keyed by seed.
struct RandAugmentConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int count = 2;
  float magnitude = 0.25f;
  std::uint64_t seed = 0;
};
void rand_augment_u8(const std::uint8_t* input, std::uint8_t* output, const RandAugmentConfig& config, cudaStream_t stream = nullptr);
using CatalogColorSpace = ColorSpace;
// Converts pixels from source to destination colorspace.
struct ChangeColorspaceConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  CatalogColorSpace source = CatalogColorSpace::RGB;
  CatalogColorSpace destination = CatalogColorSpace::HSV;
};
void change_colorspace_u8(const std::uint8_t* input, std::uint8_t* output, const ChangeColorspaceConfig& config, cudaStream_t stream = nullptr);

// Quantizes RGB colors to the given number of clusters using seeded k-means
// over the given number of iterations.
struct KMeansColorQuantizationConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int clusters = 8;
  int iterations = 8;
  std::uint64_t seed = 0;
};
void kmeans_color_quantization_u8(const std::uint8_t* input, std::uint8_t* output, const KMeansColorQuantizationConfig& config, cudaStream_t stream = nullptr);

// Convolves with a borrowed kernel_width x kernel_height kernel, dividing by
// divisor (0 means the kernel sum) and adding bias.
struct ConvolveConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int kernel_width = 0;
  int kernel_height = 0;
  const float* kernel = nullptr;
  float divisor = 0.0f;
  float bias = 0.0f;
};
void convolve_u8(const std::uint8_t* input, std::uint8_t* output, const ConvolveConfig& config, cudaStream_t stream = nullptr);
struct SaveDebugImageEveryNBatchesConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int batch_index = 0;
  int every_n_batches = 1;
  const char* output_path = nullptr;
};
// Writes a PGM/PPM snapshot only when batch_index is a multiple of every_n_batches;
// no callback or asynchronous file API is accepted. The output image is copied
// exactly and the output buffer is always populated.
void save_debug_image_every_n_batches_u8(const std::uint8_t* input, std::uint8_t* output, const SaveDebugImageEveryNBatchesConfig& config, cudaStream_t stream = nullptr);

// Applies a polar warp about (center_x,center_y) (normalized), scaling angle
// and radius by angle_scale/radius_scale; out-of-bounds samples use fill.
struct WithPolarWarpingConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  float center_x = 0.5f;
  float center_y = 0.5f;
  float angle_scale = 1.0f;
  float radius_scale = 1.0f;
  std::uint8_t fill = 0;
};
void with_polar_warping_u8(const std::uint8_t* input, std::uint8_t* output, const WithPolarWarpingConfig& config, cudaStream_t stream = nullptr);

// Shuffles a grid_rows x grid_cols grid of tiles according to permutation, or
// a seeded deterministic permutation when permutation is null.
struct JigsawConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  int grid_rows = 2;
  int grid_cols = 2;
  const std::uint32_t* permutation = nullptr;
  std::uint64_t seed = 0;
};
void jigsaw_u8(const std::uint8_t* input, std::uint8_t* output, const JigsawConfig& config, cudaStream_t stream = nullptr);

// Fourier Domain Adaptation: a low-frequency color-statistics approximation
// (not a full FFT amplitude swap) toward a borrowed reference image, using
// low_frequency_radius as the swap radius.
struct FDAConfig {
  int width = 0;
  int height = 0;
  int channels = 0;
  const std::uint8_t* reference = nullptr;
  int reference_width = 0;
  int reference_height = 0;
  float low_frequency_radius = 0.1f;
};
void fda_u8(const std::uint8_t* input, std::uint8_t* output, const FDAConfig& config, cudaStream_t stream = nullptr);
using FourierDomainAdaptationConfig = FDAConfig;
void fourier_domain_adaptation_u8(const std::uint8_t* input, std::uint8_t* output, const FourierDomainAdaptationConfig& config, cudaStream_t stream = nullptr);
}
