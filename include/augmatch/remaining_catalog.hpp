#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "augmatch/colorspace.hpp"
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
struct AdditiveLaplaceNoiseConfig { int width=0,height=0,channels=0; float scale=0.0f; std::uint64_t seed=0; };
void additive_laplace_noise_u8(const std::uint8_t*,std::uint8_t*,const AdditiveLaplaceNoiseConfig&,cudaStream_t=nullptr);
struct AdditivePoissonNoiseConfig { int width=0,height=0,channels=0; float scale=1.0f; std::uint64_t seed=0; };
void additive_poisson_noise_u8(const std::uint8_t*,std::uint8_t*,const AdditivePoissonNoiseConfig&,cudaStream_t=nullptr);
struct CartoonConfig { int width=0,height=0,channels=0; int blur_radius=1; float edge_threshold=24.0f; std::uint8_t edge_value=0; };
void cartoon_u8(const std::uint8_t*,std::uint8_t*,const CartoonConfig&,cudaStream_t=nullptr);
struct RandAugmentConfig { int width=0,height=0,channels=0; int count=2; float magnitude=0.25f; std::uint64_t seed=0; };
void rand_augment_u8(const std::uint8_t*,std::uint8_t*,const RandAugmentConfig&,cudaStream_t=nullptr);
using CatalogColorSpace = ColorSpace;
struct ChangeColorspaceConfig { int width=0,height=0,channels=0; CatalogColorSpace source=CatalogColorSpace::RGB, destination=CatalogColorSpace::HSV; };
void change_colorspace_u8(const std::uint8_t*,std::uint8_t*,const ChangeColorspaceConfig&,cudaStream_t=nullptr);
struct KMeansColorQuantizationConfig { int width=0,height=0,channels=0; int clusters=8,iterations=8; std::uint64_t seed=0; };
void kmeans_color_quantization_u8(const std::uint8_t*,std::uint8_t*,const KMeansColorQuantizationConfig&,cudaStream_t=nullptr);
struct ConvolveConfig { int width=0,height=0,channels=0,kernel_width=0,kernel_height=0; const float* kernel=nullptr; float divisor=0.0f,bias=0.0f; };
void convolve_u8(const std::uint8_t*,std::uint8_t*,const ConvolveConfig&,cudaStream_t=nullptr);
struct SaveDebugImageEveryNBatchesConfig { int width=0,height=0,channels=0,batch_index=0,every_n_batches=1; const char* output_path=nullptr; };
// Writes a PGM/PPM snapshot only when batch_index is a multiple of every_n_batches;
// no callback or asynchronous file API is accepted. The output image is copied
// exactly and the output buffer is always populated.
void save_debug_image_every_n_batches_u8(const std::uint8_t*,std::uint8_t*,const SaveDebugImageEveryNBatchesConfig&,cudaStream_t=nullptr);
struct WithPolarWarpingConfig { int width=0,height=0,channels=0; float center_x=0.5f,center_y=0.5f; float angle_scale=1.0f,radius_scale=1.0f; std::uint8_t fill=0; };
void with_polar_warping_u8(const std::uint8_t*,std::uint8_t*,const WithPolarWarpingConfig&,cudaStream_t=nullptr);
struct JigsawConfig { int width=0,height=0,channels=0,grid_rows=2,grid_cols=2; const std::uint32_t* permutation=nullptr; std::uint64_t seed=0; };
void jigsaw_u8(const std::uint8_t*,std::uint8_t*,const JigsawConfig&,cudaStream_t=nullptr);
struct FDAConfig { int width=0,height=0,channels=0; const std::uint8_t* reference=nullptr; int reference_width=0,reference_height=0; float low_frequency_radius=0.1f; };
void fda_u8(const std::uint8_t*,std::uint8_t*,const FDAConfig&,cudaStream_t=nullptr);
using FourierDomainAdaptationConfig = FDAConfig;
void fourier_domain_adaptation_u8(const std::uint8_t*,std::uint8_t*,const FourierDomainAdaptationConfig&,cudaStream_t=nullptr);
}
