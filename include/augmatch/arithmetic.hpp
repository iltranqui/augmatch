#pragma once
#include <cstdint>
#ifndef AUGMATCH_HAS_CUDA
#define AUGMATCH_HAS_CUDA 0
#endif
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime_api.h>
#else
using cudaStream_t = void*;
#endif
namespace augmatch {
struct ScalarArithmeticConfig { int width=0,height=0,channels=0; float value=0.0f; };
void add_u8(const std::uint8_t*,std::uint8_t*,const ScalarArithmeticConfig&,cudaStream_t stream=nullptr);
void multiply_u8(const std::uint8_t*,std::uint8_t*,const ScalarArithmeticConfig&,cudaStream_t stream=nullptr);
// AddElementwise applies one borrowed offset to every HWC byte.  The values
// array, when non-null, contains width*height*channels host floats for the CPU
// API and device floats for the CUDA API.  When it is null, values are generated
// deterministically from seed in [min_value,max_value); no global RNG is used.
struct AddElementwiseConfig {
  int width=0,height=0,channels=0;
  const float* values=nullptr;
  float min_value=0.0f,max_value=0.0f;
  std::uint64_t seed=0;
};
void make_add_elementwise_values(float* values,int width,int height,int channels,
                                 float min_value,float max_value,std::uint64_t seed);
void add_elementwise_u8(const std::uint8_t*,std::uint8_t*,const AddElementwiseConfig&,cudaStream_t stream=nullptr);
// MultiplyElementwise uses the same borrowed/generated layout and range
// contract, with factors applied independently to each byte.
struct MultiplyElementwiseConfig {
  int width=0,height=0,channels=0;
  const float* values=nullptr;
  float min_value=1.0f,max_value=1.0f;
  std::uint64_t seed=0;
};
void make_multiply_elementwise_values(float* values,int width,int height,int channels,
                                      float min_value,float max_value,std::uint64_t seed);
void multiply_elementwise_u8(const std::uint8_t*,std::uint8_t*,const MultiplyElementwiseConfig&,cudaStream_t stream=nullptr);
using ElementwiseAddConfig = AddElementwiseConfig;
using ElementwiseMultiplyConfig = MultiplyElementwiseConfig;
struct PosterizeConfig { int width=0,height=0,channels=0,bits=8; };
void posterize_u8(const std::uint8_t*,std::uint8_t*,const PosterizeConfig&,cudaStream_t stream=nullptr);
struct SolarizeConfig { int width=0,height=0,channels=0; int threshold=128; };
void solarize_u8(const std::uint8_t*,std::uint8_t*,const SolarizeConfig&,cudaStream_t stream=nullptr);
void invert_u8(const std::uint8_t*,std::uint8_t*,int width,int height,int channels,cudaStream_t stream=nullptr);
struct GaussianNoiseConfig { int width=0,height=0,channels=0; float stddev=0.0f; std::uint64_t seed=0; };
void gaussian_noise_u8(const std::uint8_t*,std::uint8_t*,const GaussianNoiseConfig&,cudaStream_t stream=nullptr);
struct SaltPepperConfig {
  int width=0,height=0,channels=0;
  // Keep the original seeded SaltPepper aggregate prefix source-compatible.
  float probability=0.0f,salt_probability=0.5f;
  std::uint64_t seed=0;
  // A mask has width*height bytes in row-major pixel order. For Salt/Pepper,
  // nonzero selects a pixel. For SaltAndPepper, 1 selects salt and 2 selects
  // pepper. The mask is borrowed host memory on CPU and device memory on CUDA.
  const std::uint8_t* mask=nullptr;
  struct Rectangle { int x0=0,y0=0,x1=0,y1=0; };
  const Rectangle* rectangles=nullptr;
  int rectangle_count=0;
  // Coarse operations draw one decision per block. Fine operations force a
  // 1x1 block. Explicit masks/rectangles take precedence over probability.
  int block_width=1,block_height=1;
};
using SaltConfig = SaltPepperConfig;
using PepperConfig = SaltPepperConfig;
using SaltAndPepperConfig = SaltPepperConfig;
using CoarseSaltConfig = SaltPepperConfig;
using CoarsePepperConfig = SaltPepperConfig;
using CoarseSaltAndPepperConfig = SaltPepperConfig;
using CoarseSaltPepperConfig = CoarseSaltAndPepperConfig;
using SaltPepperRectangle = SaltPepperConfig::Rectangle;
void salt_pepper_u8(const std::uint8_t*,std::uint8_t*,const SaltPepperConfig&,cudaStream_t stream=nullptr);
void salt_u8(const std::uint8_t*,std::uint8_t*,const SaltConfig&,cudaStream_t stream=nullptr);
void pepper_u8(const std::uint8_t*,std::uint8_t*,const PepperConfig&,cudaStream_t stream=nullptr);
void salt_and_pepper_u8(const std::uint8_t*,std::uint8_t*,const SaltAndPepperConfig&,cudaStream_t stream=nullptr);
void coarse_salt_u8(const std::uint8_t*,std::uint8_t*,const CoarseSaltConfig&,cudaStream_t stream=nullptr);
void coarse_pepper_u8(const std::uint8_t*,std::uint8_t*,const CoarsePepperConfig&,cudaStream_t stream=nullptr);
void coarse_salt_and_pepper_u8(const std::uint8_t*,std::uint8_t*,const CoarseSaltAndPepperConfig&,cudaStream_t stream=nullptr);
void coarse_salt_pepper_u8(const std::uint8_t*,std::uint8_t*,const CoarseSaltPepperConfig&,cudaStream_t stream=nullptr);
// ReplaceElementwise uses borrowed one-byte-per-element arrays when mask is non-null.
// A nonzero mask selects replacement; values is optional and falls back to
// replacement_value.  If mask is null, probability and seed define a deterministic
// Bernoulli mask.  Arrays are host-owned for the CPU API and device-owned for CUDA;
// they must remain valid for the duration of the call and are never freed here.
struct ReplaceElementwiseConfig {
  int width=0,height=0,channels=0;
  const std::uint8_t* mask=nullptr;
  const std::uint8_t* values=nullptr;
  std::uint8_t replacement_value=0;
  float probability=0.0f;
  std::uint64_t seed=0;
};
void make_replace_elementwise_mask(std::uint8_t* mask,int width,int height,int channels,
                                   float probability,std::uint64_t seed);
void replace_elementwise_u8(const std::uint8_t*,std::uint8_t*,const ReplaceElementwiseConfig&,cudaStream_t stream=nullptr);
// ImpulseNoise shares the explicit borrowed mask/value contract.  In seeded mode,
// probability is the impulse rate and salt_probability selects 255 versus 0.
// Explicit values take precedence over salt_probability and make the result exact.
struct ImpulseNoiseConfig {
  int width=0,height=0,channels=0;
  const std::uint8_t* mask=nullptr;
  const std::uint8_t* values=nullptr;
  float probability=0.0f;
  float salt_probability=0.5f;
  std::uint64_t seed=0;
};
void make_impulse_noise_mask(std::uint8_t* mask,int width,int height,int channels,
                             float probability,std::uint64_t seed);
void impulse_noise_u8(const std::uint8_t*,std::uint8_t*,const ImpulseNoiseConfig&,cudaStream_t stream=nullptr);
}
