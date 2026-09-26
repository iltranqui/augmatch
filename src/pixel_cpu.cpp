#include "augmatch/pixel.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace augmatch { namespace { void check(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid pixel dimensions");} unsigned char sat(float x){return static_cast<unsigned char>(std::lround(std::max(0.0f,std::min(255.0f,x))));} } 
void brightness_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const BrightnessContrastConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||!std::isfinite(c.brightness)||!std::isfinite(c.contrast)||c.contrast<0)throw std::invalid_argument("invalid brightness/contrast configuration");const float beta=c.brightness*255.0f;for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=sat(c.contrast*in[i]+beta);}
void gamma_u8(const std::uint8_t* in,std::uint8_t* out,const GammaConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||!std::isfinite(c.gamma)||c.gamma<=0)throw std::invalid_argument("invalid gamma configuration");for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=sat(std::pow(in[i]/255.0f,c.gamma)*255.0f);}
void grayscale_u8(const std::uint8_t* in,std::uint8_t* out,const GrayscaleConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null grayscale buffer");for(int i=0;i<c.width*c.height;++i){const auto* p=in+i*c.channels;const auto y=sat(.299f*p[0]+.587f*p[std::min(1,c.channels-1)]+.114f*p[std::min(2,c.channels-1)]);for(int ch=0;ch<c.channels;++ch)out[i*c.channels+ch]=y;}}
}
