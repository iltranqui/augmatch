#include "augmatch/arithmetic.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace augmatch { namespace { void check(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid arithmetic dimensions");} unsigned char sat(float x){return static_cast<unsigned char>(std::lround(std::max(0.0f,std::min(255.0f,x))));} std::uint64_t mix(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} float uni(std::uint64_t x){return static_cast<float>((mix(x)>>11)*(1.0/9007199254740992.0));} float normal(std::uint64_t x){float a=std::max(uni(x),std::numeric_limits<float>::min());return std::sqrt(-2.0f*std::log(a))*std::cos(6.28318530718f*uni(x^0xd1b54a32d192ed03ULL));} }
void add_u8(const std::uint8_t* in,std::uint8_t* out,const ScalarArithmeticConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||!std::isfinite(c.value))throw std::invalid_argument("invalid add configuration");for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=sat(in[i]+c.value);}
void multiply_u8(const std::uint8_t* in,std::uint8_t* out,const ScalarArithmeticConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||!std::isfinite(c.value)||c.value<0)throw std::invalid_argument("invalid multiply configuration");for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=static_cast<std::uint8_t>(std::max(0.0f,std::min(255.0f,in[i]*c.value)));}
namespace {
std::size_t element_count(int width,int height,int channels){
  check(width,height,channels);
  return static_cast<std::size_t>(width)*static_cast<std::size_t>(height)*static_cast<std::size_t>(channels);
}
void check_range(float min_value,float max_value,bool multiply){
  if(!std::isfinite(min_value)||!std::isfinite(max_value)||min_value>max_value||(multiply&&(min_value<0.0f||max_value<0.0f)))
    throw std::invalid_argument("invalid elementwise arithmetic range");
}
float generated_value(std::uint64_t seed,std::size_t index,float min_value,float max_value){
  const float unit=uni(seed+static_cast<std::uint64_t>(index)*0x9e3779b97f4a7c15ULL);
  return min_value+(max_value-min_value)*unit;
}
template<class Config>
void check_elementwise(const std::uint8_t* in,std::uint8_t* out,const Config& c,bool multiply,std::size_t n){
  if(!in||!out)throw std::invalid_argument("null elementwise arithmetic buffer");
  if(c.values==nullptr)check_range(c.min_value,c.max_value,multiply);
  if(c.values!=nullptr)for(std::size_t i=0;i<n;++i)if(!std::isfinite(c.values[i])||(multiply&&c.values[i]<0.0f))throw std::invalid_argument("invalid elementwise arithmetic value");
}
}
void make_add_elementwise_values(float* values,int width,int height,int channels,float min_value,float max_value,std::uint64_t seed){
  const std::size_t n=element_count(width,height,channels);
  if(!values)throw std::invalid_argument("null add elementwise values");
  check_range(min_value,max_value,false);
  for(std::size_t i=0;i<n;++i)values[i]=generated_value(seed,i,min_value,max_value);
}
void add_elementwise_u8(const std::uint8_t* in,std::uint8_t* out,const AddElementwiseConfig& c,cudaStream_t){
  const std::size_t n=element_count(c.width,c.height,c.channels);check_elementwise(in,out,c,false,n);
  for(std::size_t i=0;i<n;++i){const float value=c.values?c.values[i]:generated_value(c.seed,i,c.min_value,c.max_value);out[i]=sat(static_cast<float>(in[i])+value);}
}
void make_multiply_elementwise_values(float* values,int width,int height,int channels,float min_value,float max_value,std::uint64_t seed){
  const std::size_t n=element_count(width,height,channels);
  if(!values)throw std::invalid_argument("null multiply elementwise values");
  check_range(min_value,max_value,true);
  for(std::size_t i=0;i<n;++i)values[i]=generated_value(seed,i,min_value,max_value);
}
void multiply_elementwise_u8(const std::uint8_t* in,std::uint8_t* out,const MultiplyElementwiseConfig& c,cudaStream_t){
  const std::size_t n=element_count(c.width,c.height,c.channels);check_elementwise(in,out,c,true,n);
  for(std::size_t i=0;i<n;++i){const float value=c.values?c.values[i]:generated_value(c.seed,i,c.min_value,c.max_value);out[i]=sat(static_cast<float>(in[i])*value);}
}
void posterize_u8(const std::uint8_t* in,std::uint8_t* out,const PosterizeConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.bits<1||c.bits>8)throw std::invalid_argument("posterize bits must be in [1,8]");const int mask=255<<(8-c.bits);for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=static_cast<std::uint8_t>(in[i]&mask);}
void solarize_u8(const std::uint8_t* in,std::uint8_t* out,const SolarizeConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.threshold<0||c.threshold>255)throw std::invalid_argument("solarize threshold must be in [0,255]");for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=in[i]>c.threshold?static_cast<std::uint8_t>(255-in[i]):in[i];}
void invert_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,cudaStream_t){check(w,h,c);if(!in||!out)throw std::invalid_argument("null invert buffer");for(int i=0,n=w*h*c;i<n;++i)out[i]=static_cast<std::uint8_t>(255-in[i]);}
void gaussian_noise_u8(const std::uint8_t* in,std::uint8_t* out,const GaussianNoiseConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.stddev<0||!std::isfinite(c.stddev))throw std::invalid_argument("invalid Gaussian noise configuration");for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=sat(in[i]+c.stddev*normal(c.seed+static_cast<std::uint64_t>(i)*0x9e3779b97f4a7c15ULL));}
void salt_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const SaltPepperConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.probability<0||c.probability>1||c.salt_probability<0||c.salt_probability>1)throw std::invalid_argument("invalid salt and pepper configuration");for(int i=0,n=c.width*c.height*c.channels;i<n;++i){float r=uni(c.seed+static_cast<std::uint64_t>(i)*0x9e3779b97f4a7c15ULL);out[i]=r<c.probability?(uni(c.seed^0xabcdefULL+static_cast<std::uint64_t>(i))<c.salt_probability?255:0):in[i];}}
void make_replace_elementwise_mask(std::uint8_t* mask,int width,int height,int channels,float probability,std::uint64_t seed){
  const std::size_t n=element_count(width,height,channels);
  if(!mask||!std::isfinite(probability)||probability<0.0f||probability>1.0f)throw std::invalid_argument("invalid replacement mask configuration");
  for(std::size_t i=0;i<n;++i)mask[i]=uni(seed+static_cast<std::uint64_t>(i)*0x9e3779b97f4a7c15ULL)<probability?1:0;
}
void replace_elementwise_u8(const std::uint8_t* in,std::uint8_t* out,const ReplaceElementwiseConfig& c,cudaStream_t){
  const std::size_t n=element_count(c.width,c.height,c.channels);
  if(!in||!out||(c.mask==nullptr&&(!std::isfinite(c.probability)||c.probability<0.0f||c.probability>1.0f))||(c.mask==nullptr&&c.values!=nullptr))throw std::invalid_argument("invalid ReplaceElementwise configuration");
  for(std::size_t i=0;i<n;++i){const bool selected=c.mask?c.mask[i]!=0:uni(c.seed+static_cast<std::uint64_t>(i)*0x9e3779b97f4a7c15ULL)<c.probability;out[i]=selected?(c.values?c.values[i]:c.replacement_value):in[i];}
}
void make_impulse_noise_mask(std::uint8_t* mask,int width,int height,int channels,float probability,std::uint64_t seed){
  make_replace_elementwise_mask(mask,width,height,channels,probability,seed);
}
void impulse_noise_u8(const std::uint8_t* in,std::uint8_t* out,const ImpulseNoiseConfig& c,cudaStream_t){
  const std::size_t n=element_count(c.width,c.height,c.channels);
  if(!in||!out||(c.mask==nullptr&&(!std::isfinite(c.probability)||c.probability<0.0f||c.probability>1.0f))||(c.mask==nullptr&&c.values!=nullptr)||(!c.values&&(!std::isfinite(c.salt_probability)||c.salt_probability<0.0f||c.salt_probability>1.0f)))throw std::invalid_argument("invalid ImpulseNoise configuration");
  for(std::size_t i=0;i<n;++i){const bool selected=c.mask?c.mask[i]!=0:uni(c.seed+static_cast<std::uint64_t>(i)*0x9e3779b97f4a7c15ULL)<c.probability;const std::uint8_t impulse=c.values?c.values[i]:(uni(c.seed^0xabcdefULL+static_cast<std::uint64_t>(i))<c.salt_probability?255:0);out[i]=selected?impulse:in[i];}
}
namespace {
enum class SaltMode { Salt, Pepper, SaltAndPepper };
void validate_salt(const std::uint8_t* in,const std::uint8_t* out,const SaltPepperConfig& c){
  check(c.width,c.height,c.channels);
  if(!in||!out||c.rectangle_count<0||(c.rectangle_count>0&&!c.rectangles)||c.block_width<=0||c.block_height<=0||
     !std::isfinite(c.probability)||c.probability<0.0f||c.probability>1.0f||
     !std::isfinite(c.salt_probability)||c.salt_probability<0.0f||c.salt_probability>1.0f)
    throw std::invalid_argument("invalid salt and pepper configuration");
  for(int i=0;i<c.rectangle_count;++i){const auto& r=c.rectangles[i];if(r.x0<0||r.y0<0||r.x1<r.x0||r.y1<r.y0||r.x1>c.width||r.y1>c.height)throw std::invalid_argument("salt and pepper rectangle is outside the image");}
}
bool in_salt_rectangle(int x,int y,const SaltPepperConfig& c){for(int i=0;i<c.rectangle_count;++i){const auto& r=c.rectangles[i];if(x>=r.x0&&x<r.x1&&y>=r.y0&&y<r.y1)return true;}return false;}
void salt_impl(const std::uint8_t* in,std::uint8_t* out,const SaltPepperConfig& c,SaltMode mode,bool coarse){
  validate_salt(in,out,c);
  const bool explicit_selection=c.mask!=nullptr||c.rectangle_count!=0;
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x){
    const int pixel=y*c.width+x;
    bool selected=false; bool salt=true;
    if(c.mask){selected=c.mask[pixel]!=0;if(mode==SaltMode::SaltAndPepper&&c.mask[pixel]==2)salt=false;}
    if(c.rectangle_count!=0&&in_salt_rectangle(x,y,c)){selected=true;salt=c.salt_probability>=0.5f;}
    if(!explicit_selection){const int bw=coarse?c.block_width:1,bh=coarse?c.block_height:1;const int blocks_x=(c.width+bw-1)/bw;const std::uint64_t block=static_cast<std::uint64_t>(y/bh*blocks_x+x/bw);const std::uint64_t key=c.seed+block*0x9e3779b97f4a7c15ULL;selected=uni(key)<c.probability;salt=uni(key^0xd1b54a32d192ed03ULL)<c.salt_probability;}
    if(!selected) salt=true;
    const std::uint8_t value=mode==SaltMode::Salt?255:mode==SaltMode::Pepper?0:(salt?255:0);
    for(int ch=0;ch<c.channels;++ch){const int i=pixel*c.channels+ch;out[i]=selected?value:in[i];}
  }
}
}
void salt_u8(const std::uint8_t* in,std::uint8_t* out,const SaltConfig& c,cudaStream_t){salt_impl(in,out,c,SaltMode::Salt,false);}
void pepper_u8(const std::uint8_t* in,std::uint8_t* out,const PepperConfig& c,cudaStream_t){salt_impl(in,out,c,SaltMode::Pepper,false);}
void salt_and_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const SaltAndPepperConfig& c,cudaStream_t){salt_impl(in,out,c,SaltMode::SaltAndPepper,false);}
void coarse_salt_u8(const std::uint8_t* in,std::uint8_t* out,const CoarseSaltConfig& c,cudaStream_t){salt_impl(in,out,c,SaltMode::Salt,true);}
void coarse_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const CoarsePepperConfig& c,cudaStream_t){salt_impl(in,out,c,SaltMode::Pepper,true);}
void coarse_salt_and_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const CoarseSaltAndPepperConfig& c,cudaStream_t){salt_impl(in,out,c,SaltMode::SaltAndPepper,true);}
void coarse_salt_pepper_u8(const std::uint8_t* in,std::uint8_t* out,const CoarseSaltPepperConfig& c,cudaStream_t){coarse_salt_and_pepper_u8(in,out,c,nullptr);}
}

