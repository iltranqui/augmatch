#include "augmatch/color/color.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace augmatch { namespace { void check(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid color dimensions");} std::uint8_t sat(int x){return static_cast<std::uint8_t>(std::max(0,std::min(255,x)));} int round_u8(float x){if(x<=0.f)return 0;if(x>=255.f)return 255;return static_cast<int>(std::floor(x+0.5f));}
std::uint64_t splitmix64(std::uint64_t value){value+=0x9e3779b97f4a7c15ULL;value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;value=(value^(value>>27))*0x94d049bb133111ebULL;return value^(value>>31);}
void validate_random_jitter(const RandomColorJitterConfig& c){
  check(c.width,c.height,c.channels);
  const float values[]={c.brightness_min,c.brightness_max,c.contrast_min,c.contrast_max,c.saturation_min,c.saturation_max,c.hue_min,c.hue_max};
  for(float value:values)if(!std::isfinite(value))throw std::invalid_argument("random color jitter ranges must be finite");
  if(c.brightness_min>c.brightness_max||c.contrast_min>c.contrast_max||c.saturation_min>c.saturation_max||c.hue_min>c.hue_max||c.contrast_min<0.0f||c.saturation_min<0.0f)
    throw std::invalid_argument("invalid random color jitter range");
}
float random_jitter_value(float minimum,float maximum,std::uint64_t seed,int index){
  const float unit=static_cast<float>(splitmix64(seed+static_cast<std::uint64_t>(index))>>40)/16777216.0f;
  return minimum+(maximum-minimum)*unit;
}
}
void rgb_shift_u8(const std::uint8_t* in,std::uint8_t* out,const RGBShiftConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.channels<3)throw std::invalid_argument("RGB shift requires at least three channels");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const auto* p=in+(y*c.width+x)*c.channels;auto* q=out+(y*c.width+x)*c.channels;for(int ch=0;ch<c.channels;++ch)q[ch]=sat(static_cast<int>(p[ch])+(ch==0?c.shift0:ch==1?c.shift1:ch==2?c.shift2:0));}}
void channel_shuffle_u8(const std::uint8_t* in,std::uint8_t* out,const ChannelShuffleConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.order0<0||c.order0>=c.channels||c.order1<0||c.order1>=c.channels||c.order2<0||c.order2>=c.channels)throw std::invalid_argument("invalid channel permutation");for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;if(c.channels>=3){q[0]=p[c.order0];q[1]=p[c.order1];q[2]=p[c.order2];}for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}}
void color_jitter_u8(const std::uint8_t* in,std::uint8_t* out,const ColorJitterConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.channels<3||!std::isfinite(c.brightness)||!std::isfinite(c.contrast)||!std::isfinite(c.saturation)||!std::isfinite(c.hue)||c.contrast<0.f||c.saturation<0.f)throw std::invalid_argument("invalid color jitter configuration");for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;int rgb[3];for(int ch=0;ch<3;++ch)rgb[ch]=round_u8(c.contrast*static_cast<float>(p[ch])+c.brightness*255.f);int red=rgb[0],green=rgb[1],blue=rgb[2],value=std::max(red,std::max(green,blue)),minimum=std::min(red,std::min(green,blue)),difference=value-minimum,hue_bin=0;if(difference>0){int numerator=value==red?green-blue+6*difference:value==green?blue-red+2*difference:red-green+4*difference;hue_bin=(numerator*30+difference/2)/difference;if(hue_bin<0)hue_bin+=180;}int hue_shift=static_cast<int>(std::floor(c.hue+0.5f));hue_bin=(hue_bin+hue_shift)%180;if(hue_bin<0)hue_bin+=180;int saturation_bin=value==0?0:(difference*255+value/2)/value,adjusted_saturation=round_u8(saturation_bin*c.saturation),remainder=hue_bin%30;int p_value=(value*(255-adjusted_saturation)+127)/255,q_value=(value*(255-(adjusted_saturation*remainder)/30)+127)/255,t_value=(value*(255-(adjusted_saturation*(30-remainder))/30)+127)/255;switch(hue_bin/30){case 0:q[0]=value;q[1]=t_value;q[2]=p_value;break;case 1:q[0]=q_value;q[1]=value;q[2]=p_value;break;case 2:q[0]=p_value;q[1]=value;q[2]=t_value;break;case 3:q[0]=p_value;q[1]=q_value;q[2]=value;break;case 4:q[0]=t_value;q[1]=p_value;q[2]=value;break;default:q[0]=value;q[1]=p_value;q[2]=q_value;break;}for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}}
ColorJitterConfig make_random_color_jitter_config(const RandomColorJitterConfig& c){
  validate_random_jitter(c);
  return {c.width,c.height,c.channels,
    random_jitter_value(c.brightness_min,c.brightness_max,c.seed,0),
    random_jitter_value(c.contrast_min,c.contrast_max,c.seed,1),
    random_jitter_value(c.saturation_min,c.saturation_max,c.seed,2),
    random_jitter_value(c.hue_min,c.hue_max,c.seed,3)};
}
void random_color_jitter_u8(const std::uint8_t* in,std::uint8_t* out,const RandomColorJitterConfig& c,cudaStream_t stream){
  color_jitter_u8(in,out,make_random_color_jitter_config(c),stream);
}
namespace {
struct PlanckianRGB { float red,green,blue; };
PlanckianRGB planckian_rgb(float kelvin){
  const float t=kelvin/100.0f;
  PlanckianRGB rgb{};
  if(t<=66.0f){
    rgb.red=255.0f;
    rgb.green=99.4708025861f*std::log(t)-161.1195681661f;
    rgb.blue=t<=19.0f?0.0f:138.5177312231f*std::log(t-10.0f)-305.0447927307f;
  }else{
    rgb.red=329.698727446f*std::pow(t-60.0f,-0.1332047592f);
    rgb.green=288.1221695283f*std::pow(t-60.0f,-0.0755148492f);
    rgb.blue=255.0f;
  }
  rgb.red=std::max(0.0f,std::min(255.0f,rgb.red));
  rgb.green=std::max(0.0f,std::min(255.0f,rgb.green));
  rgb.blue=std::max(0.0f,std::min(255.0f,rgb.blue));
  return rgb;
}
}
void planckian_jitter_u8(const std::uint8_t* in,std::uint8_t* out,const PlanckianJitterConfig& c,cudaStream_t){
  check(c.width,c.height,c.channels);
  if(!in||!out||c.channels<3||!std::isfinite(c.temperature_kelvin)||c.temperature_kelvin<1000.0f||c.temperature_kelvin>40000.0f)throw std::invalid_argument("PlanckianJitter temperature must be finite and in [1000,40000] K");
  const PlanckianRGB reference=planckian_rgb(6500.0f), color=planckian_rgb(c.temperature_kelvin);
  const float gains[3]={color.red/reference.red,color.green/reference.green,color.blue/reference.blue};
  for(int i=0,n=c.width*c.height;i<n;++i){
    const auto* p=in+i*c.channels;auto* q=out+i*c.channels;
    for(int ch=0;ch<3;++ch)q[ch]=round_u8(static_cast<float>(p[ch])*gains[ch]);
    for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];
  }
}
void change_color_temperature_u8(const std::uint8_t* in,std::uint8_t* out,const ChangeColorTemperatureConfig& c,cudaStream_t stream){
  planckian_jitter_u8(in,out,{c.width,c.height,c.channels,c.temperature_kelvin},stream);
}
void chromatic_aberration_u8(const std::uint8_t* in,std::uint8_t* out,const ChromaticAberrationConfig& c,cudaStream_t){
  check(c.width,c.height,c.channels);
  if(!in||!out||c.channels<3||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1||
     !std::isfinite(c.radial0)||!std::isfinite(c.radial1)||!std::isfinite(c.radial2)||!std::isfinite(c.gain0)||!std::isfinite(c.gain1)||!std::isfinite(c.gain2)||c.gain0<0.0f||c.gain1<0.0f||c.gain2<0.0f)
    throw std::invalid_argument("invalid chromatic aberration configuration");
  const float cx=(c.width-1)*0.5f,cy=(c.height-1)*0.5f,fx=std::max(1.0f,c.width*0.5f),fy=std::max(1.0f,c.height*0.5f);
  const float radial[3]={c.radial0,c.radial1,c.radial2},gain[3]={c.gain0,c.gain1,c.gain2};
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x){
    const float xn=(x-cx)/fx,yn=(y-cy)/fy,r2=xn*xn+yn*yn;
    auto sample=[&](float sx,float sy,int ch){
      if(!std::isfinite(sx)||!std::isfinite(sy)||std::fabs(sx)>=static_cast<float>(std::numeric_limits<int>::max())||std::fabs(sy)>=static_cast<float>(std::numeric_limits<int>::max()))return static_cast<float>(c.fill);
      if(c.interpolation==Interpolation::Nearest){
        const int ix=static_cast<int>(std::lround(sx)),iy=static_cast<int>(std::lround(sy));
        return (ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?static_cast<float>(in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]):static_cast<float>(c.fill);
      }
      const int x0=static_cast<int>(std::floor(sx)),y0=static_cast<int>(std::floor(sy));const float ax=sx-x0,ay=sy-y0;
      const auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};
      return (1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));
    };
    auto* q=out+(static_cast<std::size_t>(y)*c.width+x)*c.channels;
    for(int ch=0;ch<3;++ch){const float scale=1.0f+radial[ch]*r2;const float sx=cx+xn*scale*fx,sy=cy+yn*scale*fy;q[ch]=round_u8(sample(sx,sy,ch)*gain[ch]);}
    for(int ch=3;ch<c.channels;++ch)q[ch]=in[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch];
  }
}
void optical_chromatic_aberration_u8(const std::uint8_t* in,std::uint8_t* out,const OpticalChromaticAberrationConfig& c,cudaStream_t s){
  chromatic_aberration_u8(in,out,{c.width,c.height,c.channels,c.radial0,c.radial1,c.radial2,c.gain0,c.gain1,c.gain2,c.interpolation,c.fill},s);
}
void lateral_chromatic_aberration_u8(const std::uint8_t* in,std::uint8_t* out,const LateralChromaticAberrationConfig& c,cudaStream_t s){
  chromatic_aberration_u8(in,out,{c.width,c.height,c.channels,c.radial0,c.radial1,c.radial2,1.0f,1.0f,1.0f,c.interpolation,c.fill},s);
}
void longitudinal_chromatic_aberration_u8(const std::uint8_t* in,std::uint8_t* out,const LongitudinalChromaticAberrationConfig& c,cudaStream_t){
  check(c.width,c.height,c.channels); if(!in||!out||c.channels<3||c.radius0<0||c.radius1<0||c.radius2<0||c.radius0>32||c.radius1>32||c.radius2>32||!std::isfinite(c.sigma)||c.sigma<=0.0f) throw std::invalid_argument("invalid longitudinal chromatic aberration configuration");
  const int radii[3]={c.radius0,c.radius1,c.radius2};
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x){ const std::size_t dst=(static_cast<std::size_t>(y)*c.width+x)*c.channels;
    for(int ch=0;ch<3;++ch){ const int r=radii[ch]; if(r==0){out[dst+ch]=in[dst+ch];continue;} float sum=0.0f,norm=0.0f; const float inv=1.0f/(2.0f*c.sigma*c.sigma);
      for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx){ const float w=std::exp(-(dx*dx+dy*dy)*inv); const int px=std::max(0,std::min(c.width-1,x+dx)),py=std::max(0,std::min(c.height-1,y+dy)); sum+=w*in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]; norm+=w; }
      out[dst+ch]=round_u8(sum/norm);
    }
    for(int ch=3;ch<c.channels;++ch) out[dst+ch]=in[dst+ch];
  }
}
void fancy_pca_u8(const std::uint8_t* in,std::uint8_t* out,const FancyPCAConfig& c,cudaStream_t){
  check(c.width,c.height,c.channels);
  if(!in||!out||c.channels<3)throw std::invalid_argument("FancyPCA requires at least three channels");
  for(int i=0;i<9;++i)if(!std::isfinite(c.basis[i]))throw std::invalid_argument("FancyPCA basis must be finite");
  for(int i=0;i<3;++i)if(!std::isfinite(c.eigenvalues[i])||c.eigenvalues[i]<0.0f||!std::isfinite(c.perturbation[i]))throw std::invalid_argument("FancyPCA eigenvalues and perturbation must be finite; eigenvalues must be nonnegative");
  if(!std::isfinite(c.alpha))throw std::invalid_argument("FancyPCA alpha must be finite");
  for(int col=0;col<3;++col){float norm=0.0f;for(int row=0;row<3;++row)norm+=c.basis[row*3+col]*c.basis[row*3+col];if(std::fabs(norm-1.0f)>1e-3f)throw std::invalid_argument("FancyPCA basis must be orthonormal");for(int other=0;other<col;++other){float dot=0.0f;for(int row=0;row<3;++row)dot+=c.basis[row*3+col]*c.basis[row*3+other];if(std::fabs(dot)>1e-3f)throw std::invalid_argument("FancyPCA basis must be orthonormal");}}
  float delta[3]={0.0f,0.0f,0.0f};
  for(int row=0;row<3;++row)for(int col=0;col<3;++col)delta[row]+=c.basis[row*3+col]*c.eigenvalues[col]*c.perturbation[col]*c.alpha;
  for(int i=0,n=c.width*c.height;i<n;++i){const auto* p=in+static_cast<std::size_t>(i)*c.channels;auto* q=out+static_cast<std::size_t>(i)*c.channels;for(int ch=0;ch<3;++ch)q[ch]=round_u8(static_cast<float>(p[ch])+delta[ch]);for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}
}
namespace {
std::uint64_t plasma_mix(std::uint64_t value){
  value^=value>>30; value*=0xbf58476d1ce4e5b9ULL;
  value^=value>>27; value*=0x94d049bb133111ebULL;
  return value^(value>>31);
}
float plasma_value(int x,int y,std::uint64_t seed){
  const std::uint64_t scales[]={8ULL,4ULL,2ULL,1ULL};
  std::uint64_t weights=0;
  for(int scale=0;scale<4;++scale){
    const std::uint64_t key=seed+static_cast<std::uint64_t>(x>>scale)*0x9e3779b97f4a7c15ULL+static_cast<std::uint64_t>(y>>scale)*0xd1b54a32d192ed03ULL+static_cast<std::uint64_t>(scale)*0x94d049bb133111ebULL;
    weights+=((plasma_mix(key)>>48)&0xffffULL)*scales[scale];
  }
  return static_cast<float>(weights)/(65535.0f*15.0f);
}
void validate_plasma(const PlasmaContrastConfig& c){
  check(c.width,c.height,c.channels);
  if(!std::isfinite(c.contrast)||!std::isfinite(c.plasma_strength)||c.contrast<0.0f||c.plasma_strength<0.0f||c.plasma_strength>1.0f)
    throw std::invalid_argument("invalid PlasmaContrast scaling configuration");
  if(c.field){
    for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i)
      if(!std::isfinite(c.field[i])||c.field[i]<0.0f||c.field[i]>1.0f)throw std::invalid_argument("PlasmaContrast field values must be finite and in [0,1]");
  }
}
void validate_plasma(const PlasmaBrightnessContrastConfig& c){
  check(c.width,c.height,c.channels);
  if(!std::isfinite(c.brightness)||!std::isfinite(c.contrast)||!std::isfinite(c.plasma_strength)||c.contrast<0.0f||c.plasma_strength<0.0f||c.plasma_strength>1.0f)
    throw std::invalid_argument("invalid PlasmaBrightnessContrast scaling configuration");
  if(c.field){
    for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i)
      if(!std::isfinite(c.field[i])||c.field[i]<0.0f||c.field[i]>1.0f)throw std::invalid_argument("PlasmaBrightnessContrast field values must be finite and in [0,1]");
  }
}
void validate_plasma(const PlasmaShadowConfig& c){
  check(c.width,c.height,c.channels);
  if(!std::isfinite(c.threshold)||!std::isfinite(c.strength)||c.threshold<0.0f||c.threshold>1.0f||c.strength<0.0f||c.strength>1.0f)
    throw std::invalid_argument("invalid PlasmaShadow threshold or strength");
  if(c.field){
    for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i)
      if(!std::isfinite(c.field[i])||c.field[i]<0.0f||c.field[i]>1.0f)throw std::invalid_argument("PlasmaShadow field values must be finite and in [0,1]");
  }
}
}
void make_plasma_field(float* field,int width,int height,std::uint64_t seed){
  if(!field||width<=0||height<=0)throw std::invalid_argument("invalid plasma field dimensions or buffer");
  for(int y=0;y<height;++y)for(int x=0;x<width;++x)field[static_cast<std::size_t>(y)*width+x]=plasma_value(x,y,seed);
}
void plasma_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const PlasmaContrastConfig& c,cudaStream_t){
  validate_plasma(c);
  if(!in||!out)throw std::invalid_argument("null PlasmaContrast buffer");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){
    const std::size_t pixel=static_cast<std::size_t>(y)*c.width+x;
    const float field=c.field?c.field[pixel]:plasma_value(x,y,c.seed);
    const float factor=c.contrast*(1.0f+c.plasma_strength*(2.0f*field-1.0f));
    const auto* p=in+pixel*c.channels;auto* q=out+pixel*c.channels;
    for(int ch=0;ch<c.channels;++ch)q[ch]=round_u8(127.5f+(static_cast<float>(p[ch])-127.5f)*factor);
  }
}
void plasma_brightness_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const PlasmaBrightnessContrastConfig& c,cudaStream_t){
  validate_plasma(c);
  if(!in||!out)throw std::invalid_argument("null PlasmaBrightnessContrast buffer");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){
    const std::size_t pixel=static_cast<std::size_t>(y)*c.width+x;
    const float field=c.field?c.field[pixel]:plasma_value(x,y,c.seed);
    const float factor=c.contrast*(1.0f+c.plasma_strength*(2.0f*field-1.0f));
    const auto* p=in+pixel*c.channels;auto* q=out+pixel*c.channels;
    for(int ch=0;ch<c.channels;++ch)q[ch]=round_u8(127.5f+(static_cast<float>(p[ch])-127.5f)*factor+c.brightness*255.0f);
  }
}
void plasma_shadow_u8(const std::uint8_t* in,std::uint8_t* out,const PlasmaShadowConfig& c,cudaStream_t){
  validate_plasma(c);
  if(!in||!out)throw std::invalid_argument("null PlasmaShadow buffer");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){
    const std::size_t pixel=static_cast<std::size_t>(y)*c.width+x;
    const float field=c.field?c.field[pixel]:plasma_value(x,y,c.seed);
    const float shadow=c.threshold>0.0f&&field<c.threshold?(c.threshold-field)/c.threshold:0.0f;
    const float factor=1.0f-c.strength*shadow;
    const auto* p=in+pixel*c.channels;auto* q=out+pixel*c.channels;
    for(int ch=0;ch<c.channels;++ch)q[ch]=round_u8(static_cast<float>(p[ch])*factor);
  }
}
void uniform_color_quantization_u8(const std::uint8_t* in,std::uint8_t* out,const UniformColorQuantizationConfig& c,cudaStream_t){
  check(c.width,c.height,c.channels);
  if(!in||!out||c.levels<2||c.levels>256)throw std::invalid_argument("UniformColorQuantization levels must be in [2,256]");
  const float bin=256.0f/static_cast<float>(c.levels);
  const int quantized_channels=c.channels==4?3:c.channels;
  if(c.levels==256){std::copy(in,in+static_cast<std::size_t>(c.width)*c.height*c.channels,out);return;}
  for(int i=0,n=c.width*c.height;i<n;++i){
    const auto* p=in+static_cast<std::size_t>(i)*c.channels;auto* q=out+static_cast<std::size_t>(i)*c.channels;
    for(int ch=0;ch<quantized_channels;++ch)q[ch]=round_u8(std::floor(static_cast<float>(p[ch])/bin)*bin+0.5f*bin);
    for(int ch=quantized_channels;ch<c.channels;++ch)q[ch]=p[ch];
  }
}
void uniform_color_quantization_to_n_bits_u8(const std::uint8_t* in,std::uint8_t* out,const UniformColorQuantizationToNBitsConfig& c,cudaStream_t stream){
  check(c.width,c.height,c.channels);
  if(!in||!out||c.bits<1||c.bits>8)throw std::invalid_argument("UniformColorQuantizationToNBits bits must be in [1,8]");
  const int quantized_channels=c.channels==4?3:c.channels;
  const std::uint8_t mask=static_cast<std::uint8_t>(0xffu << (8-c.bits));
  const std::size_t pixels=static_cast<std::size_t>(c.width)*c.height;
  for(std::size_t i=0;i<pixels;++i){
    const auto* p=in+i*c.channels;auto* q=out+i*c.channels;
    for(int ch=0;ch<quantized_channels;++ch)q[ch]=static_cast<std::uint8_t>(p[ch]&mask);
    for(int ch=quantized_channels;ch<c.channels;++ch)q[ch]=p[ch];
  }
}
void to_rgb_u8(const std::uint8_t* in,std::uint8_t* out,const ToRGBConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||(c.input_channels!=1&&c.input_channels!=3))throw std::invalid_argument("ToRGB requires one or three input channels");for(int i=0,n=c.width*c.height;i<n;++i){if(c.input_channels==1)out[3*i]=out[3*i+1]=out[3*i+2]=in[i];else{out[3*i]=in[3*i];out[3*i+1]=in[3*i+1];out[3*i+2]=in[3*i+2];}}}
namespace {
float photometric_clip(float x,float lo,float hi){if(!std::isfinite(x))return lo;if(lo>hi)std::swap(lo,hi);return std::max(lo,std::min(hi,x));}
float photometric_unit(std::uint64_t key){return static_cast<float>((splitmix64(key)>>40)&0xffffffULL)/16777216.0f;}
float photometric_normal(std::uint64_t key){const float u=std::max(1.0e-7f,photometric_unit(key));const float v=photometric_unit(key+0x9e3779b97f4a7c15ULL);return std::sqrt(-2.0f*std::log(u))*std::cos(6.2831853071795864769f*v);}
void validate_photo(int w,int h,int c,const std::uint8_t* in,const std::uint8_t* out){check(w,h,c);if(!in||!out||c<3)throw std::invalid_argument("color photometric operations require RGB HWC buffers");}
void validate_clip(float lo,float hi){if(!std::isfinite(lo)||!std::isfinite(hi)||lo>hi)throw std::invalid_argument("invalid normalized clipping range");}
void validate_matrix(const float* m,const float* o,float lo,float hi){validate_clip(lo,hi);for(int i=0;i<9;++i)if(!std::isfinite(m[i]))throw std::invalid_argument("color matrix must be finite");for(int i=0;i<3;++i)if(!std::isfinite(o[i]))throw std::invalid_argument("color offset must be finite");}
void apply_rgb_matrix(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,const float* m,const float* offset,float lo,float hi,float gain0=1,float gain1=1,float gain2=1,float noise=0,std::uint64_t seed=0){for(std::size_t i=0,n=static_cast<std::size_t>(w)*h;i<n;++i){const auto* p=in+i*c;auto* q=out+i*c;float x[3]={p[0]/255.0f,p[1]/255.0f,p[2]/255.0f};for(int r=0;r<3;++r){float y=offset[r];for(int k=0;k<3;++k)y+=m[r*3+k]*x[k];const float g=r==0?gain0:r==1?gain1:gain2;y*=g;if(noise!=0)y+=noise*photometric_normal(seed+static_cast<std::uint64_t>(i)*3ULL+static_cast<std::uint64_t>(r));q[r]=round_u8(photometric_clip(y,lo,hi)*255.0f);}for(int ch=3;ch<c;++ch)q[ch]=p[ch];}}
void validate_noise(float s){if(!std::isfinite(s)||s<0)throw std::invalid_argument("noise standard deviation must be finite and nonnegative");}
void rgb_to_ycc(float r,float g,float b,float& y,float& cb,float& cr){y=.299f*r+.587f*g+.114f*b;cb=b-y;cr=r-y;}
void ycc_to_rgb(float y,float cb,float cr,float& r,float& g,float& b){r=y+cr;b=y+cb;g=(y-.299f*r-.114f*b)/.587f;}
}
void color_matrix_perturbation_u8(const std::uint8_t* in,std::uint8_t* out,const ColorMatrixPerturbationConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_matrix(c.matrix,c.offset,c.clip_min,c.clip_max);apply_rgb_matrix(in,out,c.width,c.height,c.channels,c.matrix,c.offset,c.clip_min,c.clip_max);}
void camera_color_profile_variation_u8(const std::uint8_t* in,std::uint8_t* out,const CameraColorProfileVariationConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_matrix(c.matrix,c.offset,c.clip_min,c.clip_max);validate_noise(c.noise_stddev);for(float g:c.gains)if(!std::isfinite(g)||g<0)throw std::invalid_argument("profile gains must be finite and nonnegative");apply_rgb_matrix(in,out,c.width,c.height,c.channels,c.matrix,c.offset,c.clip_min,c.clip_max,c.gains[0],c.gains[1],c.gains[2],c.noise_stddev,c.seed);}
void rgb_channel_cross_talk_u8(const std::uint8_t* in,std::uint8_t* out,const RGBChannelCrossTalkConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_matrix(c.matrix,c.offset,c.clip_min,c.clip_max);apply_rgb_matrix(in,out,c.width,c.height,c.channels,c.matrix,c.offset,c.clip_min,c.clip_max);}
void sensor_spectral_response_variation_u8(const std::uint8_t* in,std::uint8_t* out,const SensorSpectralResponseVariationConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_matrix(c.response,c.offset,c.clip_min,c.clip_max);validate_noise(c.response_stddev);float m[9];for(int i=0;i<9;++i)m[i]=c.response[i]+c.response_stddev*photometric_normal(c.seed+static_cast<std::uint64_t>(i));apply_rgb_matrix(in,out,c.width,c.height,c.channels,m,c.offset,c.clip_min,c.clip_max);}
void color_clipping_u8(const std::uint8_t* in,std::uint8_t* out,const ColorClippingConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);for(int k=0;k<3;++k)validate_clip(c.clip_min[k],c.clip_max[k]);for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;for(int k=0;k<3;++k)q[k]=round_u8(photometric_clip(p[k]/255.0f,c.clip_min[k],c.clip_max[k])*255.0f);for(int k=3;k<c.channels;++k)q[k]=p[k];}}
void white_balance_clipping_u8(const std::uint8_t* in,std::uint8_t* out,const WhiteBalanceClippingConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_clip(c.clip_min,c.clip_max);for(float g:c.gains)if(!std::isfinite(g)||g<0)throw std::invalid_argument("white-balance gains must be finite and nonnegative");for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;for(int k=0;k<3;++k)q[k]=round_u8(photometric_clip(p[k]/255.0f*c.gains[k],c.clip_min,c.clip_max)*255.0f);for(int k=3;k<c.channels;++k)q[k]=p[k];}}
void chroma_noise_u8(const std::uint8_t* in,std::uint8_t* out,const ChromaNoiseConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_noise(c.stddev);for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;float y,cb,cr;rgb_to_ycc(p[0]/255.f,p[1]/255.f,p[2]/255.f,y,cb,cr);cb+=c.stddev*photometric_normal(c.seed+i*3+0);cr+=c.stddev*photometric_normal(c.seed+i*3+1);float r,g,b;ycc_to_rgb(y,cb,cr,r,g,b);q[0]=round_u8(photometric_clip(r,0,1)*255);q[1]=round_u8(photometric_clip(g,0,1)*255);q[2]=round_u8(photometric_clip(b,0,1)*255);for(int k=3;k<c.channels;++k)q[k]=p[k];}}
void luma_noise_u8(const std::uint8_t* in,std::uint8_t* out,const LumaNoiseConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_noise(c.stddev);for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;float y,cb,cr;rgb_to_ycc(p[0]/255.f,p[1]/255.f,p[2]/255.f,y,cb,cr);y+=c.stddev*photometric_normal(c.seed+i*3);float r,g,b;ycc_to_rgb(y,cb,cr,r,g,b);q[0]=round_u8(photometric_clip(r,0,1)*255);q[1]=round_u8(photometric_clip(g,0,1)*255);q[2]=round_u8(photometric_clip(b,0,1)*255);for(int k=3;k<c.channels;++k)q[k]=p[k];}}
void correlated_luma_chroma_noise_u8(const std::uint8_t* in,std::uint8_t* out,const CorrelatedLumaChromaNoiseConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_noise(c.luma_stddev);validate_noise(c.chroma_stddev);if(!std::isfinite(c.luma_chroma_correlation)||c.luma_chroma_correlation<-1||c.luma_chroma_correlation>1)throw std::invalid_argument("luma/chroma correlation must be in [-1,1]");const float s=std::sqrt(std::max(0.0f,1.0f-c.luma_chroma_correlation*c.luma_chroma_correlation));for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;float y,cb,cr;rgb_to_ycc(p[0]/255.f,p[1]/255.f,p[2]/255.f,y,cb,cr);float z0=photometric_normal(c.seed+i*3),z1=photometric_normal(c.seed+i*3+1),z2=photometric_normal(c.seed+i*3+2);y+=c.luma_stddev*z0;cb+=c.chroma_stddev*(c.luma_chroma_correlation*z0+s*z1);cr+=c.chroma_stddev*(c.luma_chroma_correlation*z0+s*z2);float r,g,b;ycc_to_rgb(y,cb,cr,r,g,b);q[0]=round_u8(photometric_clip(r,0,1)*255);q[1]=round_u8(photometric_clip(g,0,1)*255);q[2]=round_u8(photometric_clip(b,0,1)*255);for(int k=3;k<c.channels;++k)q[k]=p[k];}}
void chroma_subsampling_artifacts_u8(const std::uint8_t* in,std::uint8_t* out,const ChromaSubsamplingArtifactsConfig& c,cudaStream_t){
  validate_photo(c.width,c.height,c.channels,in,out);
  const int mode=static_cast<int>(c.subsampling); if(mode<0||mode>2)throw std::invalid_argument("chroma subsampling must be Y444, Y422, or Y420");
  if(mode==0){std::copy(in,in+static_cast<std::size_t>(c.width)*c.height*c.channels,out);return;}
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){
    const auto* p=in+(static_cast<std::size_t>(y)*c.width+x)*c.channels; auto* q=out+(static_cast<std::size_t>(y)*c.width+x)*c.channels;
    float yv,cb,cr; rgb_to_ycc(p[0]/255.0f,p[1]/255.0f,p[2]/255.0f,yv,cb,cr);
    const int x0=(x/2)*2, y0=mode==2?(y/2)*2:y;
    float cb_sum=0.0f,cr_sum=0.0f; int count=0;
    for(int yy=y0;yy<=y0+(mode==2?1:0)&&yy<c.height;++yy)for(int xx=x0;xx<=x0+1&&xx<c.width;++xx){float ny,ncb,ncr;const auto* s=in+(static_cast<std::size_t>(yy)*c.width+xx)*c.channels;rgb_to_ycc(s[0]/255.0f,s[1]/255.0f,s[2]/255.0f,ny,ncb,ncr);cb_sum+=ncb;cr_sum+=ncr;++count;}
    cb=cb_sum/count; cr=cr_sum/count; float r,g,b; ycc_to_rgb(yv,cb,cr,r,g,b);
    q[0]=round_u8(photometric_clip(r,0,1)*255.0f);q[1]=round_u8(photometric_clip(g,0,1)*255.0f);q[2]=round_u8(photometric_clip(b,0,1)*255.0f);for(int k=3;k<c.channels;++k)q[k]=p[k];
  }
}
void local_tone_mapping_noise_u8(const std::uint8_t* in,std::uint8_t* out,const LocalToneMappingNoiseConfig& c,cudaStream_t){
  validate_photo(c.width,c.height,c.channels,in,out); if(c.radius<0||c.radius>32||!std::isfinite(c.tone_strength)||c.tone_strength<0.0f||c.tone_strength>1.0f)throw std::invalid_argument("invalid local tone-mapping strength or radius"); validate_noise(c.noise_stddev);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const std::size_t pixel=static_cast<std::size_t>(y)*c.width+x;const auto* p=in+pixel*c.channels;auto* q=out+pixel*c.channels;float yy,cb,cr;rgb_to_ycc(p[0]/255.0f,p[1]/255.0f,p[2]/255.0f,yy,cb,cr);float mean=0.0f;int count=0;for(int ny=std::max(0,y-c.radius);ny<=std::min(c.height-1,y+c.radius);++ny)for(int nx=std::max(0,x-c.radius);nx<=std::min(c.width-1,x+c.radius);++nx){const auto* s=in+(static_cast<std::size_t>(ny)*c.width+nx)*c.channels;float sy,scb,scr;rgb_to_ycc(s[0]/255.0f,s[1]/255.0f,s[2]/255.0f,sy,scb,scr);mean+=sy;++count;}yy+=c.tone_strength*(mean/count-yy)+c.noise_stddev*photometric_normal(c.seed+pixel);float r,g,b;ycc_to_rgb(yy,cb,cr,r,g,b);q[0]=round_u8(photometric_clip(r,0,1)*255.0f);q[1]=round_u8(photometric_clip(g,0,1)*255.0f);q[2]=round_u8(photometric_clip(b,0,1)*255.0f);for(int k=3;k<c.channels;++k)q[k]=p[k];}
}
void per_channel_gain_noise_u8(const std::uint8_t* in,std::uint8_t* out,const PerChannelGainNoiseConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);validate_noise(c.stddev);for(float g:c.gains)if(!std::isfinite(g)||g<0)throw std::invalid_argument("gain values must be finite and nonnegative");for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;for(int k=0;k<3;++k)q[k]=round_u8(p[k]*(c.gains[k]+c.stddev*photometric_normal(c.seed+i*3+k)));for(int k=3;k<c.channels;++k)q[k]=p[k];}}
void color_temperature_error_u8(const std::uint8_t* in,std::uint8_t* out,const ColorTemperatureErrorConfig& c,cudaStream_t){validate_photo(c.width,c.height,c.channels,in,out);if(!std::isfinite(c.temperature_kelvin)||c.temperature_kelvin<1000||c.temperature_kelvin>40000||!std::isfinite(c.strength)||c.strength<0||c.strength>1)throw std::invalid_argument("invalid color-temperature error configuration");const PlanckianRGB ref=planckian_rgb(6500),col=planckian_rgb(c.temperature_kelvin);const float gain[3]={1+c.strength*(col.red/ref.red-1),1+c.strength*(col.green/ref.green-1),1+c.strength*(col.blue/ref.blue-1)};for(std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i){const auto* p=in+i*c.channels;auto* q=out+i*c.channels;for(int k=0;k<3;++k)q[k]=round_u8(p[k]*gain[k]);for(int k=3;k<c.channels;++k)q[k]=p[k];}}
}

