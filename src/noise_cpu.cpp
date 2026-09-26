#include "augmatch/sensor/noise.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
namespace augmatch { namespace {
void valid(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid noise dimensions");}
std::uint64_t mix(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
float uni(std::uint64_t x){return static_cast<float>((mix(x)>>11)*(1.0/9007199254740992.0));}
float norm(std::uint64_t x){const float a=std::max(uni(x),std::numeric_limits<float>::min());return std::sqrt(-2.0f*std::log(a))*std::cos(6.28318530718f*uni(x^0xd1b54a32d192ed03ULL));}
std::uint64_t key(std::uint64_t s,int x,int y,int c,std::uint64_t k){return s^static_cast<std::uint64_t>(x)*0x632be59bd9b4e019ULL^static_cast<std::uint64_t>(y)*0x8cb92baa2f2f6f7dULL^static_cast<std::uint64_t>(c)*0x9e3779b97f4a7c15ULL^k;}
float pois(float lambda,std::uint64_t seed){if(lambda<=0)return 0;if(lambda<30){std::mt19937_64 g(seed);return static_cast<float>(std::poisson_distribution<int>(lambda)(g));}return std::max(0.0f,lambda+std::sqrt(lambda)*norm(seed^0x517cc1b727220a95ULL));}
// This counter-based sampler is shared by ShotNoise's CPU and CUDA contracts.
// Inversion is exact for the common low-photon regime; the normal branch keeps
// the one-thread-per-pixel CUDA path bounded for large photon counts.
float shot_pois(float lambda,std::uint64_t seed){
  if(lambda<=0.0f)return 0.0f;
  if(lambda<64.0f){
    const float u=uni(seed^0x53484f545f554e49ULL);
    float probability=std::exp(-lambda),cdf=probability;
    int count=0;
    while(u>cdf&&count<512){++count;probability*=lambda/static_cast<float>(count);cdf+=probability;}
    return static_cast<float>(count);
  }
  return std::floor(std::max(0.0f,lambda+std::sqrt(lambda)*norm(seed^0x53484f545f4e4f52ULL))+0.5f);
}
int ix(int v,int n){return std::max(0,std::min(v,n-1));}
float lum(const float* p,int w,int h,int c,int x,int y){x=ix(x,w);y=ix(y,h);const float* q=p+(y*w+x)*c;if(c==1)return q[0];return .2126f*q[0]+.7152f*q[std::min(1,c-1)]+.0722f*q[std::min(2,c-1)];}
}}
namespace augmatch {
void sensor_noise_f32(const float* in,float* out,const SensorNoiseConfig& c,cudaStream_t){valid(c.width,c.height,c.channels);if(!in||!out||c.electrons_per_unit<=0||c.amplifier_noise_electrons<0||c.reset_noise_electrons<0||c.adc_levels<2||c.white_level<=c.black_level||c.exposure<0||c.prnu_stddev<0||c.hot_pixel_probability<0||c.dead_pixel_probability<0||c.stuck_pixel_probability<0||c.stuck_pixel_probability>1||c.fpn_stddev<0||!std::isfinite(c.temperature_celsius)||!std::isfinite(c.reference_temperature_celsius)||!std::isfinite(c.dark_current_temp_coefficient))throw std::invalid_argument("invalid sensor noise configuration");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){auto k=key(c.seed,x,y,ch,0x243f6a8885a308d3ULL);float src=std::max(0.0f,in[(y*c.width+x)*c.channels+ch]);bool dead=uni(k^0x13198a2e03707344ULL)<c.dead_pixel_probability;bool hot=uni(k^0xa4093822299f31d0ULL)<c.hot_pixel_probability;bool stuck=uni(k^0x7f4a7c159e3779b9ULL)<c.stuck_pixel_probability;float prnu=1+c.prnu_stddev*norm(k^0x082efa98ec4e6c89ULL);float fpn=c.fpn_stddev*norm(key(c.seed,x,y,ch,0x6a09e667f3bcc909ULL));float expected=std::max(0.0f,src*c.exposure*prnu*c.electrons_per_unit);float electrons=stuck?c.stuck_pixel_value*c.electrons_per_unit:(dead?0:pois(expected,k^0x452821e638d01377ULL)+c.read_noise_electrons*norm(k^0xc0ac29b7c97c50ddULL)+c.amplifier_noise_electrons*norm(k^0x3c6ef372fe94f82bULL)+c.reset_noise_electrons*norm(k^0xbb67ae8584caa73bULL)+fpn+c.row_noise_electrons*norm(key(c.seed,0,y,ch,0xbe5466cf34e90c6cULL))+c.column_noise_electrons*norm(key(c.seed,x,0,ch,0x9e3779b97f4a7c15ULL))+c.correlated_read_noise_electrons*(norm(key(c.seed,0,y,0,0x510e527fade682d1ULL))+norm(key(c.seed,x,0,1,0x1f83d9abfb41bd6bULL)))*0.5f+c.dark_current_electrons*std::exp(c.dark_current_temp_coefficient*(c.temperature_celsius-c.reference_temperature_celsius))+c.dark_frame_offset_electrons);if(hot)electrons+=c.hot_pixel_electrons;float v=std::max(c.black_level,std::min(c.white_level,c.black_level+electrons/c.electrons_per_unit));float n=(v-c.black_level)/(c.white_level-c.black_level);out[(y*c.width+x)*c.channels+ch]=std::round(n*(c.adc_levels-1))/(c.adc_levels-1);}}
void pixel_response_non_uniformity_f32(const float* in,float* out,const PixelResponseNonUniformityConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.stddev)||c.stddev<0.0f)throw std::invalid_argument("invalid PRNU configuration");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
    const float gain=c.gain_map?c.gain_map[i]:1.0f+c.stddev*norm(key(c.seed,x,y,ch,0x50524e555f474149ULL));
    out[i]=in[i]*gain;
  }
}
void fixed_pattern_offset_noise_f32(const float* in,float* out,const FixedPatternOffsetNoiseConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.stddev)||c.stddev<0.0f)throw std::invalid_argument("invalid FPN configuration");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
    const float offset=c.offset_map?c.offset_map[i]:c.stddev*norm(key(c.seed,x,y,ch,0x46504e5f4f464653ULL));
    out[i]=in[i]+offset;
  }
}
void iso_noise_u8(const std::uint8_t* in,std::uint8_t* out,const ISONoiseConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.iso)||!std::isfinite(c.base_iso)||!std::isfinite(c.analog_gain)||!std::isfinite(c.digital_gain)||!std::isfinite(c.gaussian_stddev)||!std::isfinite(c.chroma_stddev)||c.iso<=0||c.base_iso<=0||c.analog_gain<=0||c.digital_gain<0||c.gaussian_stddev<0||c.chroma_stddev<0)throw std::invalid_argument("invalid ISO noise configuration");
  const float gain=c.iso/c.base_iso*c.analog_gain;
  const float luma_w[3]={0.2126f,0.7152f,0.0722f};
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){
    const std::size_t base=(static_cast<std::size_t>(y)*c.width+x)*c.channels;
    const float luma_z=norm(key(c.seed,x,y,0,0x49534f5f4c554d41ULL));
    float zsum=0.0f;
    for(int ch=0;ch<std::min(c.channels,3);++ch)zsum+=luma_w[ch]*norm(key(c.seed,x,y,ch,0x49534f5f4348524fULL));
    const float chroma_center=c.channels>=3?zsum:0.0f;
    for(int ch=0;ch<c.channels;++ch){
      const float chroma=c.channels>=3?(norm(key(c.seed,x,y,ch,0x49534f5f4348524fULL))-chroma_center):0.0f;
      const float value=(static_cast<float>(in[base+ch])/255.0f)*c.digital_gain+c.gaussian_stddev*gain*luma_z+c.chroma_stddev*gain*chroma;
      out[base+ch]=static_cast<std::uint8_t>(std::floor(std::max(0.0f,std::min(1.0f,value))*255.0f+0.5f));
    }
  }
}
void shot_noise_u8(const std::uint8_t* in,std::uint8_t* out,const ShotNoiseConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.gain)||!std::isfinite(c.scale)||!std::isfinite(c.gain*c.scale)||c.gain<0.0f||c.scale<=0.0f)throw std::invalid_argument("invalid shot noise configuration");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    const auto i=(y*c.width+x)*c.channels+ch;
    const float lambda=(static_cast<float>(in[i])/255.0f)*c.gain*c.scale;
    const float value=shot_pois(lambda,key(c.seed,x,y,ch,0x53484f545f4e4f49ULL))/c.scale;
    out[i]=static_cast<std::uint8_t>(std::floor(std::max(0.0f,std::min(1.0f,value))*255.0f+0.5f));
  }
}
void edge_dependent_noise_f32(const float* in,float* out,const EdgeDependentNoiseConfig& c,cudaStream_t){valid(c.width,c.height,c.channels);if(!in||!out||c.base_stddev<0||c.gradient_scale<0||c.laplacian_scale<0)throw std::invalid_argument("invalid edge noise configuration");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){float gx=lum(in,c.width,c.height,c.channels,x+1,y-1)+2*lum(in,c.width,c.height,c.channels,x+1,y)+lum(in,c.width,c.height,c.channels,x+1,y+1)-lum(in,c.width,c.height,c.channels,x-1,y-1)-2*lum(in,c.width,c.height,c.channels,x-1,y)-lum(in,c.width,c.height,c.channels,x-1,y+1);float gy=lum(in,c.width,c.height,c.channels,x-1,y+1)+2*lum(in,c.width,c.height,c.channels,x,y+1)+lum(in,c.width,c.height,c.channels,x+1,y+1)-lum(in,c.width,c.height,c.channels,x-1,y-1)-2*lum(in,c.width,c.height,c.channels,x,y-1)-lum(in,c.width,c.height,c.channels,x+1,y-1);float l=std::abs(lum(in,c.width,c.height,c.channels,x-1,y)+lum(in,c.width,c.height,c.channels,x+1,y)+lum(in,c.width,c.height,c.channels,x,y-1)+lum(in,c.width,c.height,c.channels,x,y+1)-4*lum(in,c.width,c.height,c.channels,x,y));float sigma=c.base_stddev+c.gradient_scale*.125f*std::sqrt(gx*gx+gy*gy)+c.laplacian_scale*l;for(int ch=0;ch<c.channels;++ch){auto i=(y*c.width+x)*c.channels+ch;out[i]=in[i]+sigma*norm(key(c.seed,x,y,ch,0xdeadbeefULL));}}}
void row_column_correlated_noise_f32(const float* in,float* out,const RowColumnCorrelatedNoiseConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.row_stddev)||!std::isfinite(c.column_stddev)||c.row_stddev<0||c.column_stddev<0||!std::isfinite(c.clip_min)||!std::isfinite(c.clip_max)||c.clip_max<c.clip_min)throw std::invalid_argument("invalid row-column noise configuration");
  if(c.row_map)for(std::size_t i=0;i<static_cast<std::size_t>(c.height)*c.channels;++i)if(!std::isfinite(c.row_map[i]))throw std::invalid_argument("row noise map contains a non-finite value");
  if(c.column_map)for(std::size_t i=0;i<static_cast<std::size_t>(c.width)*c.channels;++i)if(!std::isfinite(c.column_map[i]))throw std::invalid_argument("column noise map contains a non-finite value");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
    const float row=c.row_map?c.row_map[static_cast<std::size_t>(y)*c.channels+ch]:c.row_stddev*norm(key(c.seed,0,y,ch,0x524f575f4e4f4953ULL));
    const float column=c.column_map?c.column_map[static_cast<std::size_t>(x)*c.channels+ch]:c.column_stddev*norm(key(c.seed,x,0,ch,0x434f4c5f4e4f4953ULL));
    out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]+row+column));
  }
}
void clustered_defective_pixels_f32(const float* in,float* out,const ClusteredDefectivePixelsConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||(!c.defects&&c.defect_count)||c.cluster_count<0||c.cluster_radius<0||c.defect_count>100000||c.cluster_count>100000||!std::isfinite(c.hot_value)||!std::isfinite(c.stuck_value))throw std::invalid_argument("invalid clustered defect configuration");
  if(c.defects)for(std::size_t d=0;d<c.defect_count;++d)if(c.defects[d].radius<0||!std::isfinite(c.defects[d].value)||(c.defects[d].kind!=ClusteredDefectKind::Hot&&c.defects[d].kind!=ClusteredDefectKind::Dead&&c.defects[d].kind!=ClusteredDefectKind::Stuck)||(c.defects[d].channel>=c.channels||c.defects[d].channel< -1))throw std::invalid_argument("invalid clustered defect record");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch; float value=in[i];
    if(c.defects){for(std::size_t d=0;d<c.defect_count;++d){const auto& q=c.defects[d];const int dx=x-q.center_x,dy=y-q.center_y;if(q.radius<0||(q.channel!=-1&&q.channel!=ch)||dx*dx+dy*dy>q.radius*q.radius)continue; if(q.kind==ClusteredDefectKind::Dead)value=0.0f;else if(q.kind==ClusteredDefectKind::Stuck)value=q.value;else if(q.kind==ClusteredDefectKind::Hot)value+=q.value;}}
    else for(int d=0;d<c.cluster_count;++d){const std::uint64_t k=key(c.seed,d,0,0,0x434c55535445525fULL);const int cx=std::min(c.width-1,static_cast<int>(uni(k)*c.width)),cy=std::min(c.height-1,static_cast<int>(uni(k^0x9e3779b97f4a7c15ULL)*c.height));const int dx=x-cx,dy=y-cy;if(dx*dx+dy*dy>c.cluster_radius*c.cluster_radius)continue;const float kind=uni(k^0x243f6a8885a308d3ULL);if(kind<1.0f/3.0f)value+=c.hot_value;else if(kind<2.0f/3.0f)value=0.0f;else value=c.stuck_value;}
    out[i]=std::max(0.0f,std::min(1.0f,value));
  }
}
namespace {
void validate_adc(int w,int h,int channels,int levels,const float* in,const float* out,float stddev){
  valid(w,h,channels);
  if(!in||!out||levels<2||levels>65536||!std::isfinite(stddev)||stddev<0.0f)throw std::invalid_argument("invalid ADC non-linearity configuration");
}
float adc_lut_value(const float* lut,std::size_t index,float stddev,std::uint64_t seed,int channel,int code,std::uint64_t tag){
  const float value=lut?lut[index]:1.0f+stddev*norm(key(seed,code,0,channel,tag));
  if(!std::isfinite(value))throw std::invalid_argument("ADC non-linearity LUT contains a non-finite value");
  return value;
}
}
void adc_differential_non_linearity_f32(const float* in,float* out,const AdcDifferentialNonLinearityConfig& c,cudaStream_t){
  validate_adc(c.width,c.height,c.channels,c.levels,in,out,c.stddev);
  for(int ch=0;ch<c.channels;++ch)for(int code=0;code<c.levels;++code){
    const float width=adc_lut_value(c.lut,static_cast<std::size_t>(ch)*c.levels+code,c.stddev,c.seed,ch,code,0x444e4c5f57494454ULL);
    if(width<=0.0f)throw std::invalid_argument("ADC DNL LUT widths must be positive");
  }
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
    const float sample=std::max(0.0f,std::min(1.0f,in[i]));
    if(sample>=1.0f){out[i]=1.0f;continue;}
    float total=0.0f;for(int code=0;code<c.levels;++code){const float width=adc_lut_value(c.lut,static_cast<std::size_t>(ch)*c.levels+code,c.stddev,c.seed,ch,code,0x444e4c5f57494454ULL);total+=width*(code==0||code==c.levels-1?0.5f:1.0f);}
    float cumulative=0.0f;int code=0;for(;code<c.levels-1;++code){const float width=adc_lut_value(c.lut,static_cast<std::size_t>(ch)*c.levels+code,c.stddev,c.seed,ch,code,0x444e4c5f57494454ULL);cumulative+=width*(code==0?0.5f:1.0f);if(sample*total<cumulative)break;}
    out[i]=static_cast<float>(code)/static_cast<float>(c.levels-1);
  }
}
void adc_integral_non_linearity_f32(const float* in,float* out,const AdcIntegralNonLinearityConfig& c,cudaStream_t){
  validate_adc(c.width,c.height,c.channels,c.levels,in,out,c.stddev);
  for(int ch=0;ch<c.channels;++ch)for(int code=0;code<c.levels;++code){const float offset=c.lut?c.lut[static_cast<std::size_t>(ch)*c.levels+code]:c.stddev*norm(key(c.seed,code,0,ch,0x494e4c5f4f464653ULL));if(!std::isfinite(offset))throw std::invalid_argument("ADC INL LUT contains a non-finite value");}
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;const float sample=std::max(0.0f,std::min(1.0f,in[i]));const int code=std::min(c.levels-1,std::max(0,static_cast<int>(std::floor(sample*(c.levels-1)+0.5f))));const float offset=c.lut?c.lut[static_cast<std::size_t>(ch)*c.levels+code]:c.stddev*norm(key(c.seed,code,0,ch,0x494e4c5f4f464653ULL));out[i]=std::max(0.0f,std::min(1.0f,(static_cast<float>(code)+offset)/static_cast<float>(c.levels-1)));}
}
void sensor_well_capacity_variation_f32(const float* in,float* out,const SensorWellCapacityVariationConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.stddev)||c.stddev<0.0f)throw std::invalid_argument("invalid well-capacity configuration");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;
  if(c.capacity_map)for(std::size_t i=0;i<n;++i)if(!std::isfinite(c.capacity_map[i])||c.capacity_map[i]<=0.0f)throw std::invalid_argument("well-capacity map values must be positive and finite");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;const float capacity=c.capacity_map?c.capacity_map[i]:std::max(0.0f,1.0f+c.stddev*norm(key(c.seed,x,y,ch,0x57454c4c5f434150ULL)));out[i]=std::max(0.0f,std::min(1.0f,std::min(capacity,in[i])));}
}
void blooming_vertical_smear_f32(const float* in,float* out,const BloomingVerticalSmearConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.bright_threshold)||!std::isfinite(c.smear_strength)||!std::isfinite(c.smear_decay)||!std::isfinite(c.clip_min)||!std::isfinite(c.clip_max)||c.smear_strength<0.0f||c.smear_decay<0.0f||c.smear_decay>1.0f||c.clip_max<c.clip_min||c.bright_pixel_count>1000000||(!c.bright_pixels&&c.bright_pixel_count)) throw std::invalid_argument("invalid blooming/vertical-smear configuration");
  const std::size_t map_n=static_cast<std::size_t>(c.width)*c.channels;
  if(c.column_smear_map)for(std::size_t i=0;i<map_n;++i)if(!std::isfinite(c.column_smear_map[i])||c.column_smear_map[i]<0.0f)throw std::invalid_argument("column smear map values must be finite and nonnegative");
  if(c.bright_pixels)for(std::size_t p=0;p<c.bright_pixel_count;++p){const BrightPixel& q=c.bright_pixels[p];if(q.x<0||q.x>=c.width||q.y<0||q.y>=c.height||q.channel>=c.channels||q.channel< -1||!std::isfinite(q.value)||q.value<0.0f)throw std::invalid_argument("invalid bright-pixel record");}
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch; float value=in[i];
    if(c.bright_pixels){for(std::size_t p=0;p<c.bright_pixel_count;++p){const BrightPixel& q=c.bright_pixels[p];if(q.x!=x||(q.channel!=-1&&q.channel!=ch)||q.y>y)continue;value+=std::max(0.0f,q.value-c.bright_threshold)*c.smear_strength*std::pow(c.smear_decay,static_cast<float>(y-q.y));}}
    else for(int sy=0;sy<=y;++sy){const float source=in[(static_cast<std::size_t>(sy)*c.width+x)*c.channels+ch];if(source>c.bright_threshold)value+=(source-c.bright_threshold)*c.smear_strength*std::pow(c.smear_decay,static_cast<float>(y-sy));}
    if(c.column_smear_map)value+=c.column_smear_map[static_cast<std::size_t>(x)*c.channels+ch]*c.smear_strength*std::pow(c.smear_decay,static_cast<float>(y));
    out[i]=std::max(c.clip_min,std::min(c.clip_max,value));
  }
}
void sensor_dust_opaque_mask_f32(const float* in,float* out,const SensorDustOpaqueMaskConfig& c,cudaStream_t){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!std::isfinite(c.fill)||c.blur_radius<0)throw std::invalid_argument("invalid sensor dust mask configuration");
  const std::size_t pixels=static_cast<std::size_t>(c.width)*c.height;
  if(!c.mask){for(std::size_t i=0;i<pixels*c.channels;++i)out[i]=in[i];return;}
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const std::size_t pixel=static_cast<std::size_t>(y)*c.width+x;if(c.mask[pixel]==0){for(int ch=0;ch<c.channels;++ch)out[pixel*c.channels+ch]=in[pixel*c.channels+ch];continue;}const int x0=std::max(0,x-c.blur_radius),x1=std::min(c.width-1,x+c.blur_radius),y0=std::max(0,y-c.blur_radius),y1=std::min(c.height-1,y+c.blur_radius);for(int ch=0;ch<c.channels;++ch){float sum=0.0f;int count=0;for(int yy=y0;yy<=y1;++yy)for(int xx=x0;xx<=x1;++xx)if(c.mask[static_cast<std::size_t>(yy)*c.width+xx]==0){sum+=in[(static_cast<std::size_t>(yy)*c.width+xx)*c.channels+ch];++count;}const float value=count?sum/static_cast<float>(count):c.fill;out[pixel*c.channels+ch]=std::max(0.0f,std::min(1.0f,value));}}
}

namespace {
void valid_optical(int w,int h,int c,const float* in,const float* out,float lo,float hi){valid(w,h,c);if(!in||!out||!std::isfinite(lo)||!std::isfinite(hi)||hi<lo)throw std::invalid_argument("invalid optical artifact configuration");}
void valid_map(const float* map,std::size_t n){if(map)for(std::size_t i=0;i<n;++i)if(!std::isfinite(map[i]))throw std::invalid_argument("optical map contains a non-finite value");}
float radial_gain(const LensVignettingConfig& c,int x,int y,float aw,float ah){const float nx=(static_cast<float>(x)/aw-c.center_x)/c.radius_x,ny=(static_cast<float>(y)/ah-c.center_y)/c.radius_y,r2=nx*nx+ny*ny;return c.c0+c.c1*r2+c.c2*r2*r2+c.c3*r2*r2*r2;}
float map_value(const float* map,int mc,int w,int x,int y,int ch){return !map?1.0f:(mc==1?map[static_cast<std::size_t>(y)*w+x]:map[(static_cast<std::size_t>(y)*w+x)*mc+ch]);}
}
void lens_vignetting_f32(const float* in,float* out,const LensVignettingConfig& c,cudaStream_t){valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(!std::isfinite(c.center_x)||!std::isfinite(c.center_y)||!std::isfinite(c.radius_x)||!std::isfinite(c.radius_y)||!std::isfinite(c.c0)||!std::isfinite(c.c1)||!std::isfinite(c.c2)||!std::isfinite(c.c3)||c.radius_x<=0||c.radius_y<=0||c.map_channels!=1&&c.map_channels!=c.channels)throw std::invalid_argument("invalid lens vignetting configuration");valid_map(c.map,c.map_channels==1?static_cast<std::size_t>(c.width)*c.height:static_cast<std::size_t>(c.width)*c.height*c.channels);const float aw=std::max(1,c.width-1),ah=std::max(1,c.height-1);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*radial_gain(c,x,y,aw,ah)*map_value(c.map,c.map_channels,c.width,x,y,ch)));}}
void color_dependent_vignetting_f32(const float* in,float* out,const ColorDependentVignettingConfig& c,cudaStream_t){valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(!std::isfinite(c.center_x)||!std::isfinite(c.center_y)||!std::isfinite(c.radius_x)||!std::isfinite(c.radius_y)||!std::isfinite(c.c0)||!std::isfinite(c.c1)||!std::isfinite(c.c2)||!std::isfinite(c.c3)||c.radius_x<=0||c.radius_y<=0||c.map_channels!=1&&c.map_channels!=c.channels)throw std::invalid_argument("invalid color vignetting configuration");valid_map(c.map,c.map_channels==1?static_cast<std::size_t>(c.width)*c.height:static_cast<std::size_t>(c.width)*c.height*c.channels);if(c.coefficients)valid_map(c.coefficients,static_cast<std::size_t>(c.channels)*4);const float aw=std::max(1,c.width-1),ah=std::max(1,c.height-1);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;float g=radial_gain(c,x,y,aw,ah);if(c.coefficients){const float* p=c.coefficients+static_cast<std::size_t>(ch)*4,nx=(static_cast<float>(x)/aw-c.center_x)/c.radius_x,ny=(static_cast<float>(y)/ah-c.center_y)/c.radius_y,r2=nx*nx+ny*ny;g=p[0]+p[1]*r2+p[2]*r2*r2+p[3]*r2*r2*r2;}out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*g*map_value(c.map,c.map_channels,c.width,x,y,ch)));}}
void optical_falloff_f32(const float* in,float* out,const OpticalFalloffConfig& c,cudaStream_t s){lens_vignetting_f32(in,out,c,s);}
void lens_shading_f32(const float* in,float* out,const LensShadingConfig& c,cudaStream_t){valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(c.map_channels!=1&&c.map_channels!=c.channels)throw std::invalid_argument("invalid lens shading configuration");valid_map(c.map,c.map_channels==1?static_cast<std::size_t>(c.width)*c.height:static_cast<std::size_t>(c.width)*c.height*c.channels);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*c.gain*map_value(c.map,c.map_channels,c.width,x,y,ch)));}}
void uneven_illumination_f32(const float* in,float* out,const UnevenIlluminationConfig& c,cudaStream_t){valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(c.map_channels!=1&&c.map_channels!=c.channels)throw std::invalid_argument("invalid uneven illumination configuration");valid_map(c.map,c.map_channels==1?static_cast<std::size_t>(c.width)*c.height:static_cast<std::size_t>(c.width)*c.height*c.channels);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*(c.base_gain*map_value(c.map,c.map_channels,c.width,x,y,ch))+c.offset));}}
void sensor_lens_dust_shadows_f32(const float* in,float* out,const SensorLensDustShadowsConfig& c,cudaStream_t){valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(!std::isfinite(c.strength)||c.strength<0)throw std::invalid_argument("invalid dust shadow configuration");valid_map(c.shadow_map,static_cast<std::size_t>(c.width)*c.height);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;const float op=c.shadow_map?std::max(0.0f,std::min(1.0f,c.shadow_map[static_cast<std::size_t>(y)*c.width+x])):0.0f;out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*(1.0f-c.strength*op)));}}
namespace {
void valid_effect_map(const float* map,int map_channels,int width,int height,int channels){
  if(map_channels!=1&&map_channels!=channels)throw std::invalid_argument("optical effect map must be HxW or HxWxC");
  if(!map)return;
  const std::size_t n=static_cast<std::size_t>(width)*height*(map_channels==1?1:channels);
  for(std::size_t i=0;i<n;++i)if(!std::isfinite(map[i])||map[i]<0.0f)throw std::invalid_argument("optical effect map must be finite and nonnegative");
}
float effect_map(const float* map,int map_channels,int width,int x,int y,int ch){return !map?1.0f:(map_channels==1?map[static_cast<std::size_t>(y)*width+x]:map[(static_cast<std::size_t>(y)*width+x)*map_channels+ch]);}
float effect_clip(float value,float lo,float hi){return std::max(lo,std::min(hi,value));}
}
void flare_f32(const float* in,float* out,const FlareConfig& c,cudaStream_t){
  valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(!std::isfinite(c.opacity)||c.opacity<0||c.opacity>1||c.source_count>1000000||c.halo_count>1000000||(!c.sources&&c.source_count)||(!c.halos&&c.halo_count))throw std::invalid_argument("invalid flare configuration");
  valid_effect_map(c.map,c.map_channels,c.width,c.height,c.channels);
  for(std::size_t j=0;j<c.source_count;++j){const FlareSource& s=c.sources[j];if(!std::isfinite(s.x)||!std::isfinite(s.y)||!std::isfinite(s.radius)||!std::isfinite(s.intensity)||s.radius<0||s.intensity<0||(s.channel!=-1&&(s.channel<0||s.channel>=c.channels)))throw std::invalid_argument("invalid flare source");}
  for(std::size_t j=0;j<c.halo_count;++j){const FlareHalo& h=c.halos[j];if(!std::isfinite(h.x)||!std::isfinite(h.y)||!std::isfinite(h.radius)||!std::isfinite(h.intensity)||h.radius<0||h.intensity<0||h.source_index< -1||h.source_index>=static_cast<int>(c.source_count))throw std::invalid_argument("invalid flare halo");}
  for(std::size_t p=0;p<static_cast<std::size_t>(c.width)*c.height;++p){int x=static_cast<int>(p%c.width),y=static_cast<int>(p/c.width);for(int ch=0;ch<c.channels;++ch){float value=in[p*c.channels+ch],a=0.0f;
    for(std::size_t j=0;j<c.source_count;++j){const FlareSource& s=c.sources[j];if(s.channel!=-1&&s.channel!=ch)continue;const float dx=x-s.x,dy=y-s.y;if(dx*dx+dy*dy<=s.radius*s.radius)a+=c.opacity*s.intensity;}
    for(std::size_t j=0;j<c.halo_count;++j){const FlareHalo& h=c.halos[j];float hx=h.x,hy=h.y;if(h.source_index>=0){if(static_cast<std::size_t>(h.source_index)>=c.source_count)continue;hx=c.sources[h.source_index].x;hy=c.sources[h.source_index].y;}if(h.radius<=0)continue;const float d=std::sqrt((x-hx)*(x-hx)+(y-hy)*(y-hy));if(d<h.radius)a+=c.opacity*h.intensity*(1.0f-d/h.radius);}
    const float blend=std::max(0.0f,std::min(1.0f,a*effect_map(c.map,c.map_channels,c.width,x,y,ch)));out[p*c.channels+ch]=effect_clip((1.0f-blend)*value+blend,c.clip_min,c.clip_max);
  }}
}
void ghosting_f32(const float* in,float* out,const GhostingConfig& c,cudaStream_t){
  valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(!std::isfinite(c.opacity)||c.opacity<0||c.opacity>1||c.ghost_count>1000000||(!c.ghosts&&c.ghost_count))throw std::invalid_argument("invalid ghosting configuration");valid_effect_map(c.map,c.map_channels,c.width,c.height,c.channels);for(std::size_t j=0;j<c.ghost_count;++j){const Ghost& g=c.ghosts[j];if(!std::isfinite(g.dx)||!std::isfinite(g.dy)||!std::isfinite(g.scale)||!std::isfinite(g.alpha)||g.scale<0||g.alpha<0||g.alpha>1)throw std::invalid_argument("invalid ghost record");}
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;float value=in[i];for(std::size_t j=0;j<c.ghost_count;++j){const Ghost& g=c.ghosts[j];int sx=std::max(0,std::min(c.width-1,static_cast<int>(std::floor(x-g.dx+0.5f)))),sy=std::max(0,std::min(c.height-1,static_cast<int>(std::floor(y-g.dy+0.5f))));const float blend=std::max(0.0f,std::min(1.0f,c.opacity*g.alpha*effect_map(c.map,c.map_channels,c.width,x,y,ch)));value=(1.0f-blend)*value+blend*effect_clip(in[(static_cast<std::size_t>(sy)*c.width+sx)*c.channels+ch]*g.scale,c.clip_min,c.clip_max);}out[i]=effect_clip(value,c.clip_min,c.clip_max);}
}
void veiling_glare_f32(const float* in,float* out,const VeilingGlareConfig& c,cudaStream_t){
  valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(!std::isfinite(c.strength)||c.strength<0)throw std::invalid_argument("invalid veiling glare configuration");valid_effect_map(c.map,c.map_channels,c.width,c.height,c.channels);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const std::size_t p=(static_cast<std::size_t>(y)*c.width+x)*c.channels;float lum=0.0f;for(int ch=0;ch<c.channels;++ch)lum=std::max(lum,in[p+ch]);for(int ch=0;ch<c.channels;++ch)out[p+ch]=effect_clip(in[p+ch]+c.strength*lum*effect_map(c.map,c.map_channels,c.width,x,y,ch),c.clip_min,c.clip_max);}
}
void bloom_f32(const float* in,float* out,const BloomConfig& c,cudaStream_t){
  valid_optical(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);if(c.radius<0||!std::isfinite(c.threshold)||!std::isfinite(c.strength)||c.threshold<0||c.strength<0)throw std::invalid_argument("invalid bloom configuration");valid_effect_map(c.map,c.map_channels,c.width,c.height,c.channels);const int diameter=2*c.radius+1;const float area=static_cast<float>(diameter*diameter);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0;for(int yy=std::max(0,y-c.radius);yy<=std::min(c.height-1,y+c.radius);++yy)for(int xx=std::max(0,x-c.radius);xx<=std::min(c.width-1,x+c.radius);++xx)sum+=std::max(0.0f,in[(static_cast<std::size_t>(yy)*c.width+xx)*c.channels+ch]-c.threshold);const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;out[i]=effect_clip(in[i]+c.strength*sum/area*effect_map(c.map,c.map_channels,c.width,x,y,ch),c.clip_min,c.clip_max);}
}
}
