#include "augmatch/execution.hpp"
#include "augmatch/color.hpp"
#include "augmatch/convert.hpp"
#include "augmatch/filter.hpp"
#include "augmatch/geometric.hpp"
#include "augmatch/image.hpp"
#include "augmatch/noise.hpp"
#include "augmatch/dropout.hpp"
#include "augmatch/dithering.hpp"
#include "augmatch/pixel.hpp"
#include "augmatch/superpixels.hpp"
#include "augmatch/transforms.hpp"
#include "augmatch/tone.hpp"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <stdexcept>
int main(){
  constexpr int w=5,h=4,c=1;
  std::uint8_t view_data[w*h*c]{};
  const auto view=augmatch::make_hwc_u8_view(view_data,w,h,c);
  if(!view.valid()||!view.contiguous()||view.memory!=augmatch::MemorySpace::Host)return 9;
  std::uint8_t convert_in[]={0,128,255}; float convert_f[3]{}; std::uint8_t convert_out[3]{};
  augmatch::to_float_u8(convert_in,convert_f,{3,1,1}); augmatch::from_float_u8(convert_f,convert_out,{3,1,1});
  if(convert_f[0]!=0.f||std::abs(convert_f[1]-128.f/255.f)>1e-6f||convert_out[1]!=128)return 15;
  augmatch::ExecutionContext context;
  context.seed=42;
  if(!context.deterministic)return 10;
  std::vector<float> sensor_in(16,0.5f),sensor_a(16),sensor_b(16);
  augmatch::SensorNoiseConfig sensor{4,4,1};
  sensor.seed=1234;
  augmatch::sensor_noise_f32(sensor_in.data(),sensor_a.data(),sensor);
  augmatch::sensor_noise_f32(sensor_in.data(),sensor_b.data(),sensor);
  for(int i=0;i<16;++i)if(sensor_a[i]!=sensor_b[i]||sensor_a[i]<0.0f||sensor_a[i]>1.0f)return 11;
  std::vector<std::uint8_t> jitter_in(16),jitter_a(16),jitter_b(16);
  for(int i=0;i<16;++i)jitter_in[i]=static_cast<std::uint8_t>(i*17);
  const augmatch::RandomColorJitterConfig jitter_config{2,2,4,-0.1f,0.1f,0.8f,1.2f,0.7f,1.3f,-8.0f,8.0f,77};
  const auto jitter_parameters=augmatch::make_random_color_jitter_config(jitter_config);
  if(jitter_parameters.brightness<-0.1f||jitter_parameters.brightness>0.1f||jitter_parameters.contrast<0.8f||jitter_parameters.contrast>1.2f||jitter_parameters.saturation<0.7f||jitter_parameters.saturation>1.3f||jitter_parameters.hue<-8.0f||jitter_parameters.hue>8.0f)return 40;
  augmatch::random_color_jitter_u8(jitter_in.data(),jitter_a.data(),jitter_config);
  augmatch::random_color_jitter_u8(jitter_in.data(),jitter_b.data(),jitter_config);
  if(jitter_a!=jitter_b||jitter_a[3]!=jitter_in[3]||jitter_a[7]!=jitter_in[7]||jitter_a[11]!=jitter_in[11]||jitter_a[15]!=jitter_in[15])return 41;
  std::vector<std::uint8_t> shot_in(16,128),shot_a(16),shot_b(16);
  augmatch::ShotNoiseConfig shot{4,4,1,1.0f,32.0f,9876};
  augmatch::shot_noise_u8(shot_in.data(),shot_a.data(),shot);
  augmatch::shot_noise_u8(shot_in.data(),shot_b.data(),shot);
  if(shot_a!=shot_b)return 32;
  shot.gain=0.0f;augmatch::shot_noise_u8(shot_in.data(),shot_a.data(),shot);
  for(std::uint8_t value:shot_a)if(value!=0)return 33;
  sensor.stuck_pixel_probability=1.0f; sensor.stuck_pixel_value=0.25f; sensor.fpn_stddev=0.0f;
  augmatch::sensor_noise_f32(sensor_in.data(),sensor_a.data(),sensor);
  for(float v:sensor_a)if(std::abs(v-0.25f)>1.0f/255.0f)return 16;
  augmatch::SensorNoiseConfig temp{1,1,1}; temp.exposure=0; temp.read_noise_electrons=0; temp.dark_current_electrons=100; temp.temperature_celsius=35; temp.dark_current_temp_coefficient=std::log(2.0f)/10.0f; temp.seed=7;
  float temp_in=0,temp_out=0; augmatch::sensor_noise_f32(&temp_in,&temp_out,temp); if(std::abs(temp_out-0.2f)>1.0f/255.0f)return 17;
  std::vector<float> edge(16,0.0f),edge_out(16);
  edge[5]=1.0f;
  augmatch::EdgeDependentNoiseConfig edge_config{4,4,1,0.01f,0.1f,0.1f,9};
  augmatch::edge_dependent_noise_f32(edge.data(),edge_out.data(),edge_config);
  std::vector<std::uint8_t> resize_in{0,10,20,30};
  std::vector<std::uint8_t> resize_out(4);
  augmatch::resize_u8(resize_in.data(),resize_out.data(),{2,2,2,2,1,augmatch::Interpolation::Nearest});
  const std::uint8_t resize_expected[]={0,10,20,30};
  for(int i=0;i<4;++i)if(resize_out[i]!=resize_expected[i])return 12;
  std::vector<std::uint32_t> shuffle_identity{0,1,2,3};
  std::vector<std::uint8_t> shuffle_in(16),shuffle_out(16);
  for(int i=0;i<16;++i)shuffle_in[i]=static_cast<std::uint8_t>(i);
  augmatch::random_grid_shuffle_u8(shuffle_in.data(),shuffle_out.data(),{4,4,1,2,2,shuffle_identity.data(),0});
  if(shuffle_out!=shuffle_in)return 19;
  std::vector<std::uint32_t> shuffle_a(6),shuffle_b(6);
  augmatch::make_random_grid_shuffle_permutation(shuffle_a.data(),2,3,77);
  augmatch::make_random_grid_shuffle_permutation(shuffle_b.data(),2,3,77);
  if(shuffle_a!=shuffle_b)return 20;
  std::vector<float> elastic_dx(static_cast<std::size_t>(w)*h,0.0f),elastic_dy(elastic_dx.size(),0.0f);
  elastic_dx[1*w+1]=1.0f;
  std::vector<std::uint8_t> elastic_in(static_cast<std::size_t>(w)*h),elastic_out(elastic_in.size());
  for(std::size_t i=0;i<elastic_in.size();++i)elastic_in[i]=static_cast<std::uint8_t>(i);
  augmatch::elastic_transform_u8(elastic_in.data(),elastic_out.data(),{w,h,c,elastic_dx.data(),elastic_dy.data(),augmatch::Interpolation::Nearest,99});
  for(int y=0;y<h;++y)for(int x=0;x<w;++x){int source_x=x+((x==1&&y==1)?1:0);std::uint8_t expected_elastic=(source_x<w)?elastic_in[y*w+source_x]:99;if(elastic_out[y*w+x]!=expected_elastic)return 18;}
  std::uint8_t blur_in[]={0,0,0,255,255,255,0,0,0};
  std::uint8_t blur_out[9]{};
  augmatch::blur_u8(blur_in,blur_out,{3,3,1,3,1.0f,augmatch::BlurMode::Average});
  if(blur_out[4]==0||blur_out[4]==255)return 13;
  std::vector<std::uint8_t> tone_in(256*3),tone_out(tone_in.size()),tone_lut(3*256);
  for(int value=0;value<256;++value)for(int channel=0;channel<3;++channel){tone_in[value*3+channel]=static_cast<std::uint8_t>(value);tone_lut[channel*256+value]=static_cast<std::uint8_t>(channel==0?value:channel==1?255-value:(value*37+11)%256);}
  augmatch::random_tone_curve_u8(tone_in.data(),tone_out.data(),{256,1,3,tone_lut.data(),0});
  for(int value=0;value<256;++value)for(int channel=0;channel<3;++channel)if(tone_out[value*3+channel]!=tone_lut[channel*256+value])return 36;
  augmatch::make_random_tone_curve_lut(tone_lut.data(),3,12345);
  for(int channel=0;channel<3;++channel)for(int value=1;value<256;++value)if(tone_lut[channel*256+value]<tone_lut[channel*256+value-1])return 37;
  std::vector<std::uint8_t> nlm_in{0,20,40,60,80,100,120,140,160,180,200,220};
  std::vector<std::uint8_t> nlm_out(nlm_in.size()),nlm_repeat(nlm_in.size());
  augmatch::NonLocalMeansDenoisingConfig nlm_config{4,3,1,1,1,12.0f};
  augmatch::non_local_means_denoising_u8(nlm_in.data(),nlm_out.data(),nlm_config);
  augmatch::non_local_means_denoising_u8(nlm_in.data(),nlm_repeat.data(),nlm_config);
  if(nlm_out!=nlm_repeat)return 29;
  nlm_config.patch_radius=0;nlm_config.search_radius=0;
  augmatch::non_local_means_denoising_u8(nlm_in.data(),nlm_out.data(),nlm_config);
  if(nlm_out!=nlm_in)return 30;
  bool nlm_rejected=false;
  try { nlm_config.search_radius=9; augmatch::non_local_means_denoising_u8(nlm_in.data(),nlm_out.data(),nlm_config); }
  catch(const std::invalid_argument&) { nlm_rejected=true; }
  if(!nlm_rejected)return 31;
  std::vector<std::uint8_t> superpixel_in(5*4),superpixel_out(superpixel_in.size());
  for(std::size_t i=0;i<superpixel_in.size();++i)superpixel_in[i]=static_cast<std::uint8_t>(i);
  augmatch::superpixels_u8(superpixel_in.data(),superpixel_out.data(),{5,4,1,3,2});
  const std::uint8_t superpixel_expected[]={4,4,4,6,6,4,4,4,6,6,14,14,14,16,16,14,14,14,16,16};
  if(!std::equal(superpixel_out.begin(),superpixel_out.end(),superpixel_expected))return 28;
  std::vector<std::uint8_t> advanced_in(5*4,37),advanced_out(advanced_in.size());
  augmatch::advanced_blur_u8(advanced_in.data(),advanced_out.data(),{5,4,1,5,1.25f,2.0f,23.0f});
  for(std::uint8_t value:advanced_out)if(value!=37)return 21;
  std::vector<std::uint8_t> ringing_in{50,100,150},ringing_out(ringing_in.size()),ringing_repeat(ringing_in.size());
  augmatch::RingingOvershootConfig ringing_config{3,1,1,3,1.0f,1.0f};
  augmatch::ringing_overshoot_u8(ringing_in.data(),ringing_out.data(),ringing_config);
  augmatch::ringing_overshoot_u8(ringing_in.data(),ringing_repeat.data(),ringing_config);
  const std::uint8_t ringing_expected[]={23,100,177};
  if(ringing_out!=ringing_repeat||!std::equal(ringing_out.begin(),ringing_out.end(),ringing_expected))return 26;
  ringing_config.amount=0.0f;augmatch::ringing_overshoot_u8(ringing_in.data(),ringing_out.data(),ringing_config);
  if(ringing_out!=ringing_in)return 27;
  if(augmatch::zoom_blur_factor_count({5,4,1,1.0f,1.25f,5})!=6)return 24;
  std::vector<std::uint8_t> zoom_in(5*4),zoom_out(zoom_in.size());
  for(std::size_t i=0;i<zoom_in.size();++i)zoom_in[i]=static_cast<std::uint8_t>(i*11);
  augmatch::zoom_blur_u8(zoom_in.data(),zoom_out.data(),{5,4,1,1.0f,1.25f,5});
  if(zoom_out==zoom_in)return 25;
  const std::size_t glass_count=augmatch::glass_blur_swap_count(9,8,1,2);
  std::vector<augmatch::GlassBlurSwap> glass_a(glass_count),glass_b(glass_count);
  augmatch::make_glass_blur_swap_sequence(glass_a.data(),9,8,1,2,123);
  augmatch::make_glass_blur_swap_sequence(glass_b.data(),9,8,1,2,123);
  for(std::size_t i=0;i<glass_count;++i)if(glass_a[i].dx!=glass_b[i].dx||glass_a[i].dy!=glass_b[i].dy)return 22;
  std::vector<std::uint8_t> glass_in(9*8),glass_out(glass_in.size()),glass_repeat(glass_in.size());
  for(std::size_t i=0;i<glass_in.size();++i)glass_in[i]=static_cast<std::uint8_t>(i*7);
  augmatch::GlassBlurConfig glass_config{9,8,1,0.7f,1,2,nullptr,123};
  augmatch::glass_blur_u8(glass_in.data(),glass_out.data(),glass_config);
  augmatch::glass_blur_u8(glass_in.data(),glass_repeat.data(),glass_config);
  if(glass_out!=glass_repeat)return 23;
  std::vector<std::uint8_t> mask_image{1,2,3,4,5,6,7,8,9,10,11,12};
  const std::uint8_t mask_values[]={0,1,0,255,0,1};
  std::vector<std::uint8_t> mask_output(mask_image.size());
  augmatch::mask_dropout_u8(mask_image.data(),mask_values,mask_output.data(),{3,2,2,231});
  const std::uint8_t mask_expected[]={1,2,231,231,5,6,231,231,9,10,231,231};
  if(!std::equal(mask_output.begin(),mask_output.end(),mask_expected))return 34;
  std::vector<std::uint8_t> mask_repeat(mask_image.size());
  augmatch::mask_dropout_u8(mask_image.data(),mask_values,mask_repeat.data(),{3,2,2,231});
  if(mask_repeat!=mask_output)return 35;
  std::uint8_t pixel_in[]={0,64,128,255};
  std::uint8_t pixel_out[4]{};
  augmatch::gamma_u8(pixel_in,pixel_out,{2,2,1,1.0f});
  for(int i=0;i<4;++i)if(pixel_out[i]!=pixel_in[i])return 14;
  std::vector<std::uint8_t> in(w*h),out(3*2);
  for(int i=0;i<w*h;++i)in[i]=static_cast<std::uint8_t>(i);
  augmatch::Geometry g{w,h,c,1,1,3,2,true,true};
  augmatch::transform_u8(in.data(),out.data(),g);
  const std::uint8_t expected[]={13,12,11,8,7,6};
  for(int i=0;i<6;++i)if(out[i]!=expected[i])return 1;
  auto p=augmatch::transform_point({2.f,2.f},g);
  if(p.x!=1.f||p.y!=0.f)return 2;
  auto b=augmatch::transform_box({1.f,1.f,4.f,3.f},g);
  if(b.x1!=0.f||b.y1!=0.f||b.x2!=3.f||b.y2!=2.f)return 3;
  augmatch::Geometry g2{23,19,3,3,4,13,11,true,true};
  auto clipped=augmatch::transform_box({5.f,6.f,17.f,16.f},g2);
  if(clipped.x1!=0.f||clipped.y1!=0.f||clipped.x2!=11.f||clipped.y2!=9.f)return 4;
  // 2x2 average pooling and nearest-neighbor keep-size expansion.
  std::uint8_t p_in[]={0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150};
  std::uint8_t p_out[16]{};
  augmatch::Pooling pooling{4,4,1,2,2,augmatch::PoolMode::Average,true};
  augmatch::pool_u8(p_in,p_out,pooling);
  const std::uint8_t p_expected[]={25,25,45,45,25,25,45,45,105,105,125,125,105,105,125,125};
  for(int i=0;i<16;++i)if(p_out[i]!=p_expected[i])return 5;
  pooling.mode=augmatch::PoolMode::Maximum;pooling.keep_size=false;
  std::uint8_t max_out[4]{};augmatch::pool_u8(p_in,max_out,pooling);
  const std::uint8_t max_expected[]={50,70,130,150};
  for(int i=0;i<4;++i)if(max_out[i]!=max_expected[i])return 6;
  pooling.mode=augmatch::PoolMode::Minimum;
  std::uint8_t min_out[4]{};augmatch::pool_u8(p_in,min_out,pooling);
  const std::uint8_t min_expected[]={0,20,80,100};
  for(int i=0;i<4;++i)if(min_out[i]!=min_expected[i])return 7;
  pooling.mode=augmatch::PoolMode::Median;
  std::uint8_t median_out[4]{};augmatch::pool_u8(p_in,median_out,pooling);
  const std::uint8_t median_expected[]={25,45,105,125};
  for(int i=0;i<4;++i)if(median_out[i]!=median_expected[i])return 8;
  std::vector<std::uint8_t> clahe_in(9*7*3),clahe_out(clahe_in.size()),clahe_repeat(clahe_in.size());
  for(int y=0;y<7;++y)for(int x=0;x<9;++x)for(int channel=0;channel<3;++channel)
    clahe_in[(y*9+x)*3+channel]=static_cast<std::uint8_t>((x*29+y*17+channel*41)%256);
  const augmatch::CLAHEConfig clahe_config{9,7,3,3.5f,3,2};
  augmatch::clahe_u8(clahe_in.data(),clahe_out.data(),clahe_config);
  augmatch::clahe_u8(clahe_in.data(),clahe_repeat.data(),clahe_config);
  if(clahe_out!=clahe_repeat||clahe_out==clahe_in)return 38;
  bool clahe_rejected=false;
  try { auto invalid=clahe_config; invalid.clip_limit=0.0f; augmatch::clahe_u8(clahe_in.data(),clahe_repeat.data(),invalid); }
  catch(const std::invalid_argument&) { clahe_rejected=true; }
  if(!clahe_rejected)return 39;
  std::vector<std::uint8_t> dither_in(13*9*3),dither_ordered(dither_in.size()),dither_repeat(dither_in.size()),dither_error(dither_in.size());
  for(std::size_t i=0;i<dither_in.size();++i)dither_in[i]=static_cast<std::uint8_t>((i*43+17)%256);
  const augmatch::DitheringConfig ordered_config{13,9,3,3,augmatch::DitheringMode::OrderedBayer4x4};
  augmatch::dithering_u8(dither_in.data(),dither_ordered.data(),ordered_config);
  augmatch::dithering_u8(dither_in.data(),dither_repeat.data(),ordered_config);
  if(dither_ordered!=dither_repeat)return 42;
  const std::uint8_t ordered_levels[]={0,36,73,109,146,182,219,255};
  for(std::uint8_t value:dither_ordered)if(std::find(std::begin(ordered_levels),std::end(ordered_levels),value)==std::end(ordered_levels))return 43;
  augmatch::dithering_u8(dither_in.data(),dither_error.data(),{13,9,3,3,augmatch::DitheringMode::ErrorDiffusionFloydSteinberg});
  for(std::uint8_t value:dither_error)if(std::find(std::begin(ordered_levels),std::end(ordered_levels),value)==std::end(ordered_levels))return 44;
  std::vector<std::uint8_t> dither_identity(dither_in.size());
  augmatch::dither_u8(dither_in.data(),dither_identity.data(),{13,9,3,8,augmatch::DitheringMode::Ordered});
  if(dither_identity!=dither_in)return 45;
  bool dither_rejected=false;
  try { augmatch::dithering_u8(dither_in.data(),dither_identity.data(),{13,9,3,0,augmatch::DitheringMode::Ordered}); }
  catch(const std::invalid_argument&) { dither_rejected=true; }
  if(!dither_rejected)return 46;
  return 0;
}
