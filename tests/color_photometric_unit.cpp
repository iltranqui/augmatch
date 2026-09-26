#include "augmatch/color.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
int main(){
  const int w=2,h=1,c=4; const std::uint8_t input[]={100,150,200,77,255,0,0,88}; std::uint8_t output[8]{};
  augmatch::ColorMatrixPerturbationConfig matrix{w,h,c};
  augmatch::color_matrix_perturbation_u8(input,output,matrix); for(int i=0;i<8;++i)assert(output[i]==input[i]);
  matrix.matrix[0]=0;matrix.matrix[1]=1;matrix.matrix[2]=0;
  augmatch::color_matrix_perturbation_u8(input,output,matrix);assert(output[0]==150&&output[1]==150&&output[2]==200&&output[3]==77);
  augmatch::ColorClippingConfig clip{w,h,c};clip.clip_min[0]=.25f;clip.clip_max[0]=.75f;augmatch::color_clipping_u8(input,output,clip);assert(output[0]==100&&output[4]==191&&output[3]==77);
  augmatch::WhiteBalanceClippingConfig wb{w,h,c};wb.gains[0]=2;augmatch::white_balance_clipping_u8(input,output,wb);assert(output[0]==200&&output[4]==255&&output[3]==77);
  augmatch::ChromaNoiseConfig noise{w,h,c,0,42};augmatch::chroma_noise_u8(input,output,noise);for(int i=0;i<8;++i)assert(output[i]==input[i]);
  augmatch::LumaNoiseConfig luma{w,h,c,0,42};augmatch::luma_noise_u8(input,output,luma);for(int i=0;i<8;++i)assert(output[i]==input[i]);
  augmatch::CameraColorProfileVariationConfig profile{w,h,c};profile.gains[1]=.5f;profile.noise_stddev=0;augmatch::camera_color_profile_variation_u8(input,output,profile);assert(output[1]==75&&output[4]==255);
  augmatch::SensorSpectralResponseVariationConfig spectral{w,h,c};spectral.response_stddev=0;augmatch::sensor_spectral_response_variation_u8(input,output,spectral);for(int i=0;i<8;++i)assert(output[i]==input[i]);
  augmatch::CorrelatedLumaChromaNoiseConfig correlated{w,h,c,0,0,0,3};augmatch::correlated_luma_chroma_noise_u8(input,output,correlated);for(int i=0;i<8;++i)assert(output[i]==input[i]);
  return 0;
}
