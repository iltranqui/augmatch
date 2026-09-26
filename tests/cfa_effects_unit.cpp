#include "augmatch/bayer.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace { void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);} }
int main(){
  constexpr int w=6,h=4; const auto map=augmatch::bayer_plane_map(augmatch::BayerPattern::RGGB);
  std::vector<float> in(w*h),out(w*h),again(w*h); for(int i=0;i<w*h;++i)in[i]=0.1f+0.01f*i;
  require(augmatch::bayer_plane_at(0,0,map)==0&&augmatch::bayer_plane_at(0,1,map)==1&&augmatch::bayer_plane_at(1,0,map)==2&&augmatch::bayer_plane_at(1,1,map)==3,"plane map");
  augmatch::BayerPlaneGainConfig gain{w,h,map,{2.0f,3.0f,4.0f,5.0f},0.0f,1.0f}; augmatch::bayer_plane_gain_f32(in.data(),out.data(),gain);
  require(std::fabs(out[0]-2*in[0])<1e-6f&&std::fabs(out[1]-3*in[1])<1e-6f&&std::fabs(out[w]-4*in[w])<1e-6f&&std::fabs(out[w+1]-5*in[w+1])<1e-6f,"plane gain");
  augmatch::CfaChannelResponseVariationConfig response{w,h,map,{1.0f,0.5f,0.25f,0.75f},0.0f,0.0f,1.0f,0}; augmatch::cfa_channel_response_variation_f32(in.data(),out.data(),response);
  require(std::fabs(out[1]-0.5f*in[1])<1e-6f&&std::fabs(out[w]-0.25f*in[w])<1e-6f,"response variation");
  augmatch::CfaLeakageConfig leak{}; leak.width=w;leak.height=h;leak.plane_map=map; leak.leakage[0]=1;leak.leakage[5]=1;leak.leakage[10]=1;leak.leakage[15]=1; augmatch::cfa_leakage_f32(in.data(),out.data(),leak);
  for(int i=0;i<w*h;++i)require(std::fabs(out[i]-in[i])<1e-6f,"identity leakage");
  std::vector<std::uint8_t> mask(w*h);mask[0]=1; augmatch::CfaMissingSamplesConfig missing{w,h,map,mask.data(),0.0f,augmatch::CfaMissingReplacement::Fill,0.77f,0.0f,1.0f,0}; augmatch::cfa_missing_samples_f32(in.data(),out.data(),missing); require(std::fabs(out[0]-.77f)<1e-6f&&std::fabs(out[1]-in[1])<1e-6f,"missing replacement");
  augmatch::BayerPlaneNoiseConfig noise{w,h,map,{0.0f,0.0f,0.0f,0.0f},0.0f,1.0f,123}; augmatch::bayer_plane_noise_f32(in.data(),out.data(),noise); augmatch::bayer_plane_noise_f32(in.data(),again.data(),noise); for(int i=0;i<w*h;++i)require(out[i]==again[i]&&out[i]==in[i],"deterministic plane noise");
  augmatch::CfaMisregistrationConfig shift{w,h,map,{0,0,0,0},{0,0,0,0},0.0f,1.0f}; augmatch::cfa_misregistration_f32(in.data(),out.data(),shift); for(int i=0;i<w*h;++i)require(out[i]==in[i],"identity misregistration");
  std::cout<<"PASS: CFA effects vectors, maps, clipping, and determinism\n";
}
