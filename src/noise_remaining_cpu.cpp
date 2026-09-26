#include "augmatch/noise.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace augmatch {
namespace {
void valid_image(int w,int h,int c,const float* in,const float* out,float lo,float hi) {
  if(w<=0||h<=0||c<=0||!in||!out||!std::isfinite(lo)||!std::isfinite(hi)||hi<lo)
    throw std::invalid_argument("invalid noise image configuration");
}
void valid_video(const TemporalBatchConfig& c,const float* in,const float* out) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(c.frames<=0||!std::isfinite(c.temporal_correlation)||c.temporal_correlation<-1||c.temporal_correlation>1)
    throw std::invalid_argument("invalid video dimensions or temporal correlation");
}
std::uint64_t mix(std::uint64_t x) { x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
float unit(std::uint64_t x) { return static_cast<float>(mix(x)>>40)/16777216.0f; }
std::uint64_t key(std::uint64_t s,int t,int y,int x,int ch,std::uint64_t tag) { return s^static_cast<std::uint64_t>(t)*0x632be59bd9b4e019ULL^static_cast<std::uint64_t>(y)*0x9e3779b97f4a7c15ULL^static_cast<std::uint64_t>(x)*0x8cb92baa2f2f6f7dULL^static_cast<std::uint64_t>(ch)*0xbf58476d1ce4e5b9ULL^tag; }
float clip(float x,float lo,float hi) { return std::max(lo,std::min(hi,x)); }
float finite_nonnegative(float x,const char* msg) { if(!std::isfinite(x)||x<0) throw std::invalid_argument(msg); return x; }
std::size_t pix(int w,int c,int y,int x,int ch) { return (static_cast<std::size_t>(y)*w+x)*c+ch; }
std::size_t frame_size(const TemporalBatchConfig& c) { return static_cast<std::size_t>(c.height)*c.width*c.channels; }
float step_record(const InterFrameCompressionNoiseConfig& c,int t) { return c.records?c.records[t].quantization_step:c.quantization_step; }
float strength_record(const InterFrameCompressionNoiseConfig& c,int t) { return c.records?c.records[t].strength:c.strength; }
float quantized_residual(float residual,float step) { return step>0 ? std::round(residual/step)*step : residual; }
WaterDroplet droplet_at(const WaterDropletsOnLensConfig& c,int i) {
  if(c.droplets) return c.droplets[i];
  const std::uint64_t b=static_cast<std::uint64_t>(i)*4;
  return {unit(c.seed+b)*c.width,unit(c.seed+b+1)*c.height,
          c.radius_min+(c.radius_max-c.radius_min)*unit(c.seed+b+2),1.0f};
}
}

void random_telegraph_signal_noise_f32(const float* in,float* out,const RandomTelegraphSignalNoiseConfig& c,cudaStream_t) {
  valid_image(c.width,c.height,c.channels,in,out,c.clip_min,c.clip_max);
  if(!std::isfinite(c.low_offset)||!std::isfinite(c.high_offset)||!std::isfinite(c.transition_probability)||c.transition_probability<0||c.transition_probability>1||c.initial_state>1)
    throw std::invalid_argument("invalid random telegraph configuration");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;
  for(std::size_t i=0;i<n;++i) {
    const bool state=c.state_map ? (c.state_map[i]!=0) : ((c.initial_state!=0) ^ (unit(key(c.seed,0,static_cast<int>(i/(c.width*c.channels)),static_cast<int>((i/c.channels)%c.width),static_cast<int>(i%c.channels),0x5254535f53544154ULL))<c.transition_probability));
    out[i]=clip(in[i]+(state?c.high_offset:c.low_offset),c.clip_min,c.clip_max);
  }
}

void inter_frame_compression_noise_f32(const float* in,float* out,const InterFrameCompressionNoiseConfig& c,cudaStream_t) {
  valid_video(c,in,out); finite_nonnegative(c.quantization_step,"invalid inter-frame quantization step"); finite_nonnegative(c.strength,"invalid inter-frame strength");
  const std::size_t plane=frame_size(c);
  for(int t=0;t<c.frames;++t) { const std::size_t base=static_cast<std::size_t>(t)*plane; if(t==0){std::copy(in,in+plane,out);continue;} const float step=finite_nonnegative(step_record(c,t),"invalid inter-frame record step"); const float gain=finite_nonnegative(strength_record(c,t),"invalid inter-frame record strength");
    for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) for(int ch=0;ch<c.channels;++ch){const std::size_t p=pix(c.width,c.channels,y,x,ch),i=base+p;const float r=in[i]-in[base-plane+p],q=quantized_residual(r,step);const float seeded=(unit(key(c.seed,t,y,x,ch,0x4946525f4e4f4953ULL))-.5f)*step*0.1f;out[i]=clip(in[i]+gain*(q-r+seeded),c.clip_min,c.clip_max);}
  }
}

void gop_keyframe_artifacts_f32(const float* in,float* out,const GopKeyframeArtifactsConfig& c,cudaStream_t) {
  valid_video(c,in,out); if(c.gop_size<=0) throw std::invalid_argument("gop_size must be positive"); finite_nonnegative(c.keyframe_strength,"invalid keyframe strength"); finite_nonnegative(c.interframe_strength,"invalid inter-frame strength");
  const std::size_t plane=frame_size(c);
  for(int t=0;t<c.frames;++t){const std::size_t base=static_cast<std::size_t>(t)*plane;const bool k=c.records?c.records[t].keyframe!=0:c.keyframe_mask?c.keyframe_mask[t]!=0:(t%c.gop_size==0);const float gain=c.records?finite_nonnegative(c.records[t].strength,"invalid GOP record strength"):(k?c.keyframe_strength:c.interframe_strength);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const std::size_t p=pix(c.width,c.channels,y,x,ch),i=base+p;float v=in[i];if(k){const float q=std::round(v/0.05f)*0.05f;v+=gain*(q-v);}else{const float r=in[i]-in[base-plane+p];const float q=quantized_residual(r,0.05f);v+=gain*(q-r+(unit(key(c.seed,t,y,x,ch,0x474f505f41525446ULL))-.5f)*0.005f);}out[i]=clip(v,c.clip_min,c.clip_max);}}
}

void block_motion_estimation_artifacts_f32(const float* in,float* out,const BlockMotionEstimationArtifactsConfig& c,cudaStream_t) {
  valid_video(c,in,out); if(c.block_width<=0||c.block_height<=0) throw std::invalid_argument("invalid motion-estimation block size"); finite_nonnegative(c.strength,"invalid motion-estimation strength");
  const int bx=(c.width+c.block_width-1)/c.block_width,by=(c.height+c.block_height-1)/c.block_height; const std::size_t plane=frame_size(c),records=static_cast<std::size_t>(bx)*by;
  for(int t=0;t<c.frames;++t){const std::size_t base=static_cast<std::size_t>(t)*plane; if(t==0){std::copy(in,in+plane,out);continue;} for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){const int bi=y/c.block_height,bj=x/c.block_width;const std::size_t ri=static_cast<std::size_t>(t)*records+static_cast<std::size_t>(bi)*bx+bj;MotionVectorRecord v=c.vectors?c.vectors[ri]:MotionVectorRecord{}; if(!c.vectors){v.dx=static_cast<int>(std::floor(unit(key(c.seed,t,bi,bj,0,0x4d564543544f5253ULL))*5.0f))-2;v.dy=static_cast<int>(std::floor(unit(key(c.seed,t,bi,bj,1,0x4d564543544f5253ULL))*5.0f))-2;v.strength=1.0f;}const int sx=std::max(0,std::min(c.width-1,x+v.dx)),sy=std::max(0,std::min(c.height-1,y+v.dy));const std::size_t p=pix(c.width,c.channels,y,x,ch),q=base-plane+pix(c.width,c.channels,sy,sx,ch);const float a=clip(c.strength*v.strength,0,1);out[base+p]=clip((1-a)*in[base+p]+a*in[q],c.clip_min,c.clip_max);}}
  }

void make_water_droplets(WaterDroplet* output,const WaterDropletsOnLensConfig& c) {
  if(c.width<=0||c.height<=0||c.channels<=0||c.droplet_count<0||!std::isfinite(c.radius_min)||!std::isfinite(c.radius_max)||c.radius_min<=0||c.radius_min>c.radius_max) throw std::invalid_argument("invalid water droplet configuration");
  if(c.droplet_count>0&&!output) throw std::invalid_argument("water droplet output is null");
  for(int i=0;i<c.droplet_count;++i) output[i]=droplet_at(c,i);
}
void water_droplets_on_lens_f32(const float* in,float* out,const WaterDropletsOnLensConfig& c,cudaStream_t) {
  if(c.width<=0||c.height<=0||c.channels<=0||c.droplet_count<0||!in||!out||c.blur_radius<0||c.blur_radius>32||!std::isfinite(c.opacity)||c.opacity<0||c.opacity>1||!std::isfinite(c.radius_min)||!std::isfinite(c.radius_max)||c.radius_min<=0||c.radius_min>c.radius_max||!std::isfinite(c.clip_min)||!std::isfinite(c.clip_max)||c.clip_max<c.clip_min) throw std::invalid_argument("invalid water droplet configuration");
  if(c.droplet_count>0&&c.droplets) for(int i=0;i<c.droplet_count;++i) if(!std::isfinite(c.droplets[i].x)||!std::isfinite(c.droplets[i].y)||!std::isfinite(c.droplets[i].radius)||!std::isfinite(c.droplets[i].opacity)||c.droplets[i].radius<=0||c.droplets[i].opacity<0||c.droplets[i].opacity>1) throw std::invalid_argument("invalid water droplet record");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;std::copy(in,in+n,out);
  for(int d=0;d<c.droplet_count;++d){const WaterDroplet z=droplet_at(c,d);const float r2=z.radius*z.radius;for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const float dx=x+0.5f-z.x,dy=y+0.5f-z.y;if(dx*dx+dy*dy>r2)continue;for(int ch=0;ch<c.channels;++ch){float sum=0;int count=0;for(int oy=-c.blur_radius;oy<=c.blur_radius;++oy)for(int ox=-c.blur_radius;ox<=c.blur_radius;++ox){const int sx=std::max(0,std::min(c.width-1,x+ox)),sy=std::max(0,std::min(c.height-1,y+oy));sum+=in[pix(c.width,c.channels,sy,sx,ch)];++count;}const std::size_t i=pix(c.width,c.channels,y,x,ch);const float a=clip(c.opacity*z.opacity,0,1);out[i]=clip((1-a)*out[i]+a*(sum/count),c.clip_min,c.clip_max);}}}
}
}
