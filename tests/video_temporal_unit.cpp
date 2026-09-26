#include "augmatch/noise.hpp"
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  constexpr int T=4,H=2,W=3,C=1; const std::size_t plane=H*W, n=T*plane;
  std::vector<float> in(n), out(n), again(n); for (std::size_t i=0;i<n;++i) in[i]=static_cast<float>(i)/static_cast<float>(n);
  std::vector<std::uint8_t> pm(plane,0); pm[1]=1;
  augmatch::DeadPixelPersistenceConfig d; d.frames=T;d.height=H;d.width=W;d.channels=C;d.mask=pm.data();d.dead_value=.2f;
  augmatch::dead_pixel_persistence_f32(in.data(),out.data(),d); for(int t=0;t<T;++t) if(std::fabs(out[t*plane+1]-.2f)>1e-6f) return 1;
  augmatch::HotPixelPersistenceConfig h; h.frames=T;h.height=H;h.width=W;h.channels=C;h.mask=pm.data();h.hot_value=.9f; augmatch::hot_pixel_persistence_f32(in.data(),out.data(),h); for(int t=0;t<T;++t) if(std::fabs(out[t*plane+1]-.9f)>1e-6f) return 2;
  std::vector<std::uint8_t> drops(T,0); drops[1]=drops[3]=1; augmatch::FrameDropConfig fd;fd.frames=T;fd.height=H;fd.width=W;fd.channels=C;fd.drop_mask=drops.data();fd.fill_value=.0f; augmatch::frame_drops_f32(in.data(),out.data(),fd); if(out[plane]!=in[0]||out[3*plane]!=in[2*plane]) return 3;
  int indices[T]={0,0,2,1}; augmatch::DuplicateFrameConfig du;du.frames=T;du.height=H;du.width=W;du.channels=C;du.source_indices=indices;augmatch::duplicate_frames_f32(in.data(),out.data(),du);if(out[plane]!=in[0]||out[3*plane]!=in[plane])return 4;
  float weights[T]={0,.5f,1,.25f}; augmatch::FrameBlendingConfig bl;bl.frames=T;bl.height=H;bl.width=W;bl.channels=C;bl.weights=weights;augmatch::frame_blending_f32(in.data(),out.data(),bl);if(std::fabs(out[2*plane]-in[plane])>1e-6f)return 5;
  augmatch::TemporalGhostTransform gt[T]={{0,1,0,.5f},{0,1,0,.5f},{0,1,0,.5f},{0,1,0,.5f}}; augmatch::TemporalGhostingConfig gc;gc.frames=T;gc.height=H;gc.width=W;gc.channels=C;gc.transforms=gt;augmatch::temporal_ghosting_f32(in.data(),out.data(),gc);if(std::fabs(out[0]-(.5f*in[0]+.5f*in[1]))>1e-6f)return 6;
  augmatch::MotionCompensationTransform mt[T]={{-1,0,0,1},{-1,0,0,1},{-1,0,0,1},{-1,0,0,1}}; augmatch::MotionCompensationErrorConfig mc;mc.frames=T;mc.height=H;mc.width=W;mc.channels=C;mc.transforms=mt;augmatch::motion_compensation_errors_f32(in.data(),out.data(),mc);if(out[plane]!=in[0])return 7;
  augmatch::RollingShutterTransform rt[T*H];for(auto& z:rt)z={0,0,0};augmatch::VideoRollingShutterConfig rs;rs.frames=T;rs.height=H;rs.width=W;rs.channels=C;rs.transforms=rt;augmatch::video_sensor_rolling_shutter_f32(in.data(),out.data(),rs);if(out!=in)return 8;
  augmatch::DeadPixelPersistenceConfig seeded;seeded.frames=T;seeded.height=H;seeded.width=W;seeded.channels=C;seeded.probability=.3f;seeded.seed=99;augmatch::dead_pixel_persistence_f32(in.data(),out.data(),seeded);augmatch::dead_pixel_persistence_f32(in.data(),again.data(),seeded);if(out!=again)return 9;
  for(float v:out)if(v<0||v>1)return 10; return 0;
}
