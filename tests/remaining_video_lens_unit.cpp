#include "augmatch/noise.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
int main(){
  const int T=3,H=2,W=4,C=1; const std::size_t plane=H*W,n=T*plane; std::vector<float> in(n,.4f),a(n),b(n);
  augmatch::RandomTelegraphSignalNoiseConfig r; r.width=W;r.height=H;r.channels=C;r.low_offset=-.1f;r.high_offset=.2f;r.transition_probability=.5f;r.seed=9;
  augmatch::random_telegraph_signal_noise_f32(in.data(),a.data(),r);augmatch::random_telegraph_signal_noise_f32(in.data(),b.data(),r);if(a!=b)return 1;for(float x:a)if(x<0||x>1)return 2;
  // End-point toggle probabilities and borrowed state maps have exact results.
  r.transition_probability=0;r.initial_state=0;r.low_offset=-.1f;r.high_offset=.2f;
  const std::size_t rts_n=static_cast<std::size_t>(H)*W*C;
  augmatch::random_telegraph_signal_noise_f32(in.data(),a.data(),r);for(std::size_t i=0;i<rts_n;++i)if(std::abs(a[i]-.3f)>1e-6f)return 7;
  r.initial_state=1;augmatch::random_telegraph_signal_noise_f32(in.data(),a.data(),r);for(std::size_t i=0;i<rts_n;++i)if(std::abs(a[i]-.6f)>1e-6f)return 8;
std::vector<std::uint8_t> states(rts_n);for(std::size_t i=0;i<rts_n;++i)states[i]=static_cast<std::uint8_t>(i&1);
  r.state_map=states.data();augmatch::random_telegraph_signal_noise_f32(in.data(),a.data(),r);
  for(std::size_t i=0;i<rts_n;++i)if(std::abs(a[i]-(states[i]?.6f:.3f))>1e-6f)return 9;
  r.state_map=nullptr;r.initial_state=2;bool rejected=false;try{augmatch::random_telegraph_signal_noise_f32(in.data(),a.data(),r);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 10;
  augmatch::InterFrameCompressionNoiseConfig f;f.frames=T;f.height=H;f.width=W;f.channels=C;f.quantization_step=.1f;f.strength=1;f.seed=3;augmatch::inter_frame_compression_noise_f32(in.data(),a.data(),f);if(a[0]!=in[0])return 3;
  augmatch::GopFrameRecord gr[T]={{1,1},{0,1},{0,1}};augmatch::GopKeyframeArtifactsConfig g;g.frames=T;g.height=H;g.width=W;g.channels=C;g.records=gr;g.keyframe_strength=.2f;g.interframe_strength=.2f;augmatch::gop_keyframe_artifacts_f32(in.data(),a.data(),g);for(float x:a)if(x<0||x>1)return 4;
  const int blocks=2; std::vector<augmatch::MotionVectorRecord> mv(T*blocks);for(auto& v:mv)v={0,0,1};augmatch::BlockMotionEstimationArtifactsConfig m;m.frames=T;m.height=H;m.width=W;m.channels=C;m.block_width=2;m.block_height=2;m.vectors=mv.data();m.strength=1;augmatch::block_motion_estimation_artifacts_f32(in.data(),a.data(),m);if(a!=in)return 5;
  augmatch::WaterDroplet d{1.5f,.5f,1.2f,1};augmatch::WaterDropletsOnLensConfig w;w.width=W;w.height=H;w.channels=C;w.droplets=&d;w.droplet_count=1;w.opacity=1;w.blur_radius=1;augmatch::water_droplets_on_lens_f32(in.data(),a.data(),w);if(a[0]==in[0]&&a[1]==in[1])return 6;return 0;
}
