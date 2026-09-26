#include "augmatch/noise.hpp"
#include <cmath>
#include <iostream>
#include <vector>
int main(){
  const int T=8,H=2,W=3,C=3; const std::size_t n=static_cast<std::size_t>(T)*H*W*C;
  std::vector<float> in(n,0.4f),a(n),b(n),zero(n);
  augmatch::TemporalGaussianNoiseConfig g;g.frames=T;g.height=H;g.width=W;g.channels=C;g.stddev=.1f;g.temporal_correlation=.7f;g.seed=42;
  augmatch::temporal_gaussian_noise_f32(in.data(),a.data(),g);augmatch::temporal_gaussian_noise_f32(in.data(),b.data(),g);
  if(a!=b){std::cerr<<"seed determinism failed\n";return 1;}
  g.stddev=0;augmatch::temporal_gaussian_noise_f32(in.data(),b.data(),g);if(b!=in){std::cerr<<"zero Gaussian failed\n";return 2;}
  augmatch::FlickerConfig f;f.frames=T;f.height=H;f.width=W;f.channels=C;f.stddev=.2f;f.temporal_correlation=.4f;f.seed=9;
  augmatch::flicker_f32(in.data(),a.data(),f);for(int t=0;t<T;++t){float ref=a[static_cast<std::size_t>(t)*H*W*C];for(std::size_t i=0;i<static_cast<std::size_t>(H)*W*C;++i)if(std::fabs(a[static_cast<std::size_t>(t)*H*W*C+i]-ref)>1e-6f){std::cerr<<"flicker is not frame-global\n";return 3;}}
  augmatch::TemporallyCorrelatedShotNoiseConfig s;s.frames=T;s.height=H;s.width=W;s.channels=C;s.photons_per_unit=100;s.temporal_correlation=.5f;s.seed=3;augmatch::temporally_correlated_shot_noise_f32(in.data(),a.data(),s);
  for(float v:a)if(v<0||v>1){std::cerr<<"shot clipping failed\n";return 4;}
  augmatch::FixedPatternNoiseDriftConfig d;d.frames=T;d.height=H;d.width=W;d.channels=C;d.base_stddev=.1f;d.drift_stddev=0;d.seed=1;augmatch::fixed_pattern_noise_drift_f32(zero.data(),a.data(),d);for(int t=1;t<T;++t)for(std::size_t p=0;p<static_cast<std::size_t>(H)*W*C;++p)if(a[static_cast<std::size_t>(t)*H*W*C+p]!=a[p]){std::cerr<<"FPN base persistence failed\n";return 5;}
  return 0;
}
