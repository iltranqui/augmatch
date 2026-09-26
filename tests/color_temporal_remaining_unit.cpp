#include "augmatch/color/color.hpp"
#include "augmatch/sensor/noise.hpp"
#include <cassert>
#include <vector>
int main(){const int w=8,h=6,c=3;std::vector<unsigned char> in(w*h*c,100),a(in.size()),b(in.size());augmatch::PerChannelGainNoiseConfig g{w,h,c,{1,1,1},.05f,3};augmatch::per_channel_gain_noise_u8(in.data(),a.data(),g);augmatch::per_channel_gain_noise_u8(in.data(),b.data(),g);assert(a==b);augmatch::ColorTemperatureErrorConfig t{w,h,c,3000,.5f};augmatch::color_temperature_error_u8(in.data(),a.data(),t);std::vector<float> fi(3* h*w*c, .5f),fo(fi.size());augmatch::RowNoisePhaseChangesConfig r;r.frames=3;r.height=h;r.width=w;r.channels=c;r.row_stddev=.1f;r.phase_change_probability=.5f;r.seed=9;augmatch::row_noise_phase_changes_f32(fi.data(),fo.data(),r);for(float x:fo)assert(x>=0&&x<=1);return 0;}
