#include "augmatch/sensor/bayer.hpp"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

namespace augmatch {
namespace {
int bayer_channel(int y, int x, BayerPattern pattern) {
  const bool ye = (y & 1) != 0;
  const bool xe = (x & 1) != 0;
  switch (pattern) {
    case BayerPattern::RGGB: return ye ? (xe ? 2 : 1) : (xe ? 1 : 0);
    case BayerPattern::BGGR: return ye ? (xe ? 0 : 1) : (xe ? 1 : 2);
    case BayerPattern::GRBG: return ye ? (xe ? 1 : 2) : (xe ? 0 : 1);
    case BayerPattern::GBRG: return ye ? (xe ? 1 : 0) : (xe ? 2 : 1);
  }
  throw std::invalid_argument("invalid Bayer pattern");
}

void validate_dimensions(const std::uint8_t* in, const std::uint8_t* out,
                         int width, int height, const char* name) {
  if (!in || !out || width <= 0 || height <= 0)
    throw std::invalid_argument(std::string("invalid ") + name + " configuration");
}
int clamp_coord(int value, int limit) { return std::max(0, std::min(limit - 1, value)); }
std::uint8_t clip_round(float value) {
  return static_cast<std::uint8_t>(std::floor(std::max(0.0f, std::min(255.0f, value)) + 0.5f));
}
float raw_at(const std::uint8_t* in, int w, int h, int x, int y) {
  return static_cast<float>(in[clamp_coord(y, h) * w + clamp_coord(x, w)]);
}
float mhc_value(const std::uint8_t* in, int w, int h, int x, int y, int target,
                BayerPattern pattern) {
  const int center = bayer_channel(y, x, pattern);
  if (center == target) return raw_at(in, w, h, x, y);
  int kind = 0; // green at red/blue; the first Malvar filter.
  bool horizontal = true;
  if (target != 1) {
    if (center == 1) {
      horizontal = bayer_channel(clamp_coord(y,h), clamp_coord(x - 1,w), pattern) == target ||
                   bayer_channel(clamp_coord(y,h), clamp_coord(x + 1,w), pattern) == target;
      kind = 1; // red/blue at green, oriented along the matching pair.
    } else {
      kind = 2; // red/blue at the opposite red/blue site.
    }
  }
  static const float green[5][5] = {
      {0,0,-1,0,0}, {0,0,2,0,0}, {-1,2,4,2,-1},
      {0,0,2,0,0}, {0,0,-1,0,0}};
  static const float same[5][5] = {
      {0,0,0.5f,0,0}, {0,-1,0,-1,0}, {-1,4,5,4,-1},
      {0,-1,0,-1,0}, {0,0,0.5f,0,0}};
  static const float diagonal[5][5] = {
      {0,0,-1.5f,0,0}, {0,2,0,2,0}, {-1.5f,0,6,0,-1.5f},
      {0,2,0,2,0}, {0,0,-1.5f,0,0}};
  float sum = 0.0f;
  for (int ky = -2; ky <= 2; ++ky)
    for (int kx = -2; kx <= 2; ++kx) {
      float coefficient = kind == 0 ? green[ky + 2][kx + 2] :
                          (kind == 2 ? diagonal[ky + 2][kx + 2] :
                           (horizontal ? same[ky + 2][kx + 2] : same[kx + 2][ky + 2]));
      sum += coefficient * raw_at(in, w, h, x + kx, y + ky);
    }
  return sum / 8.0f;
}
float edge_value(const std::uint8_t* in, int w, int h, int x, int y, int target,
                 BayerPattern pattern) {
  const int center = bayer_channel(y, x, pattern);
  if (center == target) return raw_at(in, w, h, x, y);
  const auto directional_mean = [&](int dx, int dy) {
    float sum = 0.0f; int count = 0;
    for (int sign : {-1, 1}) {
      for (int distance : {1, 2}) {
        const int xx = clamp_coord(x + sign * distance * dx, w);
        const int yy = clamp_coord(y + sign * distance * dy, h);
        if (bayer_channel(yy, xx, pattern) == target) {
          sum += raw_at(in,w,h,xx,yy); ++count; break;
        }
      }
    }
    return count == 0 ? raw_at(in,w,h,x,y) : sum / count;
  };
  if (center == 1) {
    const float gh = std::fabs(raw_at(in,w,h,x-1,y) - raw_at(in,w,h,x+1,y));
    const float gv = std::fabs(raw_at(in,w,h,x,y-1) - raw_at(in,w,h,x,y+1));
    const float horizontal = directional_mean(1,0);
    const float vertical = directional_mean(0,1);
    if (gh < gv) return horizontal;
    if (gh > gv) return vertical;
    return (horizontal + vertical) * 0.5f;
  }
  const float gd1 = std::fabs(raw_at(in,w,h,x-1,y-1) - raw_at(in,w,h,x+1,y+1));
  const float gd2 = std::fabs(raw_at(in,w,h,x-1,y+1) - raw_at(in,w,h,x+1,y-1));
  const float d1 = directional_mean(1,1);
  const float d2 = directional_mean(1,-1);
  return gd1 < gd2 ? d1 : (gd1 > gd2 ? d2 : (d1 + d2) * 0.5f);
}
}

int quad_bayer_channel_at(int y, int x, QuadBayerPattern pattern) {
  if (y < 0 || x < 0) throw std::invalid_argument("negative Quad-Bayer coordinate");
  return bayer_channel(y >> 1, x >> 1, static_cast<BayerPattern>(pattern));
}

void quad_bayer_sample_u8(const std::uint8_t* in, std::uint8_t* out,
                          const QuadBayerConfig& c, cudaStream_t) {
  validate_dimensions(in, out, c.width, c.height, "Quad-Bayer");
  for (int y = 0; y < c.height; ++y)
    for (int x = 0; x < c.width; ++x) {
      const int pixel = y * c.width + x;
      out[pixel] = in[pixel * 3 + quad_bayer_channel_at(y, x, c.pattern)];
    }
}

int rgbw_channel_at(int y, int x, RGBWPattern pattern) {
  if (y < 0 || x < 0) throw std::invalid_argument("negative RGBW coordinate");
  const bool ye = (y & 1) != 0;
  const bool xe = (x & 1) != 0;
  switch (pattern) {
    case RGBWPattern::RGWB: return ye ? (xe ? 2 : 3) : (xe ? 1 : 0);
    case RGBWPattern::RWBG: return ye ? (xe ? 1 : 2) : (xe ? 3 : 0);
    case RGBWPattern::GBWR: return ye ? (xe ? 0 : 3) : (xe ? 2 : 1);
    case RGBWPattern::WGRB: return ye ? (xe ? 2 : 0) : (xe ? 1 : 3);
  }
  throw std::invalid_argument("invalid RGBW pattern");
}

void rgbw_sample_u8(const std::uint8_t* in, std::uint8_t* out,
                    const RGBWConfig& c, cudaStream_t) {
  validate_dimensions(in, out, c.width, c.height, "RGBW");
  for (int y = 0; y < c.height; ++y)
    for (int x = 0; x < c.width; ++x) {
      const int pixel = y * c.width + x;
      out[pixel] = in[pixel * 4 + rgbw_channel_at(y, x, c.pattern)];
    }
}

int custom_cfa_channel_at(int y, int x, const CustomCfaConfig& c) {
  if (!c.mask || c.mask_width <= 0 || c.mask_height <= 0 || y < 0 || x < 0)
    throw std::invalid_argument("invalid custom CFA mask");
  return static_cast<int>(c.mask[(y % c.mask_height) * c.mask_width + (x % c.mask_width)]);
}

void custom_cfa_sample_u8(const std::uint8_t* in, std::uint8_t* out,
                          const CustomCfaConfig& c, cudaStream_t) {
  validate_dimensions(in, out, c.width, c.height, "custom CFA");
  if (c.channels <= 0 || c.mask_width <= 0 || c.mask_height <= 0 || !c.mask)
    throw std::invalid_argument("invalid custom CFA mask configuration");
  for (int my = 0; my < c.mask_height; ++my)
    for (int mx = 0; mx < c.mask_width; ++mx)
      if (c.mask[my * c.mask_width + mx] >= c.channels)
        throw std::invalid_argument("custom CFA mask channel is outside input planes");
  for (int y = 0; y < c.height; ++y)
    for (int x = 0; x < c.width; ++x) {
      const int pixel = y * c.width + x;
      out[pixel] = in[pixel * c.channels + custom_cfa_channel_at(y, x, c)];
    }
}

void bayer_sample_u8(const std::uint8_t* in, std::uint8_t* out,
                     const BayerConfig& c, cudaStream_t) {
  validate_dimensions(in, out, c.width, c.height, "Bayer");
  for (int y = 0; y < c.height; ++y)
    for (int x = 0; x < c.width; ++x) {
      const int pixel = y * c.width + x;
      out[pixel] = in[pixel * 3 + bayer_channel(y, x, c.pattern)];
    }
}

void bayer_demosaic_nearest_u8(const std::uint8_t* in, std::uint8_t* out,
                               const BayerConfig& c, cudaStream_t) {
  validate_dimensions(in, out, c.width, c.height, "Bayer");
  for (int y = 0; y < c.height; ++y)
    for (int x = 0; x < c.width; ++x)
      for (int ch = 0; ch < 3; ++ch) {
        int sx = x, sy = y;
        for (int r = 0; r <= 2; ++r) {
          bool found = false;
          for (int dy = -r; dy <= r && !found; ++dy)
            for (int dx = -r; dx <= r; ++dx)
              if (std::abs(dx) + std::abs(dy) == r) {
                const int xx = std::max(0, std::min(c.width - 1, x + dx));
                const int yy = std::max(0, std::min(c.height - 1, y + dy));
                if (bayer_channel(yy, xx, c.pattern) == ch) {
                  sx = xx; sy = yy; found = true; break;
                }
              }
          if (found) break;
        }
        out[(y * c.width + x) * 3 + ch] = in[sy * c.width + sx];
      }
}

void bayer_demosaic_bilinear_u8(const std::uint8_t* in, std::uint8_t* out,
                                const BayerConfig& c, cudaStream_t) {
  validate_dimensions(in, out, c.width, c.height, "Bayer");
  for (int y = 0; y < c.height; ++y)
    for (int x = 0; x < c.width; ++x)
      for (int ch = 0; ch < 3; ++ch) {
        int sum = 0, count = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int xx = std::max(0, std::min(c.width - 1, x + dx));
            const int yy = std::max(0, std::min(c.height - 1, y + dy));
            if (bayer_channel(yy, xx, c.pattern) == ch) {
              sum += in[yy * c.width + xx]; ++count;
            }
          }
        out[(y * c.width + x) * 3 + ch] = static_cast<std::uint8_t>((sum + count / 2) / count);
      }
}

void bayer_demosaic_malvar_he_cutler_u8(const std::uint8_t* in, std::uint8_t* out,
                                        const BayerConfig& c, cudaStream_t) {
  validate_dimensions(in, out, c.width, c.height, "Malvar-He-Cutler Bayer");
  for (int y = 0; y < c.height; ++y)
    for (int x = 0; x < c.width; ++x)
      for (int ch = 0; ch < 3; ++ch)
        out[(y * c.width + x) * 3 + ch] = clip_round(mhc_value(in,c.width,c.height,x,y,ch,c.pattern));
}

void bayer_demosaic_edge_aware_u8(const std::uint8_t* in, std::uint8_t* out,
                                  const BayerConfig& c, cudaStream_t) {
  validate_dimensions(in, out, c.width, c.height, "edge-aware Bayer");
  for (int y = 0; y < c.height; ++y)
    for (int x = 0; x < c.width; ++x)
      for (int ch = 0; ch < 3; ++ch)
        out[(y * c.width + x) * 3 + ch] = clip_round(edge_value(in,c.width,c.height,x,y,ch,c.pattern));
}

void bayer_demosaic_malvar_he_cutler_rggb_u8(const std::uint8_t* in, std::uint8_t* out, int w, int h, cudaStream_t s) { BayerConfig c{w,h,BayerPattern::RGGB}; bayer_demosaic_malvar_he_cutler_u8(in,out,c,s); }
void bayer_demosaic_malvar_he_cutler_bggr_u8(const std::uint8_t* in, std::uint8_t* out, int w, int h, cudaStream_t s) { BayerConfig c{w,h,BayerPattern::BGGR}; bayer_demosaic_malvar_he_cutler_u8(in,out,c,s); }
void bayer_demosaic_malvar_he_cutler_grbg_u8(const std::uint8_t* in, std::uint8_t* out, int w, int h, cudaStream_t s) { BayerConfig c{w,h,BayerPattern::GRBG}; bayer_demosaic_malvar_he_cutler_u8(in,out,c,s); }
void bayer_demosaic_malvar_he_cutler_gbrg_u8(const std::uint8_t* in, std::uint8_t* out, int w, int h, cudaStream_t s) { BayerConfig c{w,h,BayerPattern::GBRG}; bayer_demosaic_malvar_he_cutler_u8(in,out,c,s); }
void bayer_demosaic_edge_aware_rggb_u8(const std::uint8_t* in, std::uint8_t* out, int w, int h, cudaStream_t s) { BayerConfig c{w,h,BayerPattern::RGGB}; bayer_demosaic_edge_aware_u8(in,out,c,s); }
void bayer_demosaic_edge_aware_bggr_u8(const std::uint8_t* in, std::uint8_t* out, int w, int h, cudaStream_t s) { BayerConfig c{w,h,BayerPattern::BGGR}; bayer_demosaic_edge_aware_u8(in,out,c,s); }
void bayer_demosaic_edge_aware_grbg_u8(const std::uint8_t* in, std::uint8_t* out, int w, int h, cudaStream_t s) { BayerConfig c{w,h,BayerPattern::GRBG}; bayer_demosaic_edge_aware_u8(in,out,c,s); }
void bayer_demosaic_edge_aware_gbrg_u8(const std::uint8_t* in, std::uint8_t* out, int w, int h, cudaStream_t s) { BayerConfig c{w,h,BayerPattern::GBRG}; bayer_demosaic_edge_aware_u8(in,out,c,s); }

namespace {
void validate_plane_config(int w,int h,const float* in,const float* out,const CfaPlaneMap& map,const char* name) {
  if (!in || !out || w <= 0 || h <= 0) throw std::invalid_argument(std::string("invalid ") + name + " configuration");
  bool seen[4] = {false,false,false,false};
  for (int i=0;i<4;++i) { if (map.plane[i] >= 4 || seen[map.plane[i]]) throw std::invalid_argument("CFA plane map must be a permutation of 0..3"); seen[map.plane[i]]=true; }
}
void validate_clip(float lo,float hi) { if (!std::isfinite(lo)||!std::isfinite(hi)||hi < lo) throw std::invalid_argument("invalid CFA clipping range"); }
std::uint64_t effect_mix(std::uint64_t x) { x += 0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
float effect_uniform(std::uint64_t x) { return static_cast<float>((effect_mix(x)>>11)*(1.0/9007199254740992.0)); }
float effect_normal(std::uint64_t x) { const float u=std::max(effect_uniform(x),1.0e-7f); return std::sqrt(-2.0f*std::log(u))*std::cos(6.28318530718f*effect_uniform(x^0xd1b54a32d192ed03ULL)); }
std::uint64_t effect_key(std::uint64_t seed,int x,int y,int plane,std::uint64_t tag) { return seed^static_cast<std::uint64_t>(x)*0x632be59bd9b4e019ULL^static_cast<std::uint64_t>(y)*0x8cb92baa2f2f6f7dULL^static_cast<std::uint64_t>(plane)*0x9e3779b97f4a7c15ULL^tag; }
int nearest_plane(const float* in,int w,int h,int x,int y,int plane,const CfaPlaneMap& map,int radius) {
  for (int r=0;r<=radius;++r) for (int dy=-r;dy<=r;++dy) for (int dx=-r;dx<=r;++dx) {
    if (std::abs(dx)+std::abs(dy)!=r) continue;
    const int xx=std::max(0,std::min(w-1,x+dx)), yy=std::max(0,std::min(h-1,y+dy));
    if (map.plane[((yy&1)<<1)|(xx&1)]==plane) return static_cast<int>(static_cast<std::size_t>(yy)*w+xx);
  }
  return y*w+x;
}
}

CfaPlaneMap bayer_plane_map(BayerPattern pattern) {
  switch (pattern) {
    case BayerPattern::RGGB: return {{0,1,2,3}};
    case BayerPattern::BGGR: return {{3,2,1,0}};
    case BayerPattern::GRBG: return {{1,0,3,2}};
    case BayerPattern::GBRG: return {{2,3,0,1}};
  }
  throw std::invalid_argument("invalid Bayer pattern");
}
int bayer_plane_at(int y,int x,const CfaPlaneMap& map) {
  if (x<0||y<0) throw std::invalid_argument("negative CFA coordinate");
  return map.plane[((y&1)<<1)|(x&1)];
}

void cfa_channel_response_variation_f32(const float* in,float* out,const CfaChannelResponseVariationConfig& c,cudaStream_t) {
  validate_plane_config(c.width,c.height,in,out,c.plane_map,"CFA response"); validate_clip(c.clip_min,c.clip_max);
  if (!std::isfinite(c.variation_stddev)||c.variation_stddev<0) throw std::invalid_argument("invalid CFA response variation");
  float factor[4]; for (int p=0;p<4;++p) { if (!std::isfinite(c.response[p])) throw std::invalid_argument("non-finite CFA response"); factor[p]=c.response[p]+c.variation_stddev*effect_normal(effect_key(c.seed,p,0,p,0x43524153504f4e53ULL)); }
  for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) { const std::size_t i=static_cast<std::size_t>(y)*c.width+x; const int p=c.plane_map.plane[((y&1)<<1)|(x&1)]; out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*factor[p])); }
}
void cfa_leakage_f32(const float* in,float* out,const CfaLeakageConfig& c,cudaStream_t) {
  validate_plane_config(c.width,c.height,in,out,c.plane_map,"CFA leakage"); validate_clip(c.clip_min,c.clip_max);
  for (int i=0;i<16;++i) if (!std::isfinite(c.leakage[i])||c.leakage[i]<0) throw std::invalid_argument("CFA leakage coefficients must be finite and nonnegative");
  for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) { const std::size_t i=static_cast<std::size_t>(y)*c.width+x; const int p=c.plane_map.plane[((y&1)<<1)|(x&1)]; float value=0; for (int q=0;q<4;++q) value+=c.leakage[p*4+q]*in[nearest_plane(in,c.width,c.height,x,y,q,c.plane_map,2)]; out[i]=std::max(c.clip_min,std::min(c.clip_max,value)); }
}
void cfa_misregistration_f32(const float* in,float* out,const CfaMisregistrationConfig& c,cudaStream_t) {
  validate_plane_config(c.width,c.height,in,out,c.plane_map,"CFA misregistration"); validate_clip(c.clip_min,c.clip_max);
  for (int p=0;p<4;++p) if (std::abs(c.dx[p])>c.width||std::abs(c.dy[p])>c.height) throw std::invalid_argument("CFA misregistration offset is too large");
  for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) { const std::size_t i=static_cast<std::size_t>(y)*c.width+x; const int p=c.plane_map.plane[((y&1)<<1)|(x&1)]; const int xx=std::max(0,std::min(c.width-1,x+c.dx[p])), yy=std::max(0,std::min(c.height-1,y+c.dy[p])); out[i]=std::max(c.clip_min,std::min(c.clip_max,in[nearest_plane(in,c.width,c.height,xx,yy,p,c.plane_map,2)])); }
}
void cfa_missing_samples_f32(const float* in,float* out,const CfaMissingSamplesConfig& c,cudaStream_t) {
  validate_plane_config(c.width,c.height,in,out,c.plane_map,"CFA missing samples"); validate_clip(c.clip_min,c.clip_max);
  if (!std::isfinite(c.missing_probability)||c.missing_probability<0||c.missing_probability>1||!std::isfinite(c.replacement_value)||(c.replacement!=CfaMissingReplacement::Zero&&c.replacement!=CfaMissingReplacement::Fill&&c.replacement!=CfaMissingReplacement::Nearest)) throw std::invalid_argument("invalid CFA missing-sample configuration");
  if (!c.missing_mask && c.missing_probability==0) for (std::size_t i=0,n=static_cast<std::size_t>(c.width)*c.height;i<n;++i) out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]));
  else for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) { const std::size_t i=static_cast<std::size_t>(y)*c.width+x; const bool missing=c.missing_mask?c.missing_mask[i]!=0:effect_uniform(effect_key(c.seed,x,y,0,0x4346414d49535349ULL))<c.missing_probability; if (!missing) out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i])); else if (c.replacement==CfaMissingReplacement::Nearest) out[i]=std::max(c.clip_min,std::min(c.clip_max,in[nearest_plane(in,c.width,c.height,x,y,c.plane_map.plane[((y&1)<<1)|(x&1)],c.plane_map,2)])); else out[i]=std::max(c.clip_min,std::min(c.clip_max,c.replacement==CfaMissingReplacement::Fill?c.replacement_value:0.0f)); }
}
void bayer_plane_noise_f32(const float* in,float* out,const BayerPlaneNoiseConfig& c,cudaStream_t) {
  validate_plane_config(c.width,c.height,in,out,c.plane_map,"Bayer plane noise"); validate_clip(c.clip_min,c.clip_max);
  for (int p=0;p<4;++p) if (!std::isfinite(c.stddev[p])||c.stddev[p]<0) throw std::invalid_argument("invalid Bayer plane noise");
  for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) { const std::size_t i=static_cast<std::size_t>(y)*c.width+x; const int p=c.plane_map.plane[((y&1)<<1)|(x&1)]; out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]+c.stddev[p]*effect_normal(effect_key(c.seed,x,y,p,0x42504e4f49534531ULL)))); }
}
void bayer_plane_gain_f32(const float* in,float* out,const BayerPlaneGainConfig& c,cudaStream_t) {
  validate_plane_config(c.width,c.height,in,out,c.plane_map,"Bayer plane gain"); validate_clip(c.clip_min,c.clip_max);
  for (int p=0;p<4;++p) if (!std::isfinite(c.gain[p])||c.gain[p]<0) throw std::invalid_argument("invalid Bayer plane gain");
  for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) { const std::size_t i=static_cast<std::size_t>(y)*c.width+x; const int p=c.plane_map.plane[((y&1)<<1)|(x&1)]; out[i]=std::max(c.clip_min,std::min(c.clip_max,in[i]*c.gain[p])); }
}

namespace {
void validate_artifact(int w,int h,const std::uint8_t* in,const std::uint8_t* out,int lo,int hi,const char* name) {
  if (!in||!out||w<=0||h<=0||lo<0||hi>255||hi<lo) throw std::invalid_argument(std::string("invalid ")+name+" configuration");
}
std::uint8_t artifact_clip(float v,int lo,int hi) {
  return static_cast<std::uint8_t>(std::floor(std::max(static_cast<float>(lo),std::min(static_cast<float>(hi),v))+0.5f));
}
float directional_delta(const std::uint8_t* in,int w,int h,int x,int y,int target,BayerPattern pattern,int direction) {
  if (bayer_channel(y,x,pattern)==target) return 0.0f;
  if (bayer_channel(y,x,pattern)!=1) return 0.0f;
  const float gh=std::fabs(raw_at(in,w,h,x-1,y)-raw_at(in,w,h,x+1,y));
  const float gv=std::fabs(raw_at(in,w,h,x,y-1)-raw_at(in,w,h,x,y+1));
  const auto mean=[&](int dx,int dy) { float sum=0; int n=0; for (int sign : {-1,1}) for (int d : {1,2}) { const int xx=clamp_coord(x+sign*d*dx,w), yy=clamp_coord(y+sign*d*dy,h); if (bayer_channel(yy,xx,pattern)==target) {sum+=raw_at(in,w,h,xx,yy);++n;break;} } return n?sum/n:raw_at(in,w,h,x,y); };
  const float horizontal=mean(1,0), vertical=mean(0,1);
  const float selected=direction==1?horizontal:(direction==2?vertical:(gh<gv?horizontal:(gh>gv?vertical:(horizontal+vertical)*0.5f)));
  const float other=direction==1?vertical:(direction==2?horizontal:(gh<gv?vertical:(gh>gv?horizontal:(horizontal+vertical)*0.5f)));
  return selected-other;
}
}

void bayer_directional_demosaicing_artifacts_u8(const std::uint8_t* in,std::uint8_t* out,const DirectionalDemosaicingArtifactsConfig& c,cudaStream_t) {
  validate_artifact(c.width,c.height,in,out,c.clip_min,c.clip_max,"directional demosaicing artifacts");
  if (c.direction<0||c.direction>2||!std::isfinite(c.strength)||c.strength<0) throw std::invalid_argument("invalid directional demosaicing artifact parameters");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) for(int ch=0;ch<3;++ch) { const int i=(y*c.width+x)*3; const float base=edge_value(in,c.width,c.height,x,y,ch,c.pattern); const float d=directional_delta(in,c.width,c.height,x,y,ch,c.pattern,c.direction); out[i+ch]=artifact_clip(base+c.strength*d,c.clip_min,c.clip_max); }
}
void bayer_false_color_zipper_artifacts_u8(const std::uint8_t* in,std::uint8_t* out,const FalseColorZipperArtifactsConfig& c,cudaStream_t) {
  validate_artifact(c.width,c.height,in,out,c.clip_min,c.clip_max,"false-color zipper artifacts");
  if(!std::isfinite(c.strength)||c.strength<0) throw std::invalid_argument("invalid false-color zipper strength");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) { const float edge=std::fabs(raw_at(in,c.width,c.height,x-1,y)-raw_at(in,c.width,c.height,x+1,y)); const float d=c.strength*edge*(((x+y)&1)?-1.0f:1.0f); for(int ch=0;ch<3;++ch) { const int i=(y*c.width+x)*3; const float sign=ch==0?d:(ch==2?-d:0.0f); const float base=edge_value(in,c.width,c.height,x,y,ch,c.pattern); out[i+ch]=artifact_clip(base+(bayer_channel(y,x,c.pattern)==ch?0.0f:sign),c.clip_min,c.clip_max); } }
}
void bayer_demosaicing_aliasing_u8(const std::uint8_t* in,std::uint8_t* out,const DemosaicingAliasingConfig& c,cudaStream_t) {
  validate_artifact(c.width,c.height,in,out,c.clip_min,c.clip_max,"demosaicing aliasing");
  if(c.period<=0||!std::isfinite(c.phase)||!std::isfinite(c.strength)||c.strength<0) throw std::invalid_argument("invalid demosaicing aliasing parameters");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) for(int ch=0;ch<3;++ch) { const int i=(y*c.width+x)*3; const float wave=std::sin(6.28318530718f*(static_cast<float>(x+y)/c.period)+c.phase)*c.strength*127.5f; const float sign=ch==0?wave:(ch==2?-wave:0.0f); const float base=edge_value(in,c.width,c.height,x,y,ch,c.pattern); out[i+ch]=artifact_clip(base+(bayer_channel(y,x,c.pattern)==ch?0.0f:sign),c.clip_min,c.clip_max); }
}
void bayer_demosaicing_ringing_u8(const std::uint8_t* in,std::uint8_t* out,const DemosaicingRingingConfig& c,cudaStream_t) {
  validate_artifact(c.width,c.height,in,out,c.clip_min,c.clip_max,"demosaicing ringing");
  if(!std::isfinite(c.strength)||c.strength<0) throw std::invalid_argument("invalid demosaicing ringing strength");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) { const float center=raw_at(in,c.width,c.height,x,y); const float lap=center-(raw_at(in,c.width,c.height,x-1,y)+raw_at(in,c.width,c.height,x+1,y)+raw_at(in,c.width,c.height,x,y-1)+raw_at(in,c.width,c.height,x,y+1))*0.25f; for(int ch=0;ch<3;++ch) { const int i=(y*c.width+x)*3; const float base=edge_value(in,c.width,c.height,x,y,ch,c.pattern); out[i+ch]=artifact_clip(base+(bayer_channel(y,x,c.pattern)==ch?0.0f:c.strength*lap),c.clip_min,c.clip_max); } }
}
void bayer_demosaicing_noise_amplification_u8(const std::uint8_t* in,std::uint8_t* out,const DemosaicingNoiseAmplificationConfig& c,cudaStream_t) {
  validate_artifact(c.width,c.height,in,out,c.clip_min,c.clip_max,"demosaicing noise amplification");
  if(!std::isfinite(c.noise_stddev)||c.noise_stddev<0||!std::isfinite(c.amplification)||c.amplification<0) throw std::invalid_argument("invalid demosaicing noise amplification parameters");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) { const float gradient=std::fabs(raw_at(in,c.width,c.height,x-1,y)-raw_at(in,c.width,c.height,x+1,y))+std::fabs(raw_at(in,c.width,c.height,x,y-1)-raw_at(in,c.width,c.height,x,y+1)); for(int ch=0;ch<3;++ch) { const int i=(y*c.width+x)*3; const float gain=1.0f+c.amplification*gradient/510.0f; const float base=edge_value(in,c.width,c.height,x,y,ch,c.pattern); const float noise=c.noise_stddev*gain*effect_normal(effect_key(c.seed,x,y,ch,0x444e4f4953454152ULL)); out[i+ch]=artifact_clip(base+noise,c.clip_min,c.clip_max); } }
}
}
