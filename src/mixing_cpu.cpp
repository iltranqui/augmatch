#include "augmatch/mixing.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace augmatch { namespace {
using U = std::uint64_t;
U mix(U x) { x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31); }
float unit(U x) { return static_cast<float>((mix(x) >> 11) * (1.0 / 9007199254740992.0)); }
void dims(int w, int h, int c) { if (w <= 0 || h <= 0 || c <= 0) throw std::invalid_argument("invalid mixing dimensions"); }
void source(const ImageSourceU8& s, int w, int h, int c) { if (!s.data || s.width <= 0 || s.height <= 0 || s.channels != c) throw std::invalid_argument("invalid HWC uint8 source"); (void)w; (void)h; }
void exact(const ImageSourceU8& s, int w, int h, int c) { source(s,w,h,c); if (s.width != w || s.height != h) throw std::invalid_argument("source dimensions do not match output"); }
void output(const std::uint8_t* p, int w, int h, int c) { dims(w,h,c); if (!p) throw std::invalid_argument("null mixing output"); }
std::uint8_t cv(float x) { return static_cast<std::uint8_t>(std::max(0.f, std::min(255.f, std::floor(x + .5f)))); }
float clip(float x) { return std::max(0.f, std::min(1.f, x)); }
int bound(int x, int n) { return std::max(0, std::min(n, x)); }
BoxXYXY clipped(BoxXYXY b, int w, int h) { return clip_box(b,w,h); }
void append(const ReadOnlyMixingTargets& in, MixingTargets& out, float dx=0, float dy=0, BoxXYXY limit={0,0,0,0}, bool use_limit=false) {
  if (in.count && !in.boxes) throw std::invalid_argument("null mixing input boxes");
  for (std::size_t i=0; i<in.count; ++i) {
    if (!out.boxes || out.count >= out.capacity) throw std::invalid_argument("mixing target output capacity is insufficient");
    BoxXYXY b = in.boxes[i]; b.x1 += dx; b.x2 += dx; b.y1 += dy; b.y2 += dy;
    if (use_limit) { b.x1=std::max(b.x1,limit.x1); b.y1=std::max(b.y1,limit.y1); b.x2=std::min(b.x2,limit.x2); b.y2=std::min(b.y2,limit.y2); }
    if (box_is_valid(b)) { out.boxes[out.count]=b; if (out.labels) out.labels[out.count]=in.labels?in.labels[i]:0; ++out.count; }
  }
}
}
void mixup_u8(const ImageSourceU8& a, const ImageSourceU8& b, std::uint8_t* o, const MixUpConfig& c, cudaStream_t) {
  dims(c.width,c.height,c.channels); exact(a,c.width,c.height,c.channels); exact(b,c.width,c.height,c.channels); output(o,c.width,c.height,c.channels);
  float w = c.sample_weight ? unit(c.seed) : c.weight; if (!std::isfinite(w)) throw std::invalid_argument("MixUp weight is not finite"); w=clip(w);
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels; for (std::size_t i=0;i<n;++i) o[i]=cv((1-w)*a.data[i]+w*b.data[i]);
}
void mixup_targets(const ReadOnlyMixingTargets& a, const ReadOnlyMixingTargets& b, float w, MixingTargets& o) {
  if (w<0 || w>1 || !std::isfinite(w) || (a.count+b.count && !o.boxes)) throw std::invalid_argument("invalid MixUp target contract"); o.count=0; append(a,o); append(b,o);
}
void cutmix_u8(const ImageSourceU8& a, const ImageSourceU8& b, std::uint8_t* o, const CutMixConfig& c, cudaStream_t) {
  dims(c.width,c.height,c.channels); exact(a,c.width,c.height,c.channels); exact(b,c.width,c.height,c.channels); output(o,c.width,c.height,c.channels);
  BoxXYXY r=c.box;
  if (c.sample_box) { if (!(c.box_fraction>0 && c.box_fraction<=1)) throw std::invalid_argument("invalid CutMix box fraction"); const int side=std::max(1,static_cast<int>(std::round(std::sqrt(c.box_fraction*c.width*c.height)))); const int cx=static_cast<int>(unit(c.seed)*c.width),cy=static_cast<int>(unit(c.seed+1)*c.height); r={float(cx-side/2),float(cy-side/2),float(cx-side/2+side),float(cy-side/2+side)}; }
  r=clipped(r,c.width,c.height); if (!box_is_valid(r)) throw std::invalid_argument("invalid CutMix box");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) { const bool take=x>=r.x1&&x<r.x2&&y>=r.y1&&y<r.y2; const std::uint8_t* p=(take?b:a).data; const std::size_t i=(static_cast<std::size_t>(y)*c.width+x)*c.channels; for(int ch=0;ch<c.channels;++ch)o[i+ch]=p[i+ch]; }
}
void cutmix_targets(const ReadOnlyMixingTargets& a, const ReadOnlyMixingTargets& b, const CutMixConfig& c, MixingTargets& o) {
  dims(c.width,c.height,c.channels); o.count=0; if (a.count+b.count && !o.boxes) throw std::invalid_argument("invalid CutMix target contract"); BoxXYXY r=clipped(c.box,c.width,c.height); append(a,o); append(b,o,0,0,r,true);
}
void mosaic_u8(const ImageBatchU8& s, std::uint8_t* o, const MosaicConfig& c, cudaStream_t) {
  dims(c.width,c.height,c.channels); if (!s.sources || s.count<4) throw std::invalid_argument("Mosaic requires four sources"); output(o,c.width,c.height,c.channels);
  int cx=c.center_x<0?static_cast<int>(unit(c.seed)*c.width):c.center_x, cy=c.center_y<0?static_cast<int>(unit(c.seed+1)*c.height):c.center_y; cx=bound(cx,c.width); cy=bound(cy,c.height); const int xs[2]={0,cx}, ys[2]={0,cy}, xe[2]={cx,c.width}, ye[2]={cy,c.height};
  for(int q=0;q<4;++q){ const ImageSourceU8& in=s.sources[q]; source(in,c.width,c.height,c.channels); const int col=q&1,row=q>>1; for(int y=ys[row];y<ye[row];++y) for(int x=xs[col];x<xe[col];++x){int sx=std::min(in.width-1,(x-xs[col])*in.width/std::max(1,xe[col]-xs[col]));int sy=std::min(in.height-1,(y-ys[row])*in.height/std::max(1,ye[row]-ys[row]));const std::size_t di=(static_cast<std::size_t>(y)*c.width+x)*c.channels,si=(static_cast<std::size_t>(sy)*in.width+sx)*c.channels;for(int ch=0;ch<c.channels;++ch)o[di+ch]=in.data[si+ch];}}
}
void template_transform_u8(const ImageSourceU8& in, const ImageBatchU8& t, std::uint8_t* o, const TemplateTransformConfig& c, cudaStream_t) {
  dims(c.width,c.height,c.channels); exact(in,c.width,c.height,c.channels); if (t.count && !t.sources) throw std::invalid_argument("null template records"); if (t.count != c.template_count) throw std::invalid_argument("template count mismatch"); output(o,c.width,c.height,c.channels);
  float total=c.image_weight; for(std::size_t k=0;k<t.count;++k){exact(t.sources[k],c.width,c.height,c.channels);total += c.template_weights?c.template_weights[k]:1.f/static_cast<float>(std::max<std::size_t>(1,t.count));} if (!std::isfinite(total)||total<=0) throw std::invalid_argument("invalid TemplateTransform weights");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels; for(std::size_t i=0;i<n;++i){float v=c.image_weight*in.data[i];for(std::size_t k=0;k<t.count;++k)v+=(c.template_weights?c.template_weights[k]:1.f/static_cast<float>(t.count))*t.sources[k].data[i];o[i]=cv(v/total);}
}
void overlay_elements_u8(const ImageSourceU8& base, std::uint8_t* o, const OverlayElementsConfig& c, cudaStream_t) {
  dims(c.width,c.height,c.channels); source(base,c.width,c.height,c.channels); output(o,c.width,c.height,c.channels); if(c.element_count&&!c.elements) throw std::invalid_argument("null overlay elements"); const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels; std::copy(base.data,base.data+n,o);
  for(std::size_t e=0;e<c.element_count;++e){const OverlayElementU8& z=c.elements[e];source(z.image,c.width,c.height,c.channels);if(!std::isfinite(z.alpha)||z.alpha<0||z.alpha>1)throw std::invalid_argument("invalid overlay alpha");for(int y=0;y<z.image.height;++y)for(int x=0;x<z.image.width;++x){int dx=z.x+x,dy=z.y+y;if(dx<0||dy<0||dx>=c.width||dy>=c.height)continue;float a=clip(z.alpha)*(z.mask?z.mask[static_cast<std::size_t>(y)*z.image.width+x]/255.f:1.f);const std::size_t di=(static_cast<std::size_t>(dy)*c.width+dx)*c.channels,si=(static_cast<std::size_t>(y)*z.image.width+x)*c.channels;for(int ch=0;ch<c.channels;++ch)o[di+ch]=cv((1-a)*o[di+ch]+a*z.image.data[si+ch]);}}
}
}
