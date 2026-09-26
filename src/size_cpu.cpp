#include "augmatch/size.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace augmatch { namespace {
void check(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid size dimensions");}
int refl(int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;}
std::uint64_t splitmix64(std::uint64_t value){value+=0x9e3779b97f4a7c15ULL;value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;value=(value^(value>>27))*0x94d049bb133111ebULL;return value^(value>>31);}
int crop_limit(float fraction,int size){if(!std::isfinite(fraction)||fraction<0.0f||fraction>1.0f)throw std::invalid_argument("crop fractions must be finite values in [0,1]");return static_cast<int>(std::floor(fraction*static_cast<float>(size)));}
int choose_offset(int explicit_offset,int maximum,std::uint64_t seed,int side){if(explicit_offset>=0)return explicit_offset;return static_cast<int>(splitmix64(seed+static_cast<std::uint64_t>(side))%(static_cast<std::uint64_t>(maximum)+1ULL));}
RandomResizedCropRectangle rectangle(const RandomResizedCropConfig& c){
  check(c.input_width,c.input_height,c.channels);
  if(c.output_width<=0||c.output_height<=0||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>2) throw std::invalid_argument("invalid random resized crop output configuration");
  if(c.crop_width<0||c.crop_height<0||(c.crop_width==0)!=(c.crop_height==0)) throw std::invalid_argument("crop dimensions must both be explicit or both be zero");
  const int width=c.crop_width==0?1+static_cast<int>(splitmix64(c.seed)%(static_cast<std::uint64_t>(c.input_width))):c.crop_width;
  const int height=c.crop_height==0?1+static_cast<int>(splitmix64(c.seed+1)%(static_cast<std::uint64_t>(c.input_height))):c.crop_height;
  if(width<=0||height<=0||width>c.input_width||height>c.input_height) throw std::invalid_argument("random resized crop rectangle exceeds input");
  const int x=c.crop_x<0?static_cast<int>(splitmix64(c.seed+2)%(static_cast<std::uint64_t>(c.input_width-width+1))):c.crop_x;
  const int y=c.crop_y<0?static_cast<int>(splitmix64(c.seed+3)%(static_cast<std::uint64_t>(c.input_height-height+1))):c.crop_y;
  if(c.crop_x<-1||c.crop_y<-1||x<0||y<0||x+width>c.input_width||y+height>c.input_height) throw std::invalid_argument("random resized crop rectangle is outside input");
  return {x,y,width,height};
}
}
RandomResizedCropRectangle make_random_resized_crop_rectangle(const RandomResizedCropConfig& c){return rectangle(c);}
void random_crop_u8(const std::uint8_t* in,std::uint8_t* out,const RandomCropConfig& c,cudaStream_t){check(c.input_width,c.input_height,c.channels);if(!in||!out||c.output_width<=0||c.output_height<=0||c.output_width>c.input_width||c.output_height>c.input_height||c.left<0||c.top<0||c.left+c.output_width>c.input_width||c.top+c.output_height>c.input_height)throw std::invalid_argument("invalid random crop configuration");for(int y=0;y<c.output_height;++y)for(int x=0;x<c.output_width;++x)for(int ch=0;ch<c.channels;++ch)out[(y*c.output_width+x)*c.channels+ch]=in[((y+c.top)*c.input_width+x+c.left)*c.channels+ch];}
void center_crop_u8(const std::uint8_t* in,std::uint8_t* out,const CenterCropConfig& c,cudaStream_t){check(c.input_width,c.input_height,c.channels);if(!in||!out||c.output_width<=0||c.output_height<=0||c.output_width>c.input_width||c.output_height>c.input_height)throw std::invalid_argument("invalid center crop configuration");int ox=(c.input_width-c.output_width)/2,oy=(c.input_height-c.output_height)/2;for(int y=0;y<c.output_height;++y)for(int x=0;x<c.output_width;++x)for(int ch=0;ch<c.channels;++ch)out[(y*c.output_width+x)*c.channels+ch]=in[((y+oy)*c.input_width+x+ox)*c.channels+ch];}
void random_resized_crop_u8(const std::uint8_t* in,std::uint8_t* out,const RandomResizedCropConfig& c,cudaStream_t s){
  if(!in||!out) throw std::invalid_argument("null random resized crop buffer");
  const RandomResizedCropRectangle r=make_random_resized_crop_rectangle(c);
  std::vector<std::uint8_t> cropped(static_cast<std::size_t>(r.width)*r.height*c.channels);
  random_crop_u8(in,cropped.data(),{c.input_width,c.input_height,r.width,r.height,c.channels,r.x,r.y},s);
  resize_u8(cropped.data(),out,{r.width,r.height,c.output_width,c.output_height,c.channels,c.interpolation},s);
}
void random_sized_crop_u8(const std::uint8_t* in,std::uint8_t* out,const RandomSizedCropConfig& c,cudaStream_t s){random_resized_crop_u8(in,out,c,s);}
RandomCropFromBordersOffsets make_random_crop_from_borders_offsets(const RandomCropFromBordersConfig& c){
  check(c.input_width,c.input_height,c.channels);
  const int max_left=crop_limit(c.crop_left,c.input_width),max_right=crop_limit(c.crop_right,c.input_width),max_top=crop_limit(c.crop_top,c.input_height),max_bottom=crop_limit(c.crop_bottom,c.input_height);
  const int left_bound=c.left_offset>=0?c.left_offset:max_left,right_bound=c.right_offset>=0?c.right_offset:max_right,top_bound=c.top_offset>=0?c.top_offset:max_top,bottom_bound=c.bottom_offset>=0?c.bottom_offset:max_bottom;
  if(static_cast<long long>(left_bound)+right_bound>=c.input_width||static_cast<long long>(top_bound)+bottom_bound>=c.input_height)throw std::invalid_argument("border crop fractions and offsets can remove the entire dimension");
  if(c.left_offset < -1 || c.right_offset < -1 || c.top_offset < -1 || c.bottom_offset < -1 ||
     (c.left_offset >= 0 && c.left_offset > max_left) || (c.right_offset >= 0 && c.right_offset > max_right) ||
     (c.top_offset >= 0 && c.top_offset > max_top) || (c.bottom_offset >= 0 && c.bottom_offset > max_bottom))
    throw std::invalid_argument("border crop offset exceeds its fraction limit");
  RandomCropFromBordersOffsets result;
  result.left=choose_offset(c.left_offset,max_left,c.seed,0);result.right=choose_offset(c.right_offset,max_right,c.seed,1);
  result.top=choose_offset(c.top_offset,max_top,c.seed,2);result.bottom=choose_offset(c.bottom_offset,max_bottom,c.seed,3);
  if(static_cast<long long>(result.left)+result.right>=c.input_width||static_cast<long long>(result.top)+result.bottom>=c.input_height)throw std::invalid_argument("border crop offsets can remove the entire dimension");
  return result;
}
void random_crop_from_borders_u8(const std::uint8_t* in,std::uint8_t* out,const RandomCropFromBordersConfig& c,cudaStream_t){
  check(c.input_width,c.input_height,c.channels);if(!in||!out)throw std::invalid_argument("null border crop buffer");
  const RandomCropFromBordersOffsets offsets=make_random_crop_from_borders_offsets(c);const int ow=c.input_width-offsets.left-offsets.right,oh=c.input_height-offsets.top-offsets.bottom;
  for(int y=0;y<oh;++y)for(int x=0;x<ow;++x)for(int ch=0;ch<c.channels;++ch)out[(y*ow+x)*c.channels+ch]=in[((y+offsets.top)*c.input_width+x+offsets.left)*c.channels+ch];
}
RandomCropNearBBoxOffsets make_random_crop_near_bbox_offsets(const RandomCropNearBBoxConfig& c){
  check(c.input_width,c.input_height,c.channels);
  const BoxXYXY b=c.bbox;
  if(!std::isfinite(b.x1)||!std::isfinite(b.y1)||!std::isfinite(b.x2)||!std::isfinite(b.y2)||b.x1<0.0f||b.y1<0.0f||b.x2>c.input_width||b.y2>c.input_height||b.x2<=b.x1||b.y2<=b.y1)
    throw std::invalid_argument("RandomCropNearBBox requires a non-empty in-image box");
  if(!std::isfinite(c.max_part_shift_x)||!std::isfinite(c.max_part_shift_y)||c.max_part_shift_x<0.0f||c.max_part_shift_y<0.0f)
    throw std::invalid_argument("RandomCropNearBBox max_part_shift values must be finite and nonnegative");
  const auto maximum=[](float fraction,float extent){const double value=std::floor(static_cast<double>(fraction)*static_cast<double>(extent));if(value>static_cast<double>(std::numeric_limits<int>::max()))throw std::invalid_argument("RandomCropNearBBox margin is too large");return static_cast<int>(value);};
  const int max_x=maximum(c.max_part_shift_x,b.x2-b.x1),max_y=maximum(c.max_part_shift_y,b.y2-b.y1);
  if(c.left_offset < -1 || c.right_offset < -1 || c.top_offset < -1 || c.bottom_offset < -1 ||
     (c.left_offset>=0&&c.left_offset>max_x)||(c.right_offset>=0&&c.right_offset>max_x)||
     (c.top_offset>=0&&c.top_offset>max_y)||(c.bottom_offset>=0&&c.bottom_offset>max_y))
    throw std::invalid_argument("RandomCropNearBBox offset exceeds its margin limit");
  RandomCropNearBBoxOffsets result;
  result.left=choose_offset(c.left_offset,max_x,c.seed,0);result.right=choose_offset(c.right_offset,max_x,c.seed,1);
  result.top=choose_offset(c.top_offset,max_y,c.seed,2);result.bottom=choose_offset(c.bottom_offset,max_y,c.seed,3);
  return result;
}
RandomCropNearBBoxRectangle make_random_crop_near_bbox_rectangle(const RandomCropNearBBoxConfig& c){
  const RandomCropNearBBoxOffsets o=make_random_crop_near_bbox_offsets(c);
  const int x0=std::max(0,static_cast<int>(std::floor(c.bbox.x1))-o.left);
  const int y0=std::max(0,static_cast<int>(std::floor(c.bbox.y1))-o.top);
  const int x1=std::min(c.input_width,static_cast<int>(std::ceil(c.bbox.x2))+o.right);
  const int y1=std::min(c.input_height,static_cast<int>(std::ceil(c.bbox.y2))+o.bottom);
  return {x0,y0,x1-x0,y1-y0};
}
void random_crop_near_bbox_u8(const std::uint8_t* in,std::uint8_t* out,const RandomCropNearBBoxConfig& c,cudaStream_t s){
  if(!in||!out)throw std::invalid_argument("null RandomCropNearBBox buffer");
  const RandomCropNearBBoxRectangle r=make_random_crop_near_bbox_rectangle(c);
  random_crop_u8(in,out,{c.input_width,c.input_height,r.width,r.height,c.channels,r.x,r.y},s);
}
struct BBoxEnvelope { int x1=0,y1=0,x2=0,y2=0; };
void validate_target_config(int width,int height,int channels,const BoxXYXY* boxes,std::size_t count,float erosion){
  check(width,height,channels);
  if(count!=0&&!boxes) throw std::invalid_argument("target-aware crop box array is null");
  if(!std::isfinite(erosion)||erosion<0.0f||erosion>1.0f) throw std::invalid_argument("target-aware crop erosion must be finite in [0,1]");
  for(std::size_t i=0;i<count;++i){const BoxXYXY b=boxes[i];if(!std::isfinite(b.x1)||!std::isfinite(b.y1)||!std::isfinite(b.x2)||!std::isfinite(b.y2)||b.x1<0||b.y1<0||b.x2>width||b.y2>height||b.x2<=b.x1||b.y2<=b.y1) throw std::invalid_argument("target-aware crop boxes must be non-empty and inside the image");}
}
BBoxEnvelope target_envelope(const BoxXYXY* boxes,std::size_t count,float erosion){
  float x1=boxes[0].x1,y1=boxes[0].y1,x2=boxes[0].x2,y2=boxes[0].y2;
  for(std::size_t i=1;i<count;++i){x1=std::min(x1,boxes[i].x1);y1=std::min(y1,boxes[i].y1);x2=std::max(x2,boxes[i].x2);y2=std::max(y2,boxes[i].y2);}
  const float ex=(x2-x1)*erosion*0.5f,ey=(y2-y1)*erosion*0.5f;
  int ix1=static_cast<int>(std::floor(x1+ex)),iy1=static_cast<int>(std::floor(y1+ey)); int ix2=static_cast<int>(std::ceil(x2-ex)),iy2=static_cast<int>(std::ceil(y2-ey));
  if(ix2<=ix1) ix2=ix1+1; if(iy2<=iy1) iy2=iy1+1;
  return {ix1,iy1,ix2,iy2};
}
int random_between(int low,int high,std::uint64_t seed){if(low>high) throw std::invalid_argument("target-aware crop has no valid origin");return low+static_cast<int>(splitmix64(seed)%(static_cast<std::uint64_t>(high-low)+1ULL));}
BBoxSafeRandomCropRectangle choose_bbox_safe(const BBoxSafeRandomCropConfig& c){
  validate_target_config(c.input_width,c.input_height,c.channels,c.boxes,c.box_count,c.erosion_rate);
  const int max_w=c.max_crop_width==0?c.input_width:c.max_crop_width,max_h=c.max_crop_height==0?c.input_height:c.max_crop_height;
  if(c.min_crop_width<=0||c.min_crop_height<=0||max_w<c.min_crop_width||max_h<c.min_crop_height||max_w>c.input_width||max_h>c.input_height) throw std::invalid_argument("invalid BBoxSafeRandomCrop dimensions");
  if(c.box_count==0){const int w=random_between(c.min_crop_width,max_w,c.seed),h=random_between(c.min_crop_height,max_h,c.seed+1);return {random_between(0,c.input_width-w,c.seed+2),random_between(0,c.input_height-h,c.seed+3),w,h,true};}
  const BBoxEnvelope e=target_envelope(c.boxes,c.box_count,c.erosion_rate); const int required_w=e.x2-e.x1,required_h=e.y2-e.y1; const int low_w=std::max(c.min_crop_width,required_w),low_h=std::max(c.min_crop_height,required_h); if(low_w>max_w||low_h>max_h) throw std::invalid_argument("BBoxSafeRandomCrop max crop cannot contain boxes");
  const int w=random_between(low_w,max_w,c.seed),h=random_between(low_h,max_h,c.seed+1);
  return {random_between(std::max(0,e.x2-w),std::min(e.x1,c.input_width-w),c.seed+2),random_between(std::max(0,e.y2-h),std::min(e.y1,c.input_height-h),c.seed+3),w,h,false};
}
BBoxSafeRandomCropRectangle make_bbox_safe_random_crop_rectangle(const BBoxSafeRandomCropConfig& c){return choose_bbox_safe(c);}
void bbox_safe_random_crop_u8(const std::uint8_t* in,std::uint8_t* out,const BBoxSafeRandomCropConfig& c,cudaStream_t s){if(!in||!out) throw std::invalid_argument("null BBoxSafeRandomCrop buffer");const auto r=choose_bbox_safe(c);random_crop_u8(in,out,{c.input_width,c.input_height,r.width,r.height,c.channels,r.x,r.y},s);}
RandomSizedBBoxSafeCropRectangle make_random_sized_bbox_safe_crop_rectangle(const RandomSizedBBoxSafeCropConfig& c){
  if(c.output_width<=0||c.output_height<=0||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>2) throw std::invalid_argument("invalid RandomSizedBBoxSafeCrop output configuration");
  BBoxSafeRandomCropConfig source{c.input_width,c.input_height,c.channels,c.boxes,c.box_count,1,1,0,0,c.erosion_rate,c.seed}; const auto r=choose_bbox_safe(source);
  return {r.x,r.y,r.width,r.height,c.output_width,c.output_height,static_cast<float>(c.output_width)/r.width,static_cast<float>(c.output_height)/r.height,r.used_fallback};
}
void random_sized_bbox_safe_crop_u8(const std::uint8_t* in,std::uint8_t* out,const RandomSizedBBoxSafeCropConfig& c,cudaStream_t s){
  if(!in||!out) throw std::invalid_argument("null RandomSizedBBoxSafeCrop buffer"); const auto r=make_random_sized_bbox_safe_crop_rectangle(c); std::vector<std::uint8_t> cropped(static_cast<std::size_t>(r.width)*r.height*c.channels); random_crop_u8(in,cropped.data(),{c.input_width,c.input_height,r.width,r.height,c.channels,r.x,r.y},s); resize_u8(cropped.data(),out,{r.width,r.height,r.output_width,r.output_height,c.channels,c.interpolation},s);
}
AtLeastOneBBoxRandomCropRectangle make_at_least_one_bbox_random_crop_rectangle(const AtLeastOneBBoxRandomCropConfig& c){
  validate_target_config(c.input_width,c.input_height,c.channels,c.boxes,c.box_count,c.erosion_factor);
  if(c.crop_width<=0||c.crop_height<=0||c.crop_width>c.input_width||c.crop_height>c.input_height) throw std::invalid_argument("invalid AtLeastOneBBoxRandomCrop dimensions");
  if((c.fallback_x<0)!=(c.fallback_y<0)||c.fallback_x<-1||c.fallback_y<-1||c.fallback_x+c.crop_width>c.input_width||c.fallback_y+c.crop_height>c.input_height) throw std::invalid_argument("invalid AtLeastOneBBoxRandomCrop fallback");
  if(c.box_count==0){const int x=c.fallback_x>=0?c.fallback_x:random_between(0,c.input_width-c.crop_width,c.seed+1),y=c.fallback_y>=0?c.fallback_y:random_between(0,c.input_height-c.crop_height,c.seed+2);return {x,y,c.crop_width,c.crop_height,static_cast<std::size_t>(-1),true};}
  const std::size_t selected=static_cast<std::size_t>(splitmix64(c.seed)%c.box_count); const BoxXYXY b=c.boxes[selected]; const float ex=(b.x2-b.x1)*c.erosion_factor*0.5f,ey=(b.y2-b.y1)*c.erosion_factor*0.5f; const int x1=static_cast<int>(std::floor(b.x1+ex)),y1=static_cast<int>(std::floor(b.y1+ey)),x2=static_cast<int>(std::ceil(b.x2-ex)),y2=static_cast<int>(std::ceil(b.y2-ey));
  return {random_between(std::max(0,x1-c.crop_width+1),std::min(x2-1,c.input_width-c.crop_width),c.seed+1),random_between(std::max(0,y1-c.crop_height+1),std::min(y2-1,c.input_height-c.crop_height),c.seed+2),c.crop_width,c.crop_height,selected,false};
}
void at_least_one_bbox_random_crop_u8(const std::uint8_t* in,std::uint8_t* out,const AtLeastOneBBoxRandomCropConfig& c,cudaStream_t s){if(!in||!out) throw std::invalid_argument("null AtLeastOneBBoxRandomCrop buffer");const auto r=make_at_least_one_bbox_random_crop_rectangle(c);random_crop_u8(in,out,{c.input_width,c.input_height,r.width,r.height,c.channels,r.x,r.y},s);}
std::ptrdiff_t view_offset(const ImageView& v,int x,int y,int channel){
  if(v.layout==Layout::HWC)return static_cast<std::ptrdiff_t>(y)*v.stride_y+static_cast<std::ptrdiff_t>(x)*v.stride_x+static_cast<std::ptrdiff_t>(channel)*v.stride_c;
  return static_cast<std::ptrdiff_t>(channel)*v.stride_c+static_cast<std::ptrdiff_t>(y)*v.stride_y+static_cast<std::ptrdiff_t>(x)*v.stride_x;
}
std::ptrdiff_t mutable_view_offset(const MutableImageView& v,int x,int y,int channel){
  return view_offset(v.as_const(),x,y,channel);
}
void validate_crop_pair(const ImageMaskView& source,const MutableImageMaskView& destination,const CropNonEmptyMaskIfExistsConfig& c){
  source.image.require_valid(); destination.image.require_valid();
  if(source.image.type!=DataType::UInt8||destination.image.type!=DataType::UInt8||source.image.layout!=destination.image.layout||source.image.channels!=destination.image.channels)
    throw std::invalid_argument("CropNonEmptyMaskIfExists image views must be matching uint8 dimensions");
  if(c.input_width>0&&c.input_width!=source.image.width||c.input_height>0&&c.input_height!=source.image.height||c.channels>0&&c.channels!=source.image.channels)
    throw std::invalid_argument("CropNonEmptyMaskIfExists input dimensions disagree with image view");
  if(c.crop_width<=0||c.crop_height<=0||c.crop_width>source.image.width||c.crop_height>source.image.height)
    throw std::invalid_argument("CropNonEmptyMaskIfExists crop exceeds image");
  if((c.fallback_x< -1)||(c.fallback_y< -1)||(c.fallback_x>=0)!=(c.fallback_y>=0)) throw std::invalid_argument("CropNonEmptyMaskIfExists fallback coordinates must be both explicit or both -1");
  if(c.fallback_x>=0&&(c.fallback_x+c.crop_width>source.image.width||c.fallback_y+c.crop_height>source.image.height)) throw std::invalid_argument("CropNonEmptyMaskIfExists fallback crop is outside image");
  if(source.mask.valid()){
    if(source.mask.type!=DataType::UInt8||source.mask.width!=source.image.width||source.mask.height!=source.image.height||source.mask.channels<=0) throw std::invalid_argument("CropNonEmptyMaskIfExists mask view must match image dimensions");
    destination.mask.require_valid();
    if(destination.mask.type!=DataType::UInt8||destination.mask.layout!=source.mask.layout||destination.mask.channels!=source.mask.channels) throw std::invalid_argument("CropNonEmptyMaskIfExists output mask view mismatch");
  } else if(destination.mask.valid()) throw std::invalid_argument("CropNonEmptyMaskIfExists cannot output a mask without an input mask");
  if(destination.image.width!=c.crop_width||destination.image.height!=c.crop_height) throw std::invalid_argument("CropNonEmptyMaskIfExists image output dimensions mismatch crop");
  if(source.mask.valid()&&(destination.mask.width!=c.crop_width||destination.mask.height!=c.crop_height)) throw std::invalid_argument("CropNonEmptyMaskIfExists mask output dimensions mismatch crop");
}
CropNonEmptyMaskIfExistsRectangle choose_crop_rectangle(const ImageMaskView& source,const CropNonEmptyMaskIfExistsConfig& c){
  const int width=source.image.width,height=source.image.height;
  std::vector<std::pair<int,int>> targets;
  if(source.mask.valid()){
    const auto* bytes=static_cast<const std::uint8_t*>(source.mask.data);
    for(int y=0;y<height;++y) for(int x=0;x<width;++x){
      bool target=false; for(int ch=0;ch<source.mask.channels;++ch) if(bytes[view_offset(source.mask,x,y,ch)]>c.mask_threshold){target=true;break;}
      if(target) targets.emplace_back(x,y);
    }
  }
  const bool use_mask=!targets.empty();
  if(!use_mask){
    const int x=c.fallback_x>=0?c.fallback_x:static_cast<int>(splitmix64(c.seed)%static_cast<std::uint64_t>(width-c.crop_width+1));
    const int y=c.fallback_y>=0?c.fallback_y:static_cast<int>(splitmix64(c.seed+1)%static_cast<std::uint64_t>(height-c.crop_height+1));
    return {x,y,c.crop_width,c.crop_height,false};
  }
  const auto target=targets[static_cast<std::size_t>(splitmix64(c.seed)%targets.size())];
  const int min_x=std::max(0,target.first-c.crop_width+1),max_x=std::min(target.first,width-c.crop_width);
  const int min_y=std::max(0,target.second-c.crop_height+1),max_y=std::min(target.second,height-c.crop_height);
  const int x=min_x+static_cast<int>(splitmix64(c.seed+1)%static_cast<std::uint64_t>(max_x-min_x+1));
  const int y=min_y+static_cast<int>(splitmix64(c.seed+2)%static_cast<std::uint64_t>(max_y-min_y+1));
  return {x,y,c.crop_width,c.crop_height,true};
}
CropNonEmptyMaskIfExistsRectangle make_crop_non_empty_mask_if_exists_rectangle(const ImageMaskView& source,const CropNonEmptyMaskIfExistsConfig& c){
  if(!source.image.valid()||source.image.type!=DataType::UInt8) throw std::invalid_argument("CropNonEmptyMaskIfExists requires a uint8 image view");
  if(c.input_width>0&&c.input_width!=source.image.width||c.input_height>0&&c.input_height!=source.image.height) throw std::invalid_argument("CropNonEmptyMaskIfExists input dimensions disagree with image view");
  if(c.crop_width<=0||c.crop_height<=0||c.crop_width>source.image.width||c.crop_height>source.image.height) throw std::invalid_argument("CropNonEmptyMaskIfExists crop exceeds image");
  if(source.mask.valid()&&(source.mask.type!=DataType::UInt8||source.mask.width!=source.image.width||source.mask.height!=source.image.height)) throw std::invalid_argument("CropNonEmptyMaskIfExists mask view must match image dimensions");
  if((c.fallback_x< -1)||(c.fallback_y< -1)||(c.fallback_x>=0)!=(c.fallback_y>=0)||c.fallback_x+c.crop_width>source.image.width||c.fallback_y+c.crop_height>source.image.height) throw std::invalid_argument("invalid CropNonEmptyMaskIfExists fallback crop");
  return choose_crop_rectangle(source,c);
}
void crop_non_empty_mask_if_exists(const ImageMaskView& source,const MutableImageMaskView& destination,const CropNonEmptyMaskIfExistsConfig& c,cudaStream_t){
  validate_crop_pair(source,destination,c); const auto r=choose_crop_rectangle(source,c);
  const auto* in=static_cast<const std::uint8_t*>(source.image.data); auto* out=static_cast<std::uint8_t*>(destination.image.data);
  for(int y=0;y<c.crop_height;++y) for(int x=0;x<c.crop_width;++x) for(int ch=0;ch<source.image.channels;++ch) out[mutable_view_offset(destination.image,x,y,ch)]=in[view_offset(source.image,x+r.x,y+r.y,ch)];
  if(source.mask.valid()){
    const auto* mask=static_cast<const std::uint8_t*>(source.mask.data); auto* mask_out=static_cast<std::uint8_t*>(destination.mask.data);
    for(int y=0;y<c.crop_height;++y) for(int x=0;x<c.crop_width;++x) for(int ch=0;ch<source.mask.channels;++ch) mask_out[mutable_view_offset(destination.mask,x,y,ch)]=mask[view_offset(source.mask,x+r.x,y+r.y,ch)];
  }
}
void crop_non_empty_mask_if_exists_u8(const std::uint8_t* image,const std::uint8_t* mask,std::uint8_t* output_image,std::uint8_t* output_mask,const CropNonEmptyMaskIfExistsConfig& c,cudaStream_t s){
  if(!image||!output_image||(mask&&!output_mask)) throw std::invalid_argument("null CropNonEmptyMaskIfExists buffer");
  ImageMaskView source{make_hwc_u8_view(image,c.input_width,c.input_height,c.channels),mask?make_hwc_u8_view(mask,c.input_width,c.input_height,c.mask_channels):ImageView{}};
  MutableImageMaskView destination{make_hwc_u8_view(output_image,c.crop_width,c.crop_height,c.channels),mask?make_hwc_u8_view(output_mask,c.crop_width,c.crop_height,c.mask_channels):MutableImageView{}};
  crop_non_empty_mask_if_exists(source,destination,c,s);
}
void square_symmetric_pad_u8(const std::uint8_t* in,std::uint8_t* out,const PadConfig& c,cudaStream_t s){if(c.output_width!=c.output_height||c.output_width<std::max(c.input_width,c.input_height))throw std::invalid_argument("invalid square symmetric pad configuration");pad_to_size_u8(in,out,c,s);}
void pad_to_size_u8(const std::uint8_t* in,std::uint8_t* out,const PadConfig& c,cudaStream_t){check(c.input_width,c.input_height,c.channels);if(!in||!out||c.output_width<c.input_width||c.output_height<c.input_height||static_cast<int>(c.border)<0||static_cast<int>(c.border)>2)throw std::invalid_argument("invalid pad configuration");int ox=(c.output_width-c.input_width)/2,oy=(c.output_height-c.input_height)/2;for(int y=0;y<c.output_height;++y)for(int x=0;x<c.output_width;++x)for(int ch=0;ch<c.channels;++ch){int sx=x-ox,sy=y-oy;bool inside=sx>=0&&sx<c.input_width&&sy>=0&&sy<c.input_height;if(!inside&&c.border==PadBorder::Constant){out[(y*c.output_width+x)*c.channels+ch]=c.value;continue;}if(c.border==PadBorder::Replicate){sx=std::max(0,std::min(c.input_width-1,sx));sy=std::max(0,std::min(c.input_height-1,sy));}else if(c.border==PadBorder::Reflect101){sx=refl(sx,c.input_width);sy=refl(sy,c.input_height);}if(!inside&&c.border==PadBorder::Constant)continue;out[(y*c.output_width+x)*c.channels+ch]=in[(sy*c.input_width+sx)*c.channels+ch];}}
void crop_and_pad_u8(const std::uint8_t* in,std::uint8_t* out,const CropPadConfig& c,cudaStream_t){check(c.input_width,c.input_height,c.channels);int ow=c.input_width+c.left+c.right,oh=c.input_height+c.top+c.bottom;if(!in||!out||ow<=0||oh<=0)throw std::invalid_argument("invalid crop and pad configuration");int pl=std::max(0,c.left),pt=std::max(0,c.top),cl=std::max(0,-c.left),ct=std::max(0,-c.top);for(int y=0;y<oh;++y)for(int x=0;x<ow;++x)for(int ch=0;ch<c.channels;++ch){int sx=x-pl+cl,sy=y-pt+ct;bool inside=sx>=0&&sx<c.input_width&&sy>=0&&sy<c.input_height;out[(y*ow+x)*c.channels+ch]=inside?in[(sy*c.input_width+sx)*c.channels+ch]:c.value;}}
void pad_to_fixed_size_u8(const std::uint8_t* i,std::uint8_t* o,const PadConfig& c,cudaStream_t s){pad_to_size_u8(i,o,c,s);} void center_pad_to_fixed_size_u8(const std::uint8_t* i,std::uint8_t* o,const PadConfig& c,cudaStream_t s){pad_to_size_u8(i,o,c,s);} void pad_to_square_u8(const std::uint8_t* i,std::uint8_t* o,const PadConfig& c,cudaStream_t s){square_symmetric_pad_u8(i,o,c,s);} void center_pad_to_square_u8(const std::uint8_t* i,std::uint8_t* o,const PadConfig& c,cudaStream_t s){square_symmetric_pad_u8(i,o,c,s);} void crop_to_fixed_size_u8(const std::uint8_t* i,std::uint8_t* o,const RandomCropConfig& c,cudaStream_t s){random_crop_u8(i,o,c,s);} void center_crop_to_fixed_size_u8(const std::uint8_t* i,std::uint8_t* o,const CenterCropConfig& c,cudaStream_t s){center_crop_u8(i,o,c,s);} void crop_to_square_u8(const std::uint8_t* i,std::uint8_t* o,const CenterCropConfig& c,cudaStream_t s){if(c.output_width!=c.output_height)throw std::invalid_argument("square crop requires equal output dimensions");center_crop_u8(i,o,c,s);} void center_crop_to_square_u8(const std::uint8_t* i,std::uint8_t* o,const CenterCropConfig& c,cudaStream_t s){crop_to_square_u8(i,o,c,s);}
namespace {
int rounded_dimension(double value,SizeRounding rounding){if(!std::isfinite(value)||value<1.0||value>static_cast<double>(std::numeric_limits<int>::max()))throw std::invalid_argument("size target is outside integer range");if(rounding==SizeRounding::Floor)return static_cast<int>(std::floor(value));if(rounding==SizeRounding::Ceil)return static_cast<int>(std::ceil(value));if(rounding==SizeRounding::Nearest)return static_cast<int>(std::floor(value+0.5));throw std::invalid_argument("invalid size rounding");}
void validate_anchor(SizeAnchor a){if(a!=SizeAnchor::TopLeft&&a!=SizeAnchor::Center)throw std::invalid_argument("invalid size anchor");}
void validate_rounding(SizeRounding r){if(r!=SizeRounding::Floor&&r!=SizeRounding::Ceil&&r!=SizeRounding::Nearest)throw std::invalid_argument("invalid size rounding");}
int multiple_dimension(int n,int m,bool pad){if(n<=0||m<=0)throw std::invalid_argument("size dimensions and multiples must be positive");long long q=pad?(static_cast<long long>(n)+m-1)/m:n/m;if(q<1||q*m>std::numeric_limits<int>::max())throw std::invalid_argument("multiple target overflows");return static_cast<int>(q*m);}
int power_dimension(int n,int base,bool pad){if(n<=0||base<2)throw std::invalid_argument("size dimensions and power base are invalid");long long p=1;while(p<n){if(p>std::numeric_limits<int>::max()/base)throw std::invalid_argument("power target overflows");p*=base;}if(!pad&&p>n)p/=base;return static_cast<int>(p);}
SizeTransformDimensions aspect_dimensions(const AspectRatioConfig& c,bool pad){check(c.input_width,c.input_height,c.channels);if(!std::isfinite(c.aspect_ratio)||c.aspect_ratio<=0)throw std::invalid_argument("aspect ratio must be finite and positive");validate_anchor(c.anchor);validate_rounding(c.rounding);const double current=static_cast<double>(c.input_width)/c.input_height;int w=c.input_width,h=c.input_height;if((pad&&current<c.aspect_ratio)||(!pad&&current>c.aspect_ratio))w=rounded_dimension(static_cast<double>(c.input_height)*c.aspect_ratio,c.rounding);else if((pad&&current>c.aspect_ratio)||(!pad&&current<c.aspect_ratio))h=rounded_dimension(static_cast<double>(c.input_width)/c.aspect_ratio,c.rounding);if(w<1||h<1)throw std::invalid_argument("aspect ratio target is empty");if(pad){w=std::max(w,c.input_width);h=std::max(h,c.input_height);}else{w=std::min(w,c.input_width);h=std::min(h,c.input_height);}return {w,h};}
void validate_transform_buffers(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,int ow,int oh,SizeAnchor anchor,bool pad){check(w,h,c);if(!in||!out||ow<=0||oh<=0)throw std::invalid_argument("invalid size transform buffers");validate_anchor(anchor);if((pad&&(ow<w||oh<h))||(!pad&&(ow>w||oh>h)))throw std::invalid_argument("size transform has incompatible target");}
void execute_size_transform(const std::uint8_t* in,std::uint8_t* out,int w,int h,int c,int ow,int oh,std::uint8_t value,PadBorder border,SizeAnchor anchor,bool pad){validate_transform_buffers(in,out,w,h,c,ow,oh,anchor,pad);if(static_cast<int>(border)<0||static_cast<int>(border)>2)throw std::invalid_argument("invalid pad border");const int dx=anchor==SizeAnchor::Center?std::abs(ow-w)/2:0,dy=anchor==SizeAnchor::Center?std::abs(oh-h)/2:0;for(int y=0;y<oh;++y)for(int x=0;x<ow;++x)for(int ch=0;ch<c;++ch){int sx=pad?x-dx:x+dx,sy=pad?y-dy:y+dy;const bool inside=sx>=0&&sx<w&&sy>=0&&sy<h;const std::size_t oi=(static_cast<std::size_t>(y)*ow+x)*c+ch;if(!inside){if(!pad)throw std::logic_error("crop geometry escaped source");if(border==PadBorder::Constant){out[oi]=value;continue;}if(border==PadBorder::Replicate){sx=std::max(0,std::min(w-1,sx));sy=std::max(0,std::min(h-1,sy));}else{sx=refl(sx,w);sy=refl(sy,h);}}out[oi]=in[(static_cast<std::size_t>(sy)*w+sx)*c+ch];}}
}
SizeTransformDimensions make_pad_to_multiples_of_dimensions(const MultiplesOfConfig& c){check(c.input_width,c.input_height,c.channels);validate_anchor(c.anchor);return {multiple_dimension(c.input_width,c.multiple_width,true),multiple_dimension(c.input_height,c.multiple_height,true)};}
SizeTransformDimensions make_crop_to_multiples_of_dimensions(const MultiplesOfConfig& c){check(c.input_width,c.input_height,c.channels);validate_anchor(c.anchor);return {multiple_dimension(c.input_width,c.multiple_width,false),multiple_dimension(c.input_height,c.multiple_height,false)};}
SizeTransformDimensions make_pad_to_powers_of_dimensions(const PowersOfConfig& c){check(c.input_width,c.input_height,c.channels);validate_anchor(c.anchor);return {power_dimension(c.input_width,c.base,true),power_dimension(c.input_height,c.base,true)};}
SizeTransformDimensions make_crop_to_powers_of_dimensions(const PowersOfConfig& c){check(c.input_width,c.input_height,c.channels);validate_anchor(c.anchor);return {power_dimension(c.input_width,c.base,false),power_dimension(c.input_height,c.base,false)};}
SizeTransformDimensions make_pad_to_aspect_ratio_dimensions(const AspectRatioConfig& c){return aspect_dimensions(c,true);}
SizeTransformDimensions make_crop_to_aspect_ratio_dimensions(const AspectRatioConfig& c){return aspect_dimensions(c,false);}
void pad_to_multiples_of_u8(const std::uint8_t* i,std::uint8_t* o,const MultiplesOfConfig& c,cudaStream_t){auto d=make_pad_to_multiples_of_dimensions(c);execute_size_transform(i,o,c.input_width,c.input_height,c.channels,d.width,d.height,c.value,c.border,c.anchor,true);}
void crop_to_multiples_of_u8(const std::uint8_t* i,std::uint8_t* o,const MultiplesOfConfig& c,cudaStream_t){auto d=make_crop_to_multiples_of_dimensions(c);execute_size_transform(i,o,c.input_width,c.input_height,c.channels,d.width,d.height,0,PadBorder::Constant,c.anchor,false);}
void pad_to_powers_of_u8(const std::uint8_t* i,std::uint8_t* o,const PowersOfConfig& c,cudaStream_t){auto d=make_pad_to_powers_of_dimensions(c);execute_size_transform(i,o,c.input_width,c.input_height,c.channels,d.width,d.height,c.value,c.border,c.anchor,true);}
void crop_to_powers_of_u8(const std::uint8_t* i,std::uint8_t* o,const PowersOfConfig& c,cudaStream_t){auto d=make_crop_to_powers_of_dimensions(c);execute_size_transform(i,o,c.input_width,c.input_height,c.channels,d.width,d.height,0,PadBorder::Constant,c.anchor,false);}
void pad_to_aspect_ratio_u8(const std::uint8_t* i,std::uint8_t* o,const AspectRatioConfig& c,cudaStream_t){auto d=make_pad_to_aspect_ratio_dimensions(c);execute_size_transform(i,o,c.input_width,c.input_height,c.channels,d.width,d.height,c.value,c.border,c.anchor,true);}
void crop_to_aspect_ratio_u8(const std::uint8_t* i,std::uint8_t* o,const AspectRatioConfig& c,cudaStream_t){auto d=make_crop_to_aspect_ratio_dimensions(c);execute_size_transform(i,o,c.input_width,c.input_height,c.channels,d.width,d.height,0,PadBorder::Constant,c.anchor,false);}
void center_pad_to_multiples_of_u8(const std::uint8_t* i,std::uint8_t* o,MultiplesOfConfig c,cudaStream_t s){c.anchor=SizeAnchor::Center;pad_to_multiples_of_u8(i,o,c,s);} void center_crop_to_multiples_of_u8(const std::uint8_t* i,std::uint8_t* o,MultiplesOfConfig c,cudaStream_t s){c.anchor=SizeAnchor::Center;crop_to_multiples_of_u8(i,o,c,s);} void center_pad_to_powers_of_u8(const std::uint8_t* i,std::uint8_t* o,PowersOfConfig c,cudaStream_t s){c.anchor=SizeAnchor::Center;pad_to_powers_of_u8(i,o,c,s);} void center_crop_to_powers_of_u8(const std::uint8_t* i,std::uint8_t* o,PowersOfConfig c,cudaStream_t s){c.anchor=SizeAnchor::Center;crop_to_powers_of_u8(i,o,c,s);} void center_pad_to_aspect_ratio_u8(const std::uint8_t* i,std::uint8_t* o,AspectRatioConfig c,cudaStream_t s){c.anchor=SizeAnchor::Center;pad_to_aspect_ratio_u8(i,o,c,s);} void center_crop_to_aspect_ratio_u8(const std::uint8_t* i,std::uint8_t* o,AspectRatioConfig c,cudaStream_t s){c.anchor=SizeAnchor::Center;crop_to_aspect_ratio_u8(i,o,c,s);}
void keep_size_by_resize_u8(const std::uint8_t* i,std::uint8_t* o,const KeepSizeByResizeConfig& c,cudaStream_t s){check(c.input_width,c.input_height,c.channels);if(!i||!o||c.intermediate_width<=0||c.intermediate_height<=0||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>2)throw std::invalid_argument("invalid KeepSizeByResize configuration");if(c.intermediate_width==c.input_width&&c.intermediate_height==c.input_height){std::copy(i,i+static_cast<std::size_t>(c.input_width)*c.input_height*c.channels,o);return;}std::vector<std::uint8_t> intermediate(static_cast<std::size_t>(c.intermediate_width)*c.intermediate_height*c.channels);resize_u8(i,intermediate.data(),{c.input_width,c.input_height,c.intermediate_width,c.intermediate_height,c.channels,c.interpolation},s);resize_u8(intermediate.data(),o,{c.intermediate_width,c.intermediate_height,c.input_width,c.input_height,c.channels,c.interpolation},s);}
void keep_size_by_resize_u8(const std::uint8_t* i,std::uint8_t* o,const ResizeConfig& c,cudaStream_t s){keep_size_by_resize_u8(i,o,KeepSizeByResizeConfig{c.input_width,c.input_height,c.channels,c.output_width,c.output_height,c.interpolation},s);}
}
