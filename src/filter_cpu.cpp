#include "augmatch/filter.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace augmatch {
namespace {
int convolution_reflect101(int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;}
std::uint8_t convolution_round_u8(float value){const float clipped=std::max(0.0f,std::min(255.0f,value));return static_cast<std::uint8_t>(std::floor(clipped+0.5f));}
void apply_convolution3x3_u8(const std::uint8_t* in,std::uint8_t* out,int width,int height,int channels,const float* kernel,float alpha){
  float matrix[9];for(int i=0;i<9;++i)matrix[i]=alpha*kernel[i];matrix[4]+=(1.0f-alpha);
  for(int y=0;y<height;++y)for(int x=0;x<width;++x)for(int ch=0;ch<channels;++ch){
    float value=0.0f;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){const int px=convolution_reflect101(x+dx,width),py=convolution_reflect101(y+dy,height);value+=matrix[(dy+1)*3+dx+1]*static_cast<float>(in[(static_cast<std::size_t>(py)*width+px)*channels+ch]);}
    out[(static_cast<std::size_t>(y)*width+x)*channels+ch]=convolution_round_u8(value);
  }
}
void validate_convolution_inputs(const std::uint8_t* in,const std::uint8_t* out,int width,int height,int channels,float alpha){
  if(!in||!out||width<=0||height<=0||channels<=0||!std::isfinite(alpha)||alpha<0.0f||alpha>1.0f)throw std::invalid_argument("invalid convolution configuration");
}
}
void blur_u8(const std::uint8_t* in,std::uint8_t* out,const BlurConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.kernel_size<=0||!(c.kernel_size&1)||c.kernel_size>17||c.sigma<=0||static_cast<int>(c.mode)<0||static_cast<int>(c.mode)>2)throw std::invalid_argument("invalid blur configuration");
  const int r=c.kernel_size/2; const float inv=1.0f/(2.0f*c.sigma*c.sigma); const auto reflect101=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;}; std::vector<std::uint8_t> values(static_cast<std::size_t>(c.kernel_size)*c.kernel_size);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    if(c.mode==BlurMode::Median){int count=0;for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){int px=std::max(0,std::min(c.width-1,x+dx)),py=std::max(0,std::min(c.height-1,y+dy));values[count++]=in[(py*c.width+px)*c.channels+ch];}std::nth_element(values.begin(),values.begin()+count/2,values.begin()+count);out[(y*c.width+x)*c.channels+ch]=values[count/2];continue;}
    float sum=0,weight=0;for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){int px=reflect101(x+dx,c.width),py=reflect101(y+dy,c.height);float w=c.mode==BlurMode::Average?1.0f:std::exp(-(dx*dx+dy*dy)*inv);sum+=w*in[(py*c.width+px)*c.channels+ch];weight+=w;}out[(y*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(sum/weight));
  }
}
void bilateral_blur_u8(const std::uint8_t* in,std::uint8_t* out,const BilateralBlurConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.radius<1||c.radius>15||!std::isfinite(c.sigma_space)||c.sigma_space<=0.0f||!std::isfinite(c.sigma_color)||c.sigma_color<=0.0f)throw std::invalid_argument("invalid bilateral blur configuration");
  const int radius=c.radius; const float spatial_inv=1.0f/(2.0f*c.sigma_space*c.sigma_space), color_inv=1.0f/(2.0f*c.sigma_color*c.sigma_color);
  const auto reflect101=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;};
  std::vector<float> sum(static_cast<std::size_t>(c.channels));
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){
    const std::size_t center=(static_cast<std::size_t>(y)*c.width+x)*c.channels; std::fill(sum.begin(),sum.end(),0.0f); float weight_sum=0.0f;
    for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){
      const int px=reflect101(x+dx,c.width),py=reflect101(y+dy,c.height); const std::size_t sample=(static_cast<std::size_t>(py)*c.width+px)*c.channels;
      float color_distance=0.0f; for(int ch=0;ch<c.channels;++ch){const float difference=static_cast<float>(in[sample+ch])-static_cast<float>(in[center+ch]); color_distance+=difference*difference;}
      const float weight=std::exp(-(static_cast<float>(dx*dx+dy*dy)*spatial_inv+color_distance*color_inv)); weight_sum+=weight;
      for(int ch=0;ch<c.channels;++ch)sum[static_cast<std::size_t>(ch)]+=weight*static_cast<float>(in[sample+ch]);
    }
    for(int ch=0;ch<c.channels;++ch){const float value=std::max(0.0f,std::min(255.0f,sum[static_cast<std::size_t>(ch)]/weight_sum)); out[center+ch]=static_cast<std::uint8_t>(std::floor(value+0.5f));}
  }
}
void mean_shift_blur_u8(const std::uint8_t* in,std::uint8_t* out,const MeanShiftBlurConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.spatial_radius<1||c.spatial_radius>15||c.iterations<1||c.iterations>16||!std::isfinite(c.color_radius)||c.color_radius<=0.0f)throw std::invalid_argument("invalid mean-shift blur configuration");
  const int radius=c.spatial_radius; const float color_limit=c.color_radius*c.color_radius;
  const auto reflect101=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;};
  std::vector<float> current(static_cast<std::size_t>(c.channels)), sums(static_cast<std::size_t>(c.channels));
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){
    const std::size_t center=(static_cast<std::size_t>(y)*c.width+x)*c.channels;
    for(int ch=0;ch<c.channels;++ch)current[static_cast<std::size_t>(ch)]=static_cast<float>(in[center+ch]);
    for(int iteration=0;iteration<c.iterations;++iteration){
      std::fill(sums.begin(),sums.end(),0.0f); int count=0;
      for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){
        const int px=reflect101(x+dx,c.width),py=reflect101(y+dy,c.height); const std::size_t sample=(static_cast<std::size_t>(py)*c.width+px)*c.channels;
        float distance=0.0f; for(int ch=0;ch<c.channels;++ch){const float difference=static_cast<float>(in[sample+ch])-current[static_cast<std::size_t>(ch)];distance+=difference*difference;}
        if(distance<=color_limit){++count;for(int ch=0;ch<c.channels;++ch)sums[static_cast<std::size_t>(ch)]+=static_cast<float>(in[sample+ch]);}
      }
      if(count==0)break;
      for(int ch=0;ch<c.channels;++ch)current[static_cast<std::size_t>(ch)]=sums[static_cast<std::size_t>(ch)]/static_cast<float>(count);
    }
    for(int ch=0;ch<c.channels;++ch){const float value=std::max(0.0f,std::min(255.0f,current[static_cast<std::size_t>(ch)]));out[center+ch]=static_cast<std::uint8_t>(std::floor(value+0.5f));}
  }
}
void advanced_blur_u8(const std::uint8_t* in,std::uint8_t* out,const AdvancedBlurConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.kernel_size<=0||!(c.kernel_size&1)||c.kernel_size>15||!std::isfinite(c.sigma_x)||!std::isfinite(c.sigma_y)||c.sigma_x<=0||c.sigma_y<=0||!std::isfinite(c.angle_degrees))throw std::invalid_argument("invalid advanced blur configuration");
  const int r=c.kernel_size/2; const float angle=c.angle_degrees*3.14159265358979323846f/180.0f,co=std::cos(angle),si=std::sin(angle);
  std::vector<float> kernel(static_cast<std::size_t>(c.kernel_size)*c.kernel_size); float norm=0.0f;
  for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){const float xp=co*dx+si*dy,yp=-si*dx+co*dy,qx=xp/c.sigma_x,qy=yp/c.sigma_y;const float w=std::exp(-0.5f*(qx*qx+qy*qy));kernel[static_cast<std::size_t>(dy+r)*c.kernel_size+dx+r]=w;norm+=w;}
  const auto reflect101=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;};
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0.0f;for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){const int px=reflect101(x+dx,c.width),py=reflect101(y+dy,c.height);const float w=kernel[static_cast<std::size_t>(dy+r)*c.kernel_size+dx+r]/norm;sum+=w*in[(py*c.width+px)*c.channels+ch];}out[(y*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(std::max(0.0f,std::min(255.0f,sum))));}
}
void unsharp_mask_u8(const std::uint8_t* in,std::uint8_t* out,const UnsharpConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.kernel_size<=0||!(c.kernel_size&1)||c.sigma<=0||c.amount<0||c.threshold<0)throw std::invalid_argument("invalid unsharp configuration");int r=c.kernel_size/2;float inv=1.f/(2.f*c.sigma*c.sigma);auto sample=[&](int x,int y,int ch){auto reflect101=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;};x=reflect101(x,c.width);y=reflect101(y,c.height);return static_cast<float>(in[(y*c.width+x)*c.channels+ch]);};for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0,weight=0;for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){float w=std::exp(-(dx*dx+dy*dy)*inv);sum+=w*sample(x+dx,y+dy,ch);weight+=w;}float orig=sample(x,y,ch),detail=orig-sum/weight;float v=std::abs(detail)>=c.threshold?orig+c.amount*detail:orig;out[(y*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(std::max(0.f,std::min(255.f,v))));}}
void sharpen_u8(const std::uint8_t* in,std::uint8_t* out,const SharpenConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.kernel_size<=0||!(c.kernel_size&1)||c.sigma<=0||c.alpha<0||c.lightness<0)throw std::invalid_argument("invalid sharpen configuration");int r=c.kernel_size/2;float inv=1.f/(2.f*c.sigma*c.sigma);auto ref=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;};for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0,weight=0;for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){float w=std::exp(-(dx*dx+dy*dy)*inv);sum+=w*in[(ref(y+dy,c.height)*c.width+ref(x+dx,c.width))*c.channels+ch];weight+=w;}int orig=in[(y*c.width+x)*c.channels+ch],blur=static_cast<int>(std::lround(sum/weight));float wrapped=static_cast<unsigned char>(orig-blur);float v=orig+c.alpha*c.lightness*wrapped;out[(y*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(std::max(0.f,std::min(255.f,v))));}}
void ringing_overshoot_u8(const std::uint8_t* in,std::uint8_t* out,const RingingOvershootConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.kernel_size<=0||!(c.kernel_size&1)||c.kernel_size>15||!std::isfinite(c.sigma)||c.sigma<=0.0f||!std::isfinite(c.amount)||c.amount<0.0f||c.amount>4.0f)throw std::invalid_argument("invalid ringing overshoot configuration");
  const int r=c.kernel_size/2; const float inv=1.0f/(2.0f*c.sigma*c.sigma); const auto reflect101=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;};
  std::vector<float> kernel(static_cast<std::size_t>(c.kernel_size)*c.kernel_size); float norm=0.0f;
  for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){const float weight=std::exp(-(dx*dx+dy*dy)*inv);kernel[static_cast<std::size_t>(dy+r)*c.kernel_size+dx+r]=weight;norm+=weight;}
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float blurred=0.0f;for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){const int px=reflect101(x+dx,c.width),py=reflect101(y+dy,c.height);blurred+=(kernel[static_cast<std::size_t>(dy+r)*c.kernel_size+dx+r]/norm)*in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch];}const float original=static_cast<float>(in[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]);const float value=original+c.amount*(original-blurred);out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(std::max(0.0f,std::min(255.0f,value))));}
}
void motion_blur_u8(const std::uint8_t* in,std::uint8_t* out,const MotionBlurConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.kernel_size<=0||!(c.kernel_size&1)||c.direction<-1||c.direction>1)throw std::invalid_argument("invalid motion blur configuration");int r=c.kernel_size/2;float a=c.angle_degrees*3.14159265358979323846f/180.f,co=std::cos(a),si=std::sin(a);auto ref=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;};for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0,weight=0;for(int t=-r;t<=r;++t){int px=static_cast<int>(std::lround(x+t*co)),py=static_cast<int>(std::lround(y+t*si));float w=1.f+c.direction*(r?static_cast<float>(t)/r:0.f);sum+=w*in[(ref(py,c.height)*c.width+ref(px,c.width))*c.channels+ch];weight+=w;}out[(y*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(sum/weight));}}
void optical_motion_blur_u8(const std::uint8_t* in,std::uint8_t* out,const OpticalMotionBlurConfig& c,cudaStream_t s){if(!std::isfinite(c.angle_degrees)||!std::isfinite(c.direction))throw std::invalid_argument("invalid optical motion blur configuration");motion_blur_u8(in,out,{c.width,c.height,c.channels,c.kernel_size,c.angle_degrees,c.direction,true},s);}
namespace {
int optical_reflect101(int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;}
std::uint8_t optical_round(float v){return static_cast<std::uint8_t>(std::lround(std::max(0.0f,std::min(255.0f,v))));}
void optical_validate(const std::uint8_t* in,const std::uint8_t* out,int w,int h,int c,int samples){if(!in||!out||w<=0||h<=0||c<=0||samples<1)throw std::invalid_argument("invalid optical blur configuration");}
}
void depth_dependent_defocus_u8(const std::uint8_t* in,std::uint8_t* out,const DepthDependentDefocusConfig& c,cudaStream_t){
  optical_validate(in,out,c.width,c.height,c.channels,1); if(!c.depth||!std::isfinite(c.focus_depth)||!std::isfinite(c.blur_scale)||c.blur_scale<0||c.max_radius<0||c.max_radius>64)throw std::invalid_argument("invalid depth-dependent defocus configuration");
  for(std::size_t i=0;i<static_cast<std::size_t>(c.width)*c.height;++i)if(!std::isfinite(c.depth[i]))throw std::invalid_argument("depth field contains a non-finite value");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){int r=std::max(0,std::min(c.max_radius,static_cast<int>(std::lround(std::fabs(c.depth[y*c.width+x]-c.focus_depth)*c.blur_scale))));for(int ch=0;ch<c.channels;++ch){float sum=0;int count=0;for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx)if(dx*dx+dy*dy<=r*r){sum+=in[(optical_reflect101(y+dy,c.height)*c.width+optical_reflect101(x+dx,c.width))*c.channels+ch];++count;}out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=optical_round(count?sum/count:in[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]);}}
}
void camera_shake_blur_u8(const std::uint8_t* in,std::uint8_t* out,const CameraShakeBlurConfig& c,cudaStream_t){
  optical_validate(in,out,c.width,c.height,c.channels,c.sample_count);if(!c.offsets_x||!c.offsets_y)throw std::invalid_argument("camera-shake offsets are required");
  for(int s=0;s<c.sample_count;++s)if(!std::isfinite(c.offsets_x[s])||!std::isfinite(c.offsets_y[s]))throw std::invalid_argument("camera-shake offset is not finite");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0;for(int s=0;s<c.sample_count;++s){int px=static_cast<int>(std::lround(x+c.offsets_x[s])),py=static_cast<int>(std::lround(y+c.offsets_y[s]));sum+=in[(optical_reflect101(py,c.height)*c.width+optical_reflect101(px,c.width))*c.channels+ch];}out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=optical_round(sum/c.sample_count);}
}
void linear_directional_blur_u8(const std::uint8_t* in,std::uint8_t* out,const LinearDirectionalBlurConfig& c,cudaStream_t){
  optical_validate(in,out,c.width,c.height,c.channels,c.sample_count);if(!std::isfinite(c.length)||c.length<0||!std::isfinite(c.angle_degrees))throw std::invalid_argument("invalid linear directional blur configuration");const float a=c.angle_degrees*3.14159265358979323846f/180.0f,co=std::cos(a),si=std::sin(a);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0;for(int s=0;s<c.sample_count;++s){float t=c.sample_count==1?0.0f:static_cast<float>(s)/(c.sample_count-1)-.5f;int px=static_cast<int>(std::lround(x+t*c.length*co)),py=static_cast<int>(std::lround(y+t*c.length*si));sum+=in[(optical_reflect101(py,c.height)*c.width+optical_reflect101(px,c.width))*c.channels+ch];}out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=optical_round(sum/c.sample_count);}
}
void rotational_motion_blur_u8(const std::uint8_t* in,std::uint8_t* out,const RotationalMotionBlurConfig& c,cudaStream_t){
  optical_validate(in,out,c.width,c.height,c.channels,c.sample_count);if(!std::isfinite(c.center_x)||!std::isfinite(c.center_y)||!std::isfinite(c.angle_degrees))throw std::invalid_argument("invalid rotational motion blur configuration");const float a=c.angle_degrees*3.14159265358979323846f/180.0f;
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0;for(int s=0;s<c.sample_count;++s){float q=c.sample_count==1?0.0f:static_cast<float>(s)/(c.sample_count-1)-.5f,co=std::cos(q*a),si=std::sin(q*a),rx=x-c.center_x,ry=y-c.center_y;int px=static_cast<int>(std::lround(c.center_x+co*rx-si*ry)),py=static_cast<int>(std::lround(c.center_y+si*rx+co*ry));sum+=in[(optical_reflect101(py,c.height)*c.width+optical_reflect101(px,c.width))*c.channels+ch];}out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=optical_round(sum/c.sample_count);}
}
void rolling_shutter_motion_blur_u8(const std::uint8_t* in,std::uint8_t* out,const RollingShutterMotionBlurConfig& c,cudaStream_t){
  optical_validate(in,out,c.width,c.height,c.channels,c.sample_count);if(!c.row_offsets_x||!c.row_offsets_y)throw std::invalid_argument("rolling-shutter row offsets are required");
  for(int y=0;y<c.height;++y)if(!std::isfinite(c.row_offsets_x[y])||!std::isfinite(c.row_offsets_y[y]))throw std::invalid_argument("rolling-shutter offset is not finite");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0;for(int s=0;s<c.sample_count;++s){float q=c.sample_count==1?0.0f:static_cast<float>(s)/(c.sample_count-1)-.5f;int px=static_cast<int>(std::lround(x+q*c.row_offsets_x[y])),py=static_cast<int>(std::lround(y+q*c.row_offsets_y[y]));sum+=in[(optical_reflect101(py,c.height)*c.width+optical_reflect101(px,c.width))*c.channels+ch];}out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=optical_round(sum/c.sample_count);}
}
void defocus_u8(const std::uint8_t* in,std::uint8_t* out,const DefocusConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.radius<1||c.radius>32||c.alias_blur<0)throw std::invalid_argument("invalid defocus configuration");auto ref=[](int i,int n){if(n==1)return 0;while(i<0||i>=n)i=i<0?-i:2*n-i-2;return i;};int r=c.radius,R=std::max(8,r),g=R<=8?1:2,sigma=c.alias_blur>0?c.alias_blur:0;auto gw=[&](int d){return sigma==0?(d==0?0.5f:0.25f):std::exp(-d*d/(2*sigma*sigma));};for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0,norm=0;for(int dy=-R;dy<=R;++dy)for(int dx=-R;dx<=R;++dx){float w=0;for(int gy=-g;gy<=g;++gy)for(int gx=-g;gx<=g;++gx)if((dx-gx)*(dx-gx)+(dy-gy)*(dy-gy)<=r*r)w+=gw(gx)*gw(gy);sum+=w*in[(ref(y+dy,c.height)*c.width+ref(x+dx,c.width))*c.channels+ch];norm+=w;}out[(y*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(sum/norm));}}
void optical_defocus_u8(const std::uint8_t* in,std::uint8_t* out,const OpticalDefocusConfig& c,cudaStream_t s){if(!std::isfinite(c.alias_blur)||c.alias_blur<0.0f)throw std::invalid_argument("invalid optical defocus configuration");if(c.radius==0){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid optical defocus configuration");std::copy(in,in+static_cast<std::size_t>(c.width)*c.height*c.channels,out);return;}defocus_u8(in,out,{c.width,c.height,c.channels,c.radius,c.alias_blur},s);}
namespace {
int zoom_blur_reflect101(int value,int size){if(size==1)return 0;while(value<0||value>=size)value=value<0?-value:2*size-value-2;return value;}
float zoom_blur_sample(const std::uint8_t* image,int width,int height,int channels,float x,float y,int channel){
  const int x0=static_cast<int>(std::floor(x)),y0=static_cast<int>(std::floor(y));
  const float fx=x-static_cast<float>(x0),fy=y-static_cast<float>(y0);
  const int xa=zoom_blur_reflect101(x0,width),xb=zoom_blur_reflect101(x0+1,width);
  const int ya=zoom_blur_reflect101(y0,height),yb=zoom_blur_reflect101(y0+1,height);
  const float a=image[(static_cast<std::size_t>(ya)*width+xa)*channels+channel];
  const float b=image[(static_cast<std::size_t>(ya)*width+xb)*channels+channel];
  const float c=image[(static_cast<std::size_t>(yb)*width+xa)*channels+channel];
  const float d=image[(static_cast<std::size_t>(yb)*width+xb)*channels+channel];
  return (a+(b-a)*fx)+(c+(d-c)*fx)*fy-(a+(b-a)*fx)*fy;
}
void validate_zoom_blur(const std::uint8_t* in,const std::uint8_t* out,const ZoomBlurConfig& c){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!std::isfinite(c.min_factor)||!std::isfinite(c.max_factor)||c.min_factor<1.0f||c.max_factor<c.min_factor||c.steps<0||c.steps>256)throw std::invalid_argument("invalid zoom blur configuration");
}
}
int zoom_blur_factor_count(const ZoomBlurConfig& c){if(c.steps<0||c.steps>256)throw std::invalid_argument("invalid zoom blur step count");return c.steps+1;}
void zoom_blur_u8(const std::uint8_t* in,std::uint8_t* out,const ZoomBlurConfig& c,cudaStream_t){validate_zoom_blur(in,out,c);const float cx=0.5f*(c.width-1),cy=0.5f*(c.height-1);const int count=c.steps+1;for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0.0f;for(int i=0;i<count;++i){const float t=c.steps?static_cast<float>(i)/c.steps:0.0f;const float factor=c.min_factor+(c.max_factor-c.min_factor)*t;const float sx=cx+(static_cast<float>(x)-cx)/factor,sy=cy+(static_cast<float>(y)-cy)/factor;sum+=zoom_blur_sample(in,c.width,c.height,c.channels,sx,sy,ch);}const float value=sum/static_cast<float>(count);out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::floor(std::max(0.0f,std::min(255.0f,value+0.5f))));}}
void optical_zoom_blur_u8(const std::uint8_t* in,std::uint8_t* out,const OpticalZoomBlurConfig& c,cudaStream_t s){zoom_blur_u8(in,out,{c.width,c.height,c.channels,c.min_factor,c.max_factor,c.steps},s);}
namespace {
std::uint64_t glass_blur_splitmix64(std::uint64_t value){value+=0x9e3779b97f4a7c15ULL;value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;value=(value^(value>>27))*0x94d049bb133111ebULL;return value^(value>>31);}
int glass_blur_radius(float sigma){const int radius=static_cast<int>(std::ceil(3.0f*sigma));if(radius<1||radius>15)throw std::invalid_argument("glass blur sigma produces an unsupported kernel");return radius;}
int glass_blur_coordinate(int value,int size){if(size==1)return 0;while(value<0||value>=size)value=value<0?-value:2*size-value-2;return value;}
void glass_blur_gaussian(const std::uint8_t* in,std::uint8_t* out,const GlassBlurConfig& c){const int radius=glass_blur_radius(c.sigma),side=2*radius+1;const float inv=1.0f/(2.0f*c.sigma*c.sigma);std::vector<float> weights(static_cast<std::size_t>(side)*side);float norm=0.0f;for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){const float weight=std::exp(-(dx*dx+dy*dy)*inv);weights[static_cast<std::size_t>(dy+radius)*side+dx+radius]=weight;norm+=weight;}for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float sum=0.0f;for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){const int px=glass_blur_coordinate(x+dx,c.width),py=glass_blur_coordinate(y+dy,c.height);sum+=weights[static_cast<std::size_t>(dy+radius)*side+dx+radius]*in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch];}out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=static_cast<std::uint8_t>(std::lround(sum/norm));}}
void validate_glass_blur(const std::uint8_t* in,const std::uint8_t* out,const GlassBlurConfig& c){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!std::isfinite(c.sigma)||c.sigma<=0.0f||c.max_delta<0||c.max_delta>127||c.iterations<0||c.max_delta>c.width/2||c.max_delta>c.height/2)throw std::invalid_argument("invalid glass blur configuration");glass_blur_radius(c.sigma);if(c.swap_sequence){const std::size_t count=glass_blur_swap_count(c.width,c.height,c.max_delta,c.iterations);for(std::size_t i=0;i<count;++i)if(c.swap_sequence[i].dx < -c.max_delta||c.swap_sequence[i].dx > c.max_delta||c.swap_sequence[i].dy < -c.max_delta||c.swap_sequence[i].dy > c.max_delta)throw std::invalid_argument("glass blur swap offset is outside max_delta");}}
}
std::size_t glass_blur_swap_count(int width,int height,int max_delta,int iterations){if(width<=0||height<=0||max_delta<0||max_delta>127||iterations<0||max_delta>width/2||max_delta>height/2)throw std::invalid_argument("invalid glass blur swap dimensions");return static_cast<std::size_t>(iterations)*static_cast<std::size_t>(height-2*max_delta)*static_cast<std::size_t>(width-2*max_delta);}
void make_glass_blur_swap_sequence(GlassBlurSwap* sequence,int width,int height,int max_delta,int iterations,std::uint64_t seed){const std::size_t count=glass_blur_swap_count(width,height,max_delta,iterations);if(!sequence&&count)throw std::invalid_argument("null glass blur swap sequence");const std::uint64_t span=static_cast<std::uint64_t>(2*max_delta+1);for(std::size_t i=0;i<count;++i){sequence[i].dx=static_cast<std::int8_t>(static_cast<int>(glass_blur_splitmix64(seed+static_cast<std::uint64_t>(2*i))%span)-max_delta);sequence[i].dy=static_cast<std::int8_t>(static_cast<int>(glass_blur_splitmix64(seed+static_cast<std::uint64_t>(2*i+1))%span)-max_delta);}}
void glass_blur_u8(const std::uint8_t* in,std::uint8_t* out,const GlassBlurConfig& c,cudaStream_t){validate_glass_blur(in,out,c);const std::size_t count=glass_blur_swap_count(c.width,c.height,c.max_delta,c.iterations);std::vector<GlassBlurSwap> generated;const GlassBlurSwap* sequence=c.swap_sequence;if(!sequence&&count){generated.resize(count);make_glass_blur_swap_sequence(generated.data(),c.width,c.height,c.max_delta,c.iterations,c.seed);sequence=generated.data();}std::vector<std::uint8_t> work(static_cast<std::size_t>(c.width)*c.height*c.channels);glass_blur_gaussian(in,work.data(),c);std::size_t index=0;for(int iteration=0;iteration<c.iterations;++iteration)for(int y=c.max_delta;y<c.height-c.max_delta;++y)for(int x=c.max_delta;x<c.width-c.max_delta;++x){const GlassBlurSwap swap=sequence[index++];const int sx=x+swap.dx,sy=y+swap.dy;for(int ch=0;ch<c.channels;++ch)std::swap(work[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch],work[(static_cast<std::size_t>(sy)*c.width+sx)*c.channels+ch]);}glass_blur_gaussian(work.data(),out,c);}
void emboss_u8(const std::uint8_t* in,std::uint8_t* out,const EmbossConfig& c,cudaStream_t){
  validate_convolution_inputs(in,out,c.width,c.height,c.channels,c.alpha);if(!std::isfinite(c.strength)||c.strength<0.0f)throw std::invalid_argument("invalid emboss strength");
  const float s=c.strength;const float kernel[9]={-1.0f-s,-s,0.0f,-s,1.0f,s,0.0f,s,1.0f+s};apply_convolution3x3_u8(in,out,c.width,c.height,c.channels,kernel,c.alpha);
}
void edge_detect_u8(const std::uint8_t* in,std::uint8_t* out,const EdgeDetectConfig& c,cudaStream_t){
  validate_convolution_inputs(in,out,c.width,c.height,c.channels,c.alpha);const float kernel[9]={0.0f,1.0f,0.0f,1.0f,-4.0f,1.0f,0.0f,1.0f,0.0f};apply_convolution3x3_u8(in,out,c.width,c.height,c.channels,kernel,c.alpha);
}
void directed_edge_detect_u8(const std::uint8_t* in,std::uint8_t* out,const DirectedEdgeDetectConfig& c,cudaStream_t){
  validate_convolution_inputs(in,out,c.width,c.height,c.channels,c.alpha);if(!std::isfinite(c.direction))throw std::invalid_argument("invalid directed edge direction");
  const int degrees=static_cast<int>(c.direction*360.0f)%360;const double radians=static_cast<double>(degrees)*3.14159265358979323846/180.0;const double direction_x=std::cos(radians-0.5*3.14159265358979323846),direction_y=std::sin(radians-0.5*3.14159265358979323846);float kernel[9]={0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f};float norm=0.0f;
  for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x)if(x||y){const double dot=static_cast<double>(x)*direction_x+static_cast<double>(y)*direction_y;const double distance=std::acos(std::max(-1.0,std::min(1.0,dot/std::sqrt(static_cast<double>(x*x+y*y)))))*180.0/3.14159265358979323846;const float similarity=static_cast<float>(std::pow(1.0-distance/180.0,4.0));kernel[(y+1)*3+x+1]=-similarity;norm+=similarity;}
  for(int i=0;i<9;++i)if(i!=4)kernel[i]/=norm;apply_convolution3x3_u8(in,out,c.width,c.height,c.channels,kernel,c.alpha);
}
namespace {
float diffraction_psf_weight(float distance,int radius,float scale){
  if(radius==0)return distance==0.0f?1.0f:0.0f;
  if(distance>static_cast<float>(radius))return 0.0f;
  const float q=distance/(static_cast<float>(radius)*scale);
  if(q==0.0f)return 1.0f;
  const float s=std::sin(3.14159265358979323846f*q)/(3.14159265358979323846f*q); return s*s;
}
float disk_psf_weight(float distance,int radius,float softness){
  if(radius==0)return distance==0.0f?1.0f:0.0f;
  if(distance>static_cast<float>(radius))return 0.0f;
  if(softness<=0.0f)return 1.0f;
  const float q=distance/static_cast<float>(radius),start=1.0f-softness;
  return q<=start?1.0f:std::max(0.0f,(1.0f-q)/softness);
}
float aperture_psf_weight(float dx,float dy,const ApertureShapeBlurConfig& c){
  const float distance=std::sqrt(dx*dx+dy*dy); if(distance>static_cast<float>(c.radius))return 0.0f;
  if(distance==0.0f)return 1.0f;
  const float pi=3.14159265358979323846f,sector=2.0f*pi/static_cast<float>(c.blades);
  float angle=std::atan2(dy,dx)-c.rotation_degrees*pi/180.0f;
  angle=std::fmod(angle+pi,sector); if(angle<0.0f)angle+=sector; angle-=sector*0.5f;
  const float polygon_radius=static_cast<float>(c.radius)*std::cos(pi/static_cast<float>(c.blades))/std::cos(angle);
  const float boundary=(1.0f-c.roundness)*polygon_radius+c.roundness*static_cast<float>(c.radius);
  return distance<=boundary?1.0f:0.0f;
}
void validate_psf_u8(const std::uint8_t* in,const std::uint8_t* out,int w,int h,int channels,int radius){
  if(!in||!out||w<=0||h<=0||channels<=0||radius<0||radius>32)throw std::invalid_argument("invalid optical PSF configuration");
}
template<class Weight>
void apply_psf_u8(const std::uint8_t* in,std::uint8_t* out,int w,int h,int channels,int radius,Weight weight){
  if(radius==0){std::copy(in,in+static_cast<std::size_t>(w)*h*channels,out);return;}
  for(int y=0;y<h;++y)for(int x=0;x<w;++x){
    std::vector<float> sums(static_cast<std::size_t>(channels),0.0f); float norm=0.0f;
    for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){const float wgt=weight(x,y,dx,dy);if(wgt<=0.0f)continue;const int px=optical_reflect101(x+dx,w),py=optical_reflect101(y+dy,h);norm+=wgt;const std::size_t sample=(static_cast<std::size_t>(py)*w+px)*channels;for(int ch=0;ch<channels;++ch)sums[static_cast<std::size_t>(ch)]+=wgt*in[sample+ch];}
    const std::size_t dst=(static_cast<std::size_t>(y)*w+x)*channels;for(int ch=0;ch<channels;++ch)out[dst+ch]=optical_round(norm>0.0f?sums[static_cast<std::size_t>(ch)]/norm:in[dst+ch]);
  }
}
}
void diffraction_blur_u8(const std::uint8_t* in,std::uint8_t* out,const DiffractionBlurConfig& c,cudaStream_t){
  validate_psf_u8(in,out,c.width,c.height,c.channels,c.radius);if(!std::isfinite(c.wavelength_nm)||c.wavelength_nm<=0.0f||!std::isfinite(c.aperture_diameter_mm)||c.aperture_diameter_mm<=0.0f||!std::isfinite(c.focal_length_mm)||c.focal_length_mm<=0.0f)throw std::invalid_argument("invalid diffraction parameters");
  const float scale=std::max(0.25f,std::min(4.0f,(c.wavelength_nm/550.0f)*(c.focal_length_mm/50.0f)*(2.0f/c.aperture_diameter_mm)));
  apply_psf_u8(in,out,c.width,c.height,c.channels,c.radius,[&](int,int,int dx,int dy){return diffraction_psf_weight(std::sqrt(static_cast<float>(dx*dx+dy*dy)),c.radius,scale);});
}
void bokeh_blur_u8(const std::uint8_t* in,std::uint8_t* out,const BokehBlurConfig& c,cudaStream_t){
  validate_psf_u8(in,out,c.width,c.height,c.channels,c.radius);if(!std::isfinite(c.edge_softness)||c.edge_softness<0.0f||c.edge_softness>1.0f)throw std::invalid_argument("invalid bokeh edge softness");
  apply_psf_u8(in,out,c.width,c.height,c.channels,c.radius,[&](int,int,int dx,int dy){return disk_psf_weight(std::sqrt(static_cast<float>(dx*dx+dy*dy)),c.radius,c.edge_softness);});
}
void cat_eye_bokeh_u8(const std::uint8_t* in,std::uint8_t* out,const CatEyeBokehConfig& c,cudaStream_t){
  validate_psf_u8(in,out,c.width,c.height,c.channels,c.radius);if(!std::isfinite(c.cat_eye_strength)||c.cat_eye_strength<0.0f||c.cat_eye_strength>1.0f||!std::isfinite(c.center_x)||!std::isfinite(c.center_y))throw std::invalid_argument("invalid cat-eye bokeh parameters");
  apply_psf_u8(in,out,c.width,c.height,c.channels,c.radius,[&](int x,int y,int dx,int dy){const float nx=c.width>1?(static_cast<float>(x)/(c.width-1)-c.center_x):0.0f,ny=c.height>1?(static_cast<float>(y)/(c.height-1)-c.center_y):0.0f,edge=std::min(1.0f,std::sqrt(nx*nx+ny*ny)*2.0f),scale=std::max(0.25f,1.0f-c.cat_eye_strength*edge);return (dx*dx/(static_cast<float>(c.radius*c.radius)*scale*scale)+dy*dy/static_cast<float>(c.radius*c.radius)<=1.0f)?1.0f:0.0f;});
}
void aperture_shape_blur_u8(const std::uint8_t* in,std::uint8_t* out,const ApertureShapeBlurConfig& c,cudaStream_t){
  validate_psf_u8(in,out,c.width,c.height,c.channels,c.radius);if(c.blades<3||c.blades>32||!std::isfinite(c.rotation_degrees)||!std::isfinite(c.roundness)||c.roundness<0.0f||c.roundness>1.0f)throw std::invalid_argument("invalid aperture shape parameters");
  apply_psf_u8(in,out,c.width,c.height,c.channels,c.radius,[&](int,int,int dx,int dy){return aperture_psf_weight(static_cast<float>(dx),static_cast<float>(dy),c);});
}
}
