#include "augmatch/geometry/geometric.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
namespace augmatch {
void resize_u8(const std::uint8_t* in,std::uint8_t* out,const ResizeConfig& c,cudaStream_t){
  if(!in||!out||c.input_width<=0||c.input_height<=0||c.output_width<=0||c.output_height<=0||c.channels<=0||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>2) throw std::invalid_argument("invalid resize configuration");
  const float sx=static_cast<float>(c.input_width)/c.output_width, sy=static_cast<float>(c.input_height)/c.output_height;
  for(int y=0;y<c.output_height;++y) for(int x=0;x<c.output_width;++x) for(int ch=0;ch<c.channels;++ch){
    const std::size_t oi=(static_cast<std::size_t>(y)*c.output_width+x)*c.channels+ch;
    if(c.interpolation==Interpolation::Nearest){int ix=std::min(c.input_width-1,static_cast<int>(x*sx));int iy=std::min(c.input_height-1,static_cast<int>(y*sy));out[oi]=in[(static_cast<std::size_t>(iy)*c.input_width+ix)*c.channels+ch];continue;}
    const float fx=std::max(0.0f,std::min(static_cast<float>(c.input_width-1),(x+0.5f)*sx-0.5f)), fy=std::max(0.0f,std::min(static_cast<float>(c.input_height-1),(y+0.5f)*sy-0.5f)); const int x0=static_cast<int>(std::floor(fx)), y0=static_cast<int>(std::floor(fy)); const int x1=std::min(c.input_width-1,x0+1), y1=std::min(c.input_height-1,y0+1); const float ax=fx-x0, ay=fy-y0;
    const auto at=[&](int px,int py){return static_cast<float>(in[(static_cast<std::size_t>(py)*c.input_width+px)*c.channels+ch]);};
    const float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x1,y0))+ay*((1-ax)*at(x0,y1)+ax*at(x1,y1)); out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround(v))));
  }
}
void random_scale_u8(const std::uint8_t* in,std::uint8_t* out,const ScaleConfig& c,cudaStream_t s){if(!in||!out||c.input_width<=0||c.input_height<=0||c.output_width<=0||c.output_height<=0||c.channels<=0||c.scale<=0)throw std::invalid_argument("invalid scale configuration");resize_u8(in,out,{c.input_width,c.input_height,c.output_width,c.output_height,c.channels,c.interpolation},s);}
void downscale_u8(const std::uint8_t* in,std::uint8_t* out,const DownscaleConfig& c,cudaStream_t s){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!std::isfinite(c.scale)||c.scale<=0.0f||c.scale>1.0f||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>2||static_cast<int>(c.upsample_interpolation)<0||static_cast<int>(c.upsample_interpolation)>2) throw std::invalid_argument("invalid downscale configuration");
  const int reduced_width=std::max(1,static_cast<int>(std::floor(c.width*c.scale))), reduced_height=std::max(1,static_cast<int>(std::floor(c.height*c.scale)));
  std::vector<std::uint8_t> reduced(static_cast<std::size_t>(reduced_width)*reduced_height*c.channels);
  resize_u8(in,reduced.data(),{c.width,c.height,reduced_width,reduced_height,c.channels,c.interpolation},s);
  resize_u8(reduced.data(),out,{reduced_width,reduced_height,c.width,c.height,c.channels,c.upsample_interpolation},s);
}
void scale_x_u8(const std::uint8_t* in,std::uint8_t* out,const AxisScaleConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.scale<=0)throw std::invalid_argument("invalid x scale configuration");float t=(1-c.scale)*(c.width-1)*0.5f;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,c.scale,0,t,0,1,0,c.interpolation,c.fill},s);}
void scale_y_u8(const std::uint8_t* in,std::uint8_t* out,const AxisScaleConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.scale<=0)throw std::invalid_argument("invalid y scale configuration");float t=(1-c.scale)*(c.height-1)*0.5f;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,0,0,0,c.scale,t,c.interpolation,c.fill},s);}
void translate_x_u8(const std::uint8_t* in,std::uint8_t* out,const AxisTranslateConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid x translation configuration");affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,0,c.offset,0,1,0,c.interpolation,c.fill},s);}
void translate_y_u8(const std::uint8_t* in,std::uint8_t* out,const AxisTranslateConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid y translation configuration");affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,0,0,0,1,c.offset,c.interpolation,c.fill},s);}
void shear_x_u8(const std::uint8_t* in,std::uint8_t* out,const AxisShearConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid x shear configuration");float t=-c.shear*(c.height-1)*0.5f;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,c.shear,t,0,1,0,c.interpolation,c.fill},s);}
void shear_y_u8(const std::uint8_t* in,std::uint8_t* out,const AxisShearConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid y shear configuration");float t=-c.shear*(c.width-1)*0.5f;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,0,0,c.shear,1,t,c.interpolation,c.fill},s);}
void longest_max_size_u8(const std::uint8_t* in,std::uint8_t* out,const MaxSizeConfig& c,cudaStream_t s){if(c.max_size<=0||c.output_width<=0||c.output_height<=0)throw std::invalid_argument("invalid longest max size configuration");int m=std::max(c.input_width,c.input_height);if(m>c.max_size){}else if(c.output_width!=c.input_width||c.output_height!=c.input_height)throw std::invalid_argument("invalid longest max size dimensions");resize_u8(in,out,{c.input_width,c.input_height,c.output_width,c.output_height,c.channels,c.interpolation},s);}
void smallest_max_size_u8(const std::uint8_t* in,std::uint8_t* out,const MaxSizeConfig& c,cudaStream_t s){if(c.max_size<=0||c.output_width<=0||c.output_height<=0)throw std::invalid_argument("invalid smallest max size configuration");resize_u8(in,out,{c.input_width,c.input_height,c.output_width,c.output_height,c.channels,c.interpolation},s);}
void affine_u8(const std::uint8_t* in,std::uint8_t* out,const AffineConfig& c,cudaStream_t){if(!in||!out||c.input_width<=0||c.input_height<=0||c.output_width<=0||c.output_height<=0||c.channels<=0)throw std::invalid_argument("invalid affine configuration");float det=c.m00*c.m11-c.m01*c.m10;if(std::fabs(det)<1e-8f)throw std::invalid_argument("singular affine matrix");float ia=c.m11/det,ib=-c.m01/det,ic=-c.m10/det,id=c.m00/det;for(int y=0;y<c.output_height;++y)for(int x=0;x<c.output_width;++x)for(int ch=0;ch<c.channels;++ch){float tx=x-c.m02,ty=y-c.m12,sx=ia*tx+ib*ty,sy=ic*tx+id*ty;std::size_t oi=(static_cast<std::size_t>(y)*c.output_width+x)*c.channels+ch;if(c.interpolation==Interpolation::Nearest){int ix=static_cast<int>(std::lround(sx)),iy=static_cast<int>(std::lround(sy));out[oi]=(ix>=0&&ix<c.input_width&&iy>=0&&iy<c.input_height)?in[(static_cast<std::size_t>(iy)*c.input_width+ix)*c.channels+ch]:c.fill;continue;}int x0=static_cast<int>(std::floor(sx)),y0=static_cast<int>(std::floor(sy));float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){if(px<0||px>=c.input_width||py<0||py>=c.input_height)return static_cast<float>(c.fill);return static_cast<float>(in[(static_cast<std::size_t>(py)*c.input_width+px)*c.channels+ch]);};float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround(v))));}}
void focus_breathing_u8(const std::uint8_t* in,std::uint8_t* out,const FocusBreathingConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1||!std::isfinite(c.focus_position)||!std::isfinite(c.breathing_strength)||!std::isfinite(c.radial_strength)||!std::isfinite(c.center_x)||!std::isfinite(c.center_y)||c.center_x<0.0f||c.center_x>1.0f||c.center_y<0.0f||c.center_y>1.0f)
    throw std::invalid_argument("invalid focus breathing configuration");
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels;
  if(c.focus_position==0.0f){for(std::size_t i=0;i<n;++i)out[i]=in[i];return;}
  const float cx=c.center_x*(c.width-1),cy=c.center_y*(c.height-1),rx=std::max(1.0f,(c.width-1)*0.5f),ry=std::max(1.0f,(c.height-1)*0.5f);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const float nx=(x-cx)/rx,ny=(y-cy)/ry,r2=nx*nx+ny*ny,scale=1.0f+c.focus_position*(c.breathing_strength+c.radial_strength*r2),sx=cx+(x-cx)/scale,sy=cy+(y-cy)/scale;for(int ch=0;ch<c.channels;++ch){const std::size_t oi=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;if(!std::isfinite(scale)||scale<=0.0f||!std::isfinite(sx)||!std::isfinite(sy)){out[oi]=c.fill;continue;}if(c.interpolation==Interpolation::Nearest){const int ix=static_cast<int>(std::lround(sx)),iy=static_cast<int>(std::lround(sy));out[oi]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]:c.fill;continue;}const int x0=static_cast<int>(std::floor(sx)),y0=static_cast<int>(std::floor(sy));const float ax=sx-x0,ay=sy-y0;const auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};const float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround(v))));}}
}
void safe_rotate_u8(const std::uint8_t* in,std::uint8_t* out,const SafeRotateConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid safe rotate configuration");float a=c.angle_degrees*3.14159265358979323846f/180.0f,co=std::cos(a),si=std::sin(a),cx=(c.width-1)*0.5f,cy=(c.height-1)*0.5f;float fit=std::min(1.0f,std::min(c.width/(std::fabs(co)*c.width+std::fabs(si)*c.height),c.height/(std::fabs(si)*c.width+std::fabs(co)*c.height)));co*=fit;si*=fit;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,co,si,cx-co*cx-si*cy,-si,co,cy+si*cx-co*cy,c.interpolation,c.fill},s);}
void perspective_u8(const std::uint8_t* in,std::uint8_t* out,const PerspectiveConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||std::fabs(c.h00*c.h11*c.h22+c.h01*c.h12*c.h20+c.h02*c.h10*c.h21-c.h02*c.h11*c.h20-c.h01*c.h10*c.h22-c.h00*c.h12*c.h21)<1e-8f)throw std::invalid_argument("invalid perspective configuration");for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){std::size_t oi=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;float d=c.h20*x+c.h21*y+c.h22;if(!std::isfinite(d)||d==0.0f){out[oi]=c.fill;continue;}float sx=(c.h00*x+c.h01*y+c.h02)/d,sy=(c.h10*x+c.h11*y+c.h12)/d;if(!std::isfinite(sx)||!std::isfinite(sy)){out[oi]=c.fill;continue;}if(c.interpolation==Interpolation::Nearest){if(std::fabs(sx)>=static_cast<float>(std::numeric_limits<int>::max())||std::fabs(sy)>=static_cast<float>(std::numeric_limits<int>::max())){out[oi]=c.fill;continue;}int ix=std::lround(sx),iy=std::lround(sy);out[oi]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]:c.fill;continue;}if(sx<=static_cast<float>(std::numeric_limits<int>::min())||sx>=static_cast<float>(std::numeric_limits<int>::max())||sy<=static_cast<float>(std::numeric_limits<int>::min())||sy>=static_cast<float>(std::numeric_limits<int>::max())){out[oi]=c.fill;continue;}int x0=static_cast<int>(std::floor(sx)),y0=static_cast<int>(std::floor(sy));float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround(v))));}}
void optical_distortion_u8(const std::uint8_t* in,std::uint8_t* out,const OpticalDistortionConfig& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid optical distortion configuration");float cx=(c.width-1)*.5f,cy=(c.height-1)*.5f,fx=std::max(1.f,c.width*.5f),fy=std::max(1.f,c.height*.5f);for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){float xn=(x-cx)/fx,yn=(y-cy)/fy,r2=xn*xn+yn*yn,rad=1+c.k1*r2+c.k2*r2*r2,sxn=xn*rad+2*c.p1*xn*yn+c.p2*(r2+2*xn*xn),syn=yn*rad+c.p1*(r2+2*yn*yn)+2*c.p2*xn*yn,sx=sxn*fx+cx,sy=syn*fy+cy;std::size_t oi=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;if(c.interpolation==Interpolation::Nearest){int ix=std::lround(sx),iy=std::lround(sy);out[oi]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]:c.fill;continue;}int x0=std::floor(sx),y0=std::floor(sy);float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround(v))));}}
void thin_prism_distortion_u8(const std::uint8_t* in,std::uint8_t* out,const ThinPrismDistortionConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1||!std::isfinite(c.s1)||!std::isfinite(c.s2)||!std::isfinite(c.t1)||!std::isfinite(c.t2)) throw std::invalid_argument("invalid thin-prism distortion configuration");
  const float cx=(c.width-1)*.5f,cy=(c.height-1)*.5f,fx=std::max(1.f,c.width*.5f),fy=std::max(1.f,c.height*.5f);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const float xn=(x-cx)/fx,yn=(y-cy)/fy,r2=xn*xn+yn*yn,r4=r2*r2,sx=(xn+c.s1*r2+c.s2*r4)*fx+cx,sy=(yn+c.t1*r2+c.t2*r4)*fy+cy;for(int ch=0;ch<c.channels;++ch){const std::size_t oi=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;if(c.interpolation==Interpolation::Nearest){const int ix=std::lround(sx),iy=std::lround(sy);out[oi]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]:c.fill;continue;}const int x0=std::floor(sx),y0=std::floor(sy);const float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround((1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1))))));}}
}
void rolling_shutter_geometric_distortion_u8(const std::uint8_t* in,std::uint8_t* out,const RollingShutterGeometricDistortionConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!c.row_shift_x||!c.row_shift_y||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1) throw std::invalid_argument("invalid rolling-shutter geometric distortion configuration");
  for(int y=0;y<c.height;++y)if(!std::isfinite(c.row_shift_x[y])||!std::isfinite(c.row_shift_y[y]))throw std::invalid_argument("rolling-shutter row shift is not finite");
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const float sx=x+c.row_shift_x[y],sy=y+c.row_shift_y[y];for(int ch=0;ch<c.channels;++ch){const std::size_t oi=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;if(c.interpolation==Interpolation::Nearest){const int ix=std::lround(sx),iy=std::lround(sy);out[oi]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]:c.fill;continue;}const int x0=std::floor(sx),y0=std::floor(sy);const float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround((1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1))))));}}
}
void grid_distortion_u8(const std::uint8_t* in,std::uint8_t* out,const GridDistortionConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.grid_width<2||c.grid_height<2||!c.displacement_x||!c.displacement_y||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1)
    throw std::invalid_argument("invalid grid distortion configuration");
  const auto displacement=[&](int x,int y){
    const float gx=(c.width==1)?0.0f:static_cast<float>(x)*(c.grid_width-1)/(c.width-1);
    const float gy=(c.height==1)?0.0f:static_cast<float>(y)*(c.grid_height-1)/(c.height-1);
    const int x0=std::min(c.grid_width-2,static_cast<int>(std::floor(gx))), y0=std::min(c.grid_height-2,static_cast<int>(std::floor(gy)));
    const float ax=gx-x0, ay=gy-y0;
    const std::size_t i00=static_cast<std::size_t>(y0)*c.grid_width+x0, i10=i00+1, i01=i00+c.grid_width, i11=i01+1;
    const auto blend=[&](const float* d){return (1-ay)*((1-ax)*d[i00]+ax*d[i10])+ay*((1-ax)*d[i01]+ax*d[i11]);};
    return std::pair<float,float>{blend(c.displacement_x),blend(c.displacement_y)};
  };
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x){
    const auto d=displacement(x,y); const float sx=x+d.first, sy=y+d.second;
    for(int ch=0;ch<c.channels;++ch){
      const std::size_t oi=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
      if(c.interpolation==Interpolation::Nearest){
        if(std::fabs(sx)>=static_cast<float>(std::numeric_limits<int>::max())||std::fabs(sy)>=static_cast<float>(std::numeric_limits<int>::max())){out[oi]=c.fill;continue;}
        const int ix=static_cast<int>(std::rint(sx)), iy=static_cast<int>(std::rint(sy));
        out[oi]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]:c.fill;
        continue;
      }
      const int x0=static_cast<int>(std::floor(sx)), y0=static_cast<int>(std::floor(sy)); const float ax=sx-x0, ay=sy-y0;
      const auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};
      const float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));
      out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround(v))));
    }
  }
}
void piecewise_affine_u8(const std::uint8_t* in,std::uint8_t* out,const PiecewiseAffineConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.grid_width<2||c.grid_height<2||!c.displacement_x||!c.displacement_y||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1)
    throw std::invalid_argument("invalid piecewise affine configuration");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x){
    const float gx=c.width==1?0.0f:static_cast<float>(x)*(c.grid_width-1)/(c.width-1);
    const float gy=c.height==1?0.0f:static_cast<float>(y)*(c.grid_height-1)/(c.height-1);
    const int x0=std::min(c.grid_width-2,static_cast<int>(std::floor(gx))), y0=std::min(c.grid_height-2,static_cast<int>(std::floor(gy)));
    const float u=gx-x0,v=gy-y0; const std::size_t tl=static_cast<std::size_t>(y0)*c.grid_width+x0,tr=tl+1,bl=tl+c.grid_width,br=bl+1;
    float dx,dy;
    if(v<=u){dx=(1-u)*c.displacement_x[tl]+(u-v)*c.displacement_x[tr]+v*c.displacement_x[br];dy=(1-u)*c.displacement_y[tl]+(u-v)*c.displacement_y[tr]+v*c.displacement_y[br];}
    else{dx=(1-v)*c.displacement_x[tl]+u*c.displacement_x[br]+(v-u)*c.displacement_x[bl];dy=(1-v)*c.displacement_y[tl]+u*c.displacement_y[br]+(v-u)*c.displacement_y[bl];}
    const float sx=x+dx,sy=y+dy;
    for(int ch=0;ch<c.channels;++ch){
      const std::size_t oi=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
      if(!std::isfinite(sx)||!std::isfinite(sy)||std::fabs(sx)>=static_cast<float>(std::numeric_limits<int>::max())||std::fabs(sy)>=static_cast<float>(std::numeric_limits<int>::max())){out[oi]=c.fill;continue;}
      if(c.interpolation==Interpolation::Nearest){const int ix=static_cast<int>(std::rint(sx)),iy=static_cast<int>(std::rint(sy));out[oi]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]:c.fill;continue;}
      const int sx0=static_cast<int>(std::floor(sx)),sy0=static_cast<int>(std::floor(sy));const float ax=sx-sx0,ay=sy-sy0;
      const auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};
      const float value=(1-ay)*((1-ax)*at(sx0,sy0)+ax*at(sx0+1,sy0))+ay*((1-ax)*at(sx0,sy0+1)+ax*at(sx0+1,sy0+1));
      out[oi]=static_cast<std::uint8_t>(std::max(0L,std::min(255L,std::lround(value))));
    }
  }
}
void elastic_transform_u8(const std::uint8_t* in,std::uint8_t* out,const ElasticTransformConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!c.displacement_x||!c.displacement_y||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1)
    throw std::invalid_argument("invalid elastic transform configuration");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x){
    const std::size_t di=static_cast<std::size_t>(y)*c.width+x;
    const float dx=c.displacement_x[di],dy=c.displacement_y[di];
    const float sx=x+dx,sy=y+dy;
    for(int ch=0;ch<c.channels;++ch){
      const std::size_t oi=(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch;
      if(!std::isfinite(sx)||!std::isfinite(sy)||std::fabs(sx)>=static_cast<float>(std::numeric_limits<int>::max())||std::fabs(sy)>=static_cast<float>(std::numeric_limits<int>::max())){out[oi]=c.fill;continue;}
      if(c.interpolation==Interpolation::Nearest){
        const int ix=static_cast<int>(std::rint(sx)),iy=static_cast<int>(std::rint(sy));
        out[oi]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(static_cast<std::size_t>(iy)*c.width+ix)*c.channels+ch]:c.fill;continue;
      }
      const int x0=static_cast<int>(std::floor(sx)),y0=static_cast<int>(std::floor(sy)); const float ax=sx-x0,ay=sy-y0;
      const auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?static_cast<float>(in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch]):static_cast<float>(c.fill);};
      const float value=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));
      out[oi]=static_cast<std::uint8_t>(std::max(0.0f,std::min(255.0f,std::floor(value+0.5f))));
    }
  }
}
void perspective_transform_u8(const std::uint8_t* in,std::uint8_t* out,const PerspectiveConfig& c,cudaStream_t s){perspective_u8(in,out,c,s);}
namespace {
std::uint64_t random_grid_shuffle_splitmix64(std::uint64_t value){
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
void validate_random_grid_shuffle(const std::uint8_t* in,const std::uint8_t* out,const RandomGridShuffleConfig& c){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.grid_rows<=0||c.grid_cols<=0||c.grid_rows>c.height||c.grid_cols>c.width)
    throw std::invalid_argument("invalid random grid shuffle configuration");
  const std::size_t count=static_cast<std::size_t>(c.grid_rows)*static_cast<std::size_t>(c.grid_cols);
  if(count>std::numeric_limits<std::uint32_t>::max())throw std::invalid_argument("random grid shuffle grid is too large");
  if(c.cell_permutation){
    std::vector<bool> seen(count,false);
    for(std::size_t i=0;i<count;++i){const std::uint32_t source=c.cell_permutation[i];if(source>=count||seen[source])throw std::invalid_argument("random grid shuffle permutation is not a bijection");seen[source]=true;}
  }
}
}
void make_random_grid_shuffle_permutation(std::uint32_t* permutation,int grid_rows,int grid_cols,std::uint64_t seed){
  if(!permutation||grid_rows<=0||grid_cols<=0)throw std::invalid_argument("invalid random grid shuffle permutation dimensions");
  const std::size_t count=static_cast<std::size_t>(grid_rows)*static_cast<std::size_t>(grid_cols);
  if(count>std::numeric_limits<std::uint32_t>::max())throw std::invalid_argument("random grid shuffle grid is too large");
  for(std::size_t i=0;i<count;++i)permutation[i]=static_cast<std::uint32_t>(i);
  for(std::size_t i=count-1;i>0;--i){const std::size_t j=static_cast<std::size_t>(random_grid_shuffle_splitmix64(seed+static_cast<std::uint64_t>(i))%(i+1));std::swap(permutation[i],permutation[j]);}
}
void random_grid_shuffle_u8(const std::uint8_t* in,std::uint8_t* out,const RandomGridShuffleConfig& c,cudaStream_t){
  validate_random_grid_shuffle(in,out,c);
  const std::size_t count=static_cast<std::size_t>(c.grid_rows)*static_cast<std::size_t>(c.grid_cols);
  std::vector<std::uint32_t> generated;
  const std::uint32_t* permutation=c.cell_permutation;
  if(!permutation){generated.resize(count);make_random_grid_shuffle_permutation(generated.data(),c.grid_rows,c.grid_cols,c.seed);permutation=generated.data();}
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){
    const int destination_row=y*c.grid_rows/c.height,destination_col=x*c.grid_cols/c.width;
    const std::size_t destination=static_cast<std::size_t>(destination_row)*c.grid_cols+destination_col;
    const std::size_t source=permutation[destination];
    const int destination_x0=destination_col*c.width/c.grid_cols,destination_x1=(destination_col+1)*c.width/c.grid_cols;
    const int destination_y0=destination_row*c.height/c.grid_rows,destination_y1=(destination_row+1)*c.height/c.grid_rows;
    const int source_col=static_cast<int>(source%c.grid_cols),source_row=static_cast<int>(source/c.grid_cols);
    const int source_x0=source_col*c.width/c.grid_cols,source_x1=(source_col+1)*c.width/c.grid_cols;
    const int source_y0=source_row*c.height/c.grid_rows,source_y1=(source_row+1)*c.height/c.grid_rows;
    const int local_x=x-destination_x0,local_y=y-destination_y0,destination_width=destination_x1-destination_x0,destination_height=destination_y1-destination_y0;
    const int source_x=source_x0+std::min(source_x1-source_x0-1,static_cast<int>((static_cast<long long>(local_x)*(source_x1-source_x0))/destination_width));
    const int source_y=source_y0+std::min(source_y1-source_y0-1,static_cast<int>((static_cast<long long>(local_y)*(source_y1-source_y0))/destination_height));
    out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=in[(static_cast<std::size_t>(source_y)*c.width+source_x)*c.channels+ch];
  }
}
void rotate_u8(const std::uint8_t* in,std::uint8_t* out,const RotateConfig& c,cudaStream_t s){float a=c.angle_degrees*3.14159265358979323846f/180.0f,co=std::cos(a),si=std::sin(a),cx=(c.width-1)*0.5f,cy=(c.height-1)*0.5f;AffineConfig ac{c.width,c.height,c.width,c.height,c.channels,co,si,cx-co*cx-si*cy,-si,co,cy+si*cx-co*cy,c.interpolation,c.fill};affine_u8(in,out,ac,s);}
void shift_scale_rotate_u8(const std::uint8_t* in,std::uint8_t* out,const ShiftScaleRotateConfig& c,cudaStream_t s){float a=c.angle_degrees*3.14159265358979323846f/180.0f,co=c.scale*std::cos(a),si=c.scale*std::sin(a),cx=(c.width-1)*0.5f,cy=(c.height-1)*0.5f;AffineConfig ac{c.width,c.height,c.width,c.height,c.channels,co,si,cx-co*cx-si*cy+c.shift_x*c.width,-si,co,cy+si*cx-co*cy+c.shift_y*c.height,c.interpolation,c.fill};affine_u8(in,out,ac,s);}
void transpose_u8(const std::uint8_t* in,std::uint8_t* out,const TransposeConfig& c,cudaStream_t){if(!in||!out||c.input_width<=0||c.input_height<=0||c.channels<=0)throw std::invalid_argument("invalid transpose configuration");for(int y=0;y<c.input_height;++y)for(int x=0;x<c.input_width;++x)for(int ch=0;ch<c.channels;++ch)out[(x*c.input_height+y)*c.channels+ch]=in[(y*c.input_width+x)*c.channels+ch];}
void rotate90_u8(const std::uint8_t* in,std::uint8_t* out,const Rotate90Config& c,cudaStream_t){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid rotate90 configuration");int k=((c.k%4)+4)%4;for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x)for(int ch=0;ch<c.channels;++ch){int ox,oy;if(k==0){ox=x;oy=y;}else if(k==1){ox=c.height-1-y;oy=x;}else if(k==2){ox=c.width-1-x;oy=c.height-1-y;}else{ox=y;oy=c.width-1-x;}int ow=(k%2)?c.height:c.width;out[(oy*ow+ox)*c.channels+ch]=in[(y*c.width+x)*c.channels+ch];}}
}
