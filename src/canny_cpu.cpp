#include "augmatch/filter/canny.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace augmatch {
namespace {
int canny_index(int p,int n,BorderPolicy border){
  if(p>=0&&p<n)return p;
  return border==BorderPolicy::Clamp?std::max(0,std::min(n-1,p)): -1;
}
float canny_luma(const std::uint8_t* in,int width,int height,int channels,int x,int y,const CannyConfig& c){
  const int px=canny_index(x,width,c.border),py=canny_index(y,height,c.border);
  if(px<0||py<0)return static_cast<float>(c.border_value);
  const std::size_t at=(static_cast<std::size_t>(py)*width+px)*channels;
  if(channels==1)return static_cast<float>(in[at]);
  const float r=static_cast<float>(in[at]),g=static_cast<float>(in[at+1]),b=static_cast<float>(in[at+2]);
  return 0.299f*r+0.587f*g+0.114f*b;
}
void canny_kernel(int aperture,std::vector<float>& d,std::vector<float>& s){
  if(aperture==3){d={-1,0,1};s={1,2,1};return;}
  if(aperture==5){d={-1,-2,0,2,1};s={1,4,6,4,1};return;}
  d={-1,-4,-5,0,5,4,1};s={1,6,15,20,15,6,1};
}
void canny_gradient(const std::uint8_t* in,const CannyConfig& c,std::vector<float>& gx,std::vector<float>& gy){
  std::vector<float> dx,sm; canny_kernel(c.aperture_size,dx,sm); const int radius=c.aperture_size/2;
  const std::size_t pixels=static_cast<std::size_t>(c.width)*c.height; gx.assign(pixels,0.0f);gy.assign(pixels,0.0f);
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){float xsum=0.0f,ysum=0.0f;
    for(int ky=-radius;ky<=radius;++ky)for(int kx=-radius;kx<=radius;++kx){
      const float value=canny_luma(in,c.width,c.height,c.channels,x+kx,y+ky,c); const int ix=kx+radius,iy=ky+radius;
      xsum+=dx[ix]*sm[iy]*value; ysum+=sm[ix]*dx[iy]*value;
    }
    const std::size_t at=static_cast<std::size_t>(y)*c.width+x;gx[at]=xsum;gy[at]=ysum;
  }
}
void validate_canny(const std::uint8_t* in,const std::uint8_t* out,const CannyConfig& c){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!std::isfinite(c.low_threshold)||!std::isfinite(c.high_threshold)||c.low_threshold<0.0f||c.high_threshold<c.low_threshold||c.aperture_size<3||c.aperture_size>7||!(c.aperture_size&1)||static_cast<int>(c.border)<0||static_cast<int>(c.border)>1)throw std::invalid_argument("invalid Canny configuration");
}
}
void canny_u8(const std::uint8_t* in,std::uint8_t* out,const CannyConfig& c,cudaStream_t){
  validate_canny(in,out,c); const std::size_t pixels=static_cast<std::size_t>(c.width)*c.height;
  std::vector<float> gx,gy; canny_gradient(in,c,gx,gy); std::vector<std::uint8_t> state(pixels,0); std::vector<std::size_t> stack; stack.reserve(pixels);
  const float tangent=0.41421356237309505f;
  auto magnitude=[&](std::size_t at){return std::abs(gx[at])+std::abs(gy[at]);};
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const std::size_t at=static_cast<std::size_t>(y)*c.width+x;const float ax=std::abs(gx[at]),ay=std::abs(gy[at]),m=magnitude(at);if(m<c.low_threshold)continue;
    int dx1=0,dy1=0,dx2=0,dy2=0;
    if(ay<=ax*tangent){dx1=1;dx2=-1;} else if(ax<=ay*tangent){dy1=1;dy2=-1;} else {dx1=gx[at]*gy[at]>=0.0f?1:-1;dy1=1;dx2=-dx1;dy2=-1;}
    const int x1=x+dx1,y1=y+dy1,x2=x+dx2,y2=y+dy2;
    const float m1=(x1<0||y1<0)?0.0f:magnitude(static_cast<std::size_t>(y1)*c.width+x1),m2=(x2<0||y2<0)?0.0f:magnitude(static_cast<std::size_t>(y2)*c.width+x2);
    if(m<m1||m<m2)continue; state[at]=m>=c.high_threshold?2:1;if(state[at]==2)stack.push_back(at);
  }
  while(!stack.empty()){const std::size_t at=stack.back();stack.pop_back();const int x=static_cast<int>(at%c.width),y=static_cast<int>(at/c.width);for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)if(dx||dy){const int nx=x+dx,ny=y+dy;if(nx>=0&&nx<c.width&&ny>=0&&ny<c.height){const std::size_t next=static_cast<std::size_t>(ny)*c.width+nx;if(state[next]==1){state[next]=2;stack.push_back(static_cast<std::uint8_t>(next));}}}}
  for(int y=0;y<c.height;++y)for(int x=0;x<c.width;++x){const std::uint8_t value=state[static_cast<std::size_t>(y)*c.width+x]==2?255:0;for(int ch=0;ch<c.channels;++ch)out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=value;}
}
}
