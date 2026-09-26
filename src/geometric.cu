#include "augmatch/geometry/geometric.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__device__ unsigned char sample(const unsigned char* in,int w,int h,int c,int x,int y,int ch){x=max(0,min(w-1,x));y=max(0,min(h-1,y));return in[(y*w+x)*c+ch];}
__global__ void resize_kernel(const unsigned char* in,unsigned char* out,ResizeConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.output_width*c.output_height*c.channels;if(i>=n)return;int ch=i%c.channels,x=(i/c.channels)%c.output_width,y=i/(c.output_width*c.channels);float sx=(float)c.input_width/c.output_width,sy=(float)c.input_height/c.output_height;if(c.interpolation==Interpolation::Nearest){int ix=min(c.input_width-1,(int)(x*sx)),iy=min(c.input_height-1,(int)(y*sy));out[i]=sample(in,c.input_width,c.input_height,c.channels,ix,iy,ch);return;}float fx=fminf((float)(c.input_width-1),fmaxf(0.f,(x+.5f)*sx-.5f)),fy=fminf((float)(c.input_height-1),fmaxf(0.f,(y+.5f)*sy-.5f));int x0=(int)floorf(fx),y0=(int)floorf(fy),x1=min(c.input_width-1,x0+1),y1=min(c.input_height-1,y0+1);float ax=fx-x0,ay=fy-y0;float v=(1-ay)*((1-ax)*sample(in,c.input_width,c.input_height,c.channels,x0,y0,ch)+ax*sample(in,c.input_width,c.input_height,c.channels,x1,y0,ch))+ay*((1-ax)*sample(in,c.input_width,c.input_height,c.channels,x0,y1,ch)+ax*sample(in,c.input_width,c.input_height,c.channels,x1,y1,ch));out[i]=(unsigned char)fminf(255.f,fmaxf(0.f,rintf(v)));}
__global__ void transpose_kernel(const unsigned char* in,unsigned char* out,TransposeConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.input_width*c.input_height*c.channels;if(i>=n)return;int ch=i%c.channels,x=(i/c.channels)%c.input_width,y=i/(c.input_width*c.channels);out[(x*c.input_height+y)*c.channels+ch]=in[i];}
__global__ void rotate_kernel(const unsigned char* in,unsigned char* out,Rotate90Config c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int ch=i%c.channels,x=(i/c.channels)%c.width,y=i/(c.width*c.channels),k=((c.k%4)+4)%4,ox,oy;if(k==0){ox=x;oy=y;}else if(k==1){ox=c.height-1-y;oy=x;}else if(k==2){ox=c.width-1-x;oy=c.height-1-y;}else{ox=y;oy=c.width-1-x;}int ow=(k%2)?c.height:c.width;out[(oy*ow+ox)*c.channels+ch]=in[i];}
__device__ unsigned long long random_grid_shuffle_splitmix64(unsigned long long value){
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
__device__ unsigned int random_grid_shuffle_seed_source(int destination,int count,unsigned long long seed){
  int position=destination;
  // Undo the descending Fisher-Yates swaps in ascending order to recover the
  // source cell occupying this destination position.
  for(int i=1;i<count;++i){
    const int j=(int)(random_grid_shuffle_splitmix64(seed+(unsigned long long)i)%(unsigned long long)(i+1));
    if(position==i)position=j;else if(position==j)position=i;
  }
  return (unsigned int)position;
}
__global__ void random_grid_shuffle_kernel(const unsigned char* in,unsigned char* out,RandomGridShuffleConfig c){
  const int q=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(q>=n)return;
  const int ch=q%c.channels,x=(q/c.channels)%c.width,y=q/(c.width*c.channels);
  const int destination_row=y*c.grid_rows/c.height,destination_col=x*c.grid_cols/c.width;
  const int destination=destination_row*c.grid_cols+destination_col;
  const unsigned int source=c.cell_permutation?c.cell_permutation[destination]:random_grid_shuffle_seed_source(destination,c.grid_rows*c.grid_cols,c.seed);
  const int destination_x0=destination_col*c.width/c.grid_cols,destination_x1=(destination_col+1)*c.width/c.grid_cols;
  const int destination_y0=destination_row*c.height/c.grid_rows,destination_y1=(destination_row+1)*c.height/c.grid_rows;
  const int source_col=source%c.grid_cols,source_row=source/c.grid_cols;
  const int source_x0=source_col*c.width/c.grid_cols,source_x1=(source_col+1)*c.width/c.grid_cols;
  const int source_y0=source_row*c.height/c.grid_rows,source_y1=(source_row+1)*c.height/c.grid_rows;
  const int local_x=x-destination_x0,local_y=y-destination_y0,destination_width=destination_x1-destination_x0,destination_height=destination_y1-destination_y0;
  const int source_x=source_x0+min(source_x1-source_x0-1,(int)(((long long)local_x*(source_x1-source_x0))/destination_width));
  const int source_y=source_y0+min(source_y1-source_y0-1,(int)(((long long)local_y*(source_y1-source_y0))/destination_height));
  out[q]=in[(source_y*c.width+source_x)*c.channels+ch];
}
void check(cudaError_t e,const char* p){if(e!=cudaSuccess)throw std::runtime_error(std::string(p)+": "+cudaGetErrorString(e));}
}
void resize_u8(const std::uint8_t* in,std::uint8_t* out,const ResizeConfig& c,cudaStream_t s){if(!in||!out||c.input_width<=0||c.input_height<=0||c.output_width<=0||c.output_height<=0||c.channels<=0||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>2)throw std::invalid_argument("invalid resize configuration");int n=c.output_width*c.output_height*c.channels;resize_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"resize_kernel");}
void random_scale_u8(const std::uint8_t* in,std::uint8_t* out,const ScaleConfig& c,cudaStream_t s){if(!in||!out||c.input_width<=0||c.input_height<=0||c.output_width<=0||c.output_height<=0||c.channels<=0||c.scale<=0)throw std::invalid_argument("invalid scale configuration");resize_u8(in,out,{c.input_width,c.input_height,c.output_width,c.output_height,c.channels,c.interpolation},s);}
void downscale_u8(const std::uint8_t* in,std::uint8_t* out,const DownscaleConfig& c,cudaStream_t s){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!std::isfinite(c.scale)||c.scale<=0.0f||c.scale>1.0f||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>2||static_cast<int>(c.upsample_interpolation)<0||static_cast<int>(c.upsample_interpolation)>2) throw std::invalid_argument("invalid downscale configuration");
  const int reduced_width=std::max(1,static_cast<int>(std::floor(c.width*c.scale))), reduced_height=std::max(1,static_cast<int>(std::floor(c.height*c.scale)));
  const std::size_t reduced_bytes=static_cast<std::size_t>(reduced_width)*reduced_height*c.channels;
  std::uint8_t* reduced=nullptr;
  check(cudaMalloc(&reduced,reduced_bytes),"cudaMalloc downscale temporary");
  try {
    resize_u8(in,reduced,{c.width,c.height,reduced_width,reduced_height,c.channels,c.interpolation},s);
    resize_u8(reduced,out,{reduced_width,reduced_height,c.width,c.height,c.channels,c.upsample_interpolation},s);
    check(cudaFree(reduced),"cudaFree downscale temporary");
  } catch(...) {
    cudaFree(reduced);
    throw;
  }
}
void scale_x_u8(const std::uint8_t* in,std::uint8_t* out,const AxisScaleConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.scale<=0)throw std::invalid_argument("invalid x scale configuration");float t=(1-c.scale)*(c.width-1)*0.5f;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,c.scale,0,t,0,1,0,c.interpolation,c.fill},s);}
void scale_y_u8(const std::uint8_t* in,std::uint8_t* out,const AxisScaleConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.scale<=0)throw std::invalid_argument("invalid y scale configuration");float t=(1-c.scale)*(c.height-1)*0.5f;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,0,0,0,c.scale,t,c.interpolation,c.fill},s);}
void translate_x_u8(const std::uint8_t* in,std::uint8_t* out,const AxisTranslateConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid x translation configuration");affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,0,c.offset,0,1,0,c.interpolation,c.fill},s);}
void translate_y_u8(const std::uint8_t* in,std::uint8_t* out,const AxisTranslateConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid y translation configuration");affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,0,0,0,1,c.offset,c.interpolation,c.fill},s);}
void shear_x_u8(const std::uint8_t* in,std::uint8_t* out,const AxisShearConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid x shear configuration");float t=-c.shear*(c.height-1)*0.5f;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,c.shear,t,0,1,0,c.interpolation,c.fill},s);}
void shear_y_u8(const std::uint8_t* in,std::uint8_t* out,const AxisShearConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid y shear configuration");float t=-c.shear*(c.width-1)*0.5f;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,1,0,0,c.shear,1,t,c.interpolation,c.fill},s);}
void longest_max_size_u8(const std::uint8_t* in,std::uint8_t* out,const MaxSizeConfig& c,cudaStream_t s){if(c.max_size<=0||c.output_width<=0||c.output_height<=0)throw std::invalid_argument("invalid longest max size configuration");resize_u8(in,out,{c.input_width,c.input_height,c.output_width,c.output_height,c.channels,c.interpolation},s);}
void smallest_max_size_u8(const std::uint8_t* in,std::uint8_t* out,const MaxSizeConfig& c,cudaStream_t s){if(c.max_size<=0||c.output_width<=0||c.output_height<=0)throw std::invalid_argument("invalid smallest max size configuration");resize_u8(in,out,{c.input_width,c.input_height,c.output_width,c.output_height,c.channels,c.interpolation},s);}
__global__ void optical_kernel(const unsigned char* in,unsigned char* out,OpticalDistortionConfig c){int q=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(q>=n)return;int ch=q%c.channels,x=(q/c.channels)%c.width,y=q/(c.width*c.channels);float cx=(c.width-1)*.5f,cy=(c.height-1)*.5f,fx=fmaxf(1.f,c.width*.5f),fy=fmaxf(1.f,c.height*.5f),xn=(x-cx)/fx,yn=(y-cy)/fy,r2=xn*xn+yn*yn,rad=1+c.k1*r2+c.k2*r2*r2,sx=(xn*rad+2*c.p1*xn*yn+c.p2*(r2+2*xn*xn))*fx+cx,sy=(yn*rad+c.p1*(r2+2*yn*yn)+2*c.p2*xn*yn)*fy+cy;if(c.interpolation==Interpolation::Nearest){int ix=(int)rintf(sx),iy=(int)rintf(sy);out[q]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(iy*c.width+ix)*c.channels+ch]:c.fill;return;}int x0=(int)floorf(sx),y0=(int)floorf(sy);float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?(float)in[(py*c.width+px)*c.channels+ch]:(float)c.fill;};float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));out[q]=(unsigned char)fminf(255.f,fmaxf(0.f,rintf(v)));}
__global__ void thin_prism_kernel(const unsigned char* in,unsigned char* out,ThinPrismDistortionConfig c){int q=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(q>=n)return;int ch=q%c.channels,x=(q/c.channels)%c.width,y=q/(c.width*c.channels);float cx=(c.width-1)*.5f,cy=(c.height-1)*.5f,fx=fmaxf(1.f,c.width*.5f),fy=fmaxf(1.f,c.height*.5f),xn=(x-cx)/fx,yn=(y-cy)/fy,r2=xn*xn+yn*yn,r4=r2*r2,sx=(xn+c.s1*r2+c.s2*r4)*fx+cx,sy=(yn+c.t1*r2+c.t2*r4)*fy+cy;auto at=[&](int px,int py){return px>=0&&px<c.width&&py>=0&&py<c.height?(float)in[(py*c.width+px)*c.channels+ch]:(float)c.fill;};if(c.interpolation==Interpolation::Nearest){int ix=(int)rintf(sx),iy=(int)rintf(sy);out[q]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(iy*c.width+ix)*c.channels+ch]:c.fill;return;}int x0=(int)floorf(sx),y0=(int)floorf(sy);float ax=sx-x0,ay=sy-y0;out[q]=(unsigned char)fminf(255.f,fmaxf(0.f,floorf((1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1))+.5f)));}
__global__ void rolling_shutter_geometric_kernel(const unsigned char* in,unsigned char* out,RollingShutterGeometricDistortionConfig c){int q=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(q>=n)return;int ch=q%c.channels,x=(q/c.channels)%c.width,y=q/(c.width*c.channels);float sx=x+c.row_shift_x[y],sy=y+c.row_shift_y[y];auto at=[&](int px,int py){return px>=0&&px<c.width&&py>=0&&py<c.height?(float)in[(py*c.width+px)*c.channels+ch]:(float)c.fill;};if(c.interpolation==Interpolation::Nearest){int ix=(int)rintf(sx),iy=(int)rintf(sy);out[q]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(iy*c.width+ix)*c.channels+ch]:c.fill;return;}int x0=(int)floorf(sx),y0=(int)floorf(sy);float ax=sx-x0,ay=sy-y0;out[q]=(unsigned char)fminf(255.f,fmaxf(0.f,floorf((1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1))+.5f)));}
__global__ void grid_distortion_kernel(const unsigned char* in,unsigned char* out,GridDistortionConfig c){
  int q=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(q>=n)return;
  int ch=q%c.channels,x=(q/c.channels)%c.width,y=q/(c.width*c.channels);
  float gx=c.width==1?0.0f:(float)x*(c.grid_width-1)/(c.width-1),gy=c.height==1?0.0f:(float)y*(c.grid_height-1)/(c.height-1);
  int x0=min(c.grid_width-2,(int)floorf(gx)),y0=min(c.grid_height-2,(int)floorf(gy));float ax=gx-x0,ay=gy-y0;
  int i00=y0*c.grid_width+x0,i10=i00+1,i01=i00+c.grid_width,i11=i01+1;
  float dx=(1-ay)*((1-ax)*c.displacement_x[i00]+ax*c.displacement_x[i10])+ay*((1-ax)*c.displacement_x[i01]+ax*c.displacement_x[i11]);
  float dy=(1-ay)*((1-ax)*c.displacement_y[i00]+ax*c.displacement_y[i10])+ay*((1-ax)*c.displacement_y[i01]+ax*c.displacement_y[i11]);
  float sx=x+dx,sy=y+dy;
  if(c.interpolation==Interpolation::Nearest){if(!isfinite(sx)||!isfinite(sy)||fabsf(sx)>=2147483647.0f||fabsf(sy)>=2147483647.0f){out[q]=c.fill;return;}int ix=(int)rintf(sx),iy=(int)rintf(sy);out[q]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(iy*c.width+ix)*c.channels+ch]:c.fill;return;}
  if(!isfinite(sx)||!isfinite(sy)||fabsf(sx)>=2147483647.0f||fabsf(sy)>=2147483647.0f){out[q]=c.fill;return;}
  int sx0=(int)floorf(sx),sy0=(int)floorf(sy);float bx=sx-sx0,by=sy-sy0;
  auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?(float)in[(py*c.width+px)*c.channels+ch]:(float)c.fill;};
  float v=(1-by)*((1-bx)*at(sx0,sy0)+bx*at(sx0+1,sy0))+by*((1-bx)*at(sx0,sy0+1)+bx*at(sx0+1,sy0+1));out[q]=(unsigned char)fminf(255.f,fmaxf(0.f,rintf(v)));
}
__global__ void piecewise_affine_kernel(const unsigned char* in,unsigned char* out,PiecewiseAffineConfig c){
  int q=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(q>=n)return;
  int ch=q%c.channels,x=(q/c.channels)%c.width,y=q/(c.width*c.channels);
  float gx=c.width==1?0.0f:(float)x*(c.grid_width-1)/(c.width-1),gy=c.height==1?0.0f:(float)y*(c.grid_height-1)/(c.height-1);
  int x0=min(c.grid_width-2,(int)floorf(gx)),y0=min(c.grid_height-2,(int)floorf(gy));float u=gx-x0,v=gy-y0;
  int tl=y0*c.grid_width+x0,tr=tl+1,bl=tl+c.grid_width,br=bl+1;float dx,dy;
  if(v<=u){dx=(1-u)*c.displacement_x[tl]+(u-v)*c.displacement_x[tr]+v*c.displacement_x[br];dy=(1-u)*c.displacement_y[tl]+(u-v)*c.displacement_y[tr]+v*c.displacement_y[br];}
  else{dx=(1-v)*c.displacement_x[tl]+u*c.displacement_x[br]+(v-u)*c.displacement_x[bl];dy=(1-v)*c.displacement_y[tl]+u*c.displacement_y[br]+(v-u)*c.displacement_y[bl];}
  float sx=x+dx,sy=y+dy;if(!isfinite(sx)||!isfinite(sy)||fabsf(sx)>=2147483647.0f||fabsf(sy)>=2147483647.0f){out[q]=c.fill;return;}
  if(c.interpolation==Interpolation::Nearest){int ix=(int)rintf(sx),iy=(int)rintf(sy);out[q]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(iy*c.width+ix)*c.channels+ch]:c.fill;return;}
  int sx0=(int)floorf(sx),sy0=(int)floorf(sy);float ax=sx-sx0,ay=sy-sy0;
  auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?(float)in[(py*c.width+px)*c.channels+ch]:(float)c.fill;};
  float value=(1-ay)*((1-ax)*at(sx0,sy0)+ax*at(sx0+1,sy0))+ay*((1-ax)*at(sx0,sy0+1)+ax*at(sx0+1,sy0+1));out[q]=(unsigned char)fminf(255.f,fmaxf(0.f,rintf(value)));
}
__global__ void elastic_transform_kernel(const unsigned char* in,unsigned char* out,ElasticTransformConfig c){
  int q=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(q>=n)return;
  int ch=q%c.channels,x=(q/c.channels)%c.width,y=q/(c.width*c.channels);int di=y*c.width+x;
  float sx=x+c.displacement_x[di],sy=y+c.displacement_y[di];
  if(!isfinite(sx)||!isfinite(sy)||fabsf(sx)>=2147483647.0f||fabsf(sy)>=2147483647.0f){out[q]=c.fill;return;}
  if(c.interpolation==Interpolation::Nearest){int ix=(int)rintf(sx),iy=(int)rintf(sy);out[q]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(iy*c.width+ix)*c.channels+ch]:c.fill;return;}
  int x0=(int)floorf(sx),y0=(int)floorf(sy);float ax=sx-x0,ay=sy-y0;
  auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?(float)in[(py*c.width+px)*c.channels+ch]:(float)c.fill;};
  float value=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));
  out[q]=(unsigned char)fminf(255.0f,fmaxf(0.0f,floorf(value+0.5f)));
}
__global__ void perspective_kernel(const unsigned char* in,unsigned char* out,PerspectiveConfig c){int q=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(q>=n)return;int ch=q%c.channels,x=(q/c.channels)%c.width,y=q/(c.width*c.channels);float d=c.h20*x+c.h21*y+c.h22;if(!isfinite(d)||d==0.0f){out[q]=c.fill;return;}float sx=(c.h00*x+c.h01*y+c.h02)/d,sy=(c.h10*x+c.h11*y+c.h12)/d;if(!isfinite(sx)||!isfinite(sy)){out[q]=c.fill;return;}if(sx < -1.0f||sx > (float)c.width||sy < -1.0f||sy > (float)c.height){out[q]=c.fill;return;}if(c.interpolation==Interpolation::Nearest){int ix=(int)rintf(sx),iy=(int)rintf(sy);out[q]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(iy*c.width+ix)*c.channels+ch]:c.fill;return;}int x0=(int)floorf(sx),y0=(int)floorf(sy);float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){return (px>=0&&px<c.width&&py>=0&&py<c.height)?(float)in[(py*c.width+px)*c.channels+ch]:(float)c.fill;};float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));out[q]=(unsigned char)fminf(255.f,fmaxf(0.f,rintf(v)));}
__global__ void affine_kernel(const unsigned char* in,unsigned char* out,AffineConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.output_width*c.output_height*c.channels;if(i>=n)return;int ch=i%c.channels,x=(i/c.channels)%c.output_width,y=i/(c.output_width*c.channels);float det=c.m00*c.m11-c.m01*c.m10,ia=c.m11/det,ib=-c.m01/det,ic=-c.m10/det,id=c.m00/det,sx=ia*(x-c.m02)+ib*(y-c.m12),sy=ic*(x-c.m02)+id*(y-c.m12);if(c.interpolation==Interpolation::Nearest){int ix=(int)rintf(sx),iy=(int)rintf(sy);out[i]=(ix>=0&&ix<c.input_width&&iy>=0&&iy<c.input_height)?in[(iy*c.input_width+ix)*c.channels+ch]:c.fill;return;}int x0=(int)floorf(sx),y0=(int)floorf(sy);float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){return (px<0||px>=c.input_width||py<0||py>=c.input_height)?(float)c.fill:(float)in[(py*c.input_width+px)*c.channels+ch];};float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));out[i]=(unsigned char)fminf(255.f,fmaxf(0.f,rintf(v)));}
__global__ void focus_breathing_kernel(const unsigned char* in,unsigned char* out,FocusBreathingConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int ch=i%c.channels,x=(i/c.channels)%c.width,y=i/(c.width*c.channels);if(c.focus_position==0.0f){out[i]=in[i];return;}float cx=c.center_x*(c.width-1),cy=c.center_y*(c.height-1),rx=fmaxf(1.0f,(c.width-1)*0.5f),ry=fmaxf(1.0f,(c.height-1)*0.5f),nx=(x-cx)/rx,ny=(y-cy)/ry,r2=nx*nx+ny*ny,scale=1.0f+c.focus_position*(c.breathing_strength+c.radial_strength*r2),sx=cx+(x-cx)/scale,sy=cy+(y-cy)/scale;if(!isfinite(scale)||scale<=0.0f||!isfinite(sx)||!isfinite(sy)){out[i]=c.fill;return;}if(c.interpolation==Interpolation::Nearest){int ix=(int)rintf(sx),iy=(int)rintf(sy);out[i]=(ix>=0&&ix<c.width&&iy>=0&&iy<c.height)?in[(iy*c.width+ix)*c.channels+ch]:c.fill;return;}int x0=(int)floorf(sx),y0=(int)floorf(sy);float ax=sx-x0,ay=sy-y0;auto at=[&](int px,int py){return (px<0||px>=c.width||py<0||py>=c.height)?(float)c.fill:(float)in[(py*c.width+px)*c.channels+ch];};float v=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));out[i]=(unsigned char)fminf(255.f,fmaxf(0.f,floorf(v+0.5f)));}
void focus_breathing_u8(const std::uint8_t* in,std::uint8_t* out,const FocusBreathingConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1||!isfinite(c.focus_position)||!isfinite(c.breathing_strength)||!isfinite(c.radial_strength)||!isfinite(c.center_x)||!isfinite(c.center_y)||c.center_x<0.0f||c.center_x>1.0f||c.center_y<0.0f||c.center_y>1.0f)throw std::invalid_argument("invalid focus breathing configuration");int n=c.width*c.height*c.channels;focus_breathing_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"focus_breathing_kernel");}
void grid_distortion_u8(const std::uint8_t* in,std::uint8_t* out,const GridDistortionConfig& c,cudaStream_t s){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.grid_width<2||c.grid_height<2||!c.displacement_x||!c.displacement_y||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1)throw std::invalid_argument("invalid grid distortion configuration");
  int n=c.width*c.height*c.channels;grid_distortion_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"grid_distortion_kernel");
}
void piecewise_affine_u8(const std::uint8_t* in,std::uint8_t* out,const PiecewiseAffineConfig& c,cudaStream_t s){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.grid_width<2||c.grid_height<2||!c.displacement_x||!c.displacement_y||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1)throw std::invalid_argument("invalid piecewise affine configuration");
  int n=c.width*c.height*c.channels;piecewise_affine_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"piecewise_affine_kernel");
}
void elastic_transform_u8(const std::uint8_t* in,std::uint8_t* out,const ElasticTransformConfig& c,cudaStream_t s){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||!c.displacement_x||!c.displacement_y||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1)
    throw std::invalid_argument("invalid elastic transform configuration");
  int n=c.width*c.height*c.channels;elastic_transform_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"elastic_transform_kernel");
}
void perspective_u8(const std::uint8_t* in,std::uint8_t* out,const PerspectiveConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid perspective configuration");float d=c.h00*c.h11*c.h22+c.h01*c.h12*c.h20+c.h02*c.h10*c.h21-c.h02*c.h11*c.h20-c.h01*c.h10*c.h22-c.h00*c.h12*c.h21;if(fabsf(d)<1e-8f)throw std::invalid_argument("singular perspective matrix");int n=c.width*c.height*c.channels;perspective_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"perspective_kernel");}
void affine_u8(const std::uint8_t* in,std::uint8_t* out,const AffineConfig& c,cudaStream_t s){if(!in||!out||c.input_width<=0||c.input_height<=0||c.output_width<=0||c.output_height<=0||c.channels<=0)throw std::invalid_argument("invalid affine configuration");if(fabsf(c.m00*c.m11-c.m01*c.m10)<1e-8f)throw std::invalid_argument("singular affine matrix");int n=c.output_width*c.output_height*c.channels;affine_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"affine_kernel");}
void safe_rotate_u8(const std::uint8_t* in,std::uint8_t* out,const SafeRotateConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid safe rotate configuration");float a=c.angle_degrees*3.14159265358979323846f/180.0f,co=cosf(a),si=sinf(a),cx=(c.width-1)*0.5f,cy=(c.height-1)*0.5f;float fit=fminf(1.0f,fminf(c.width/(fabsf(co)*c.width+fabsf(si)*c.height),c.height/(fabsf(si)*c.width+fabsf(co)*c.height)));co*=fit;si*=fit;affine_u8(in,out,{c.width,c.height,c.width,c.height,c.channels,co,si,cx-co*cx-si*cy,-si,co,cy+si*cx-co*cy,c.interpolation,c.fill},s);}
void optical_distortion_u8(const std::uint8_t* in,std::uint8_t* out,const OpticalDistortionConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid optical distortion configuration");int n=c.width*c.height*c.channels;optical_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"optical_distortion_kernel");}
void thin_prism_distortion_u8(const std::uint8_t* in,std::uint8_t* out,const ThinPrismDistortionConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1||!isfinite(c.s1)||!isfinite(c.s2)||!isfinite(c.t1)||!isfinite(c.t2))throw std::invalid_argument("invalid thin-prism distortion configuration");int n=c.width*c.height*c.channels;thin_prism_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"thin_prism_kernel");}
void rolling_shutter_geometric_distortion_u8(const std::uint8_t* in,std::uint8_t* out,const RollingShutterGeometricDistortionConfig& c,cudaStream_t s){if(!in||!out||!c.row_shift_x||!c.row_shift_y||c.width<=0||c.height<=0||c.channels<=0||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1)throw std::invalid_argument("invalid rolling-shutter geometric distortion configuration");int n=c.width*c.height*c.channels;rolling_shutter_geometric_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"rolling_shutter_geometric_kernel");}
void perspective_transform_u8(const std::uint8_t* in,std::uint8_t* out,const PerspectiveConfig& c,cudaStream_t s){perspective_u8(in,out,c,s);}
void rotate_u8(const std::uint8_t* in,std::uint8_t* out,const RotateConfig& c,cudaStream_t s){float a=c.angle_degrees*3.14159265358979323846f/180.0f,co=cosf(a),si=sinf(a),cx=(c.width-1)*0.5f,cy=(c.height-1)*0.5f;AffineConfig ac{c.width,c.height,c.width,c.height,c.channels,co,si,cx-co*cx-si*cy,-si,co,cy+si*cx-co*cy,c.interpolation,c.fill};affine_u8(in,out,ac,s);}
void shift_scale_rotate_u8(const std::uint8_t* in,std::uint8_t* out,const ShiftScaleRotateConfig& c,cudaStream_t s){float a=c.angle_degrees*3.14159265358979323846f/180.0f,co=c.scale*cosf(a),si=c.scale*sinf(a),cx=(c.width-1)*0.5f,cy=(c.height-1)*0.5f;AffineConfig ac{c.width,c.height,c.width,c.height,c.channels,co,si,cx-co*cx-si*cy+c.shift_x*c.width,-si,co,cy+si*cx-co*cy+c.shift_y*c.height,c.interpolation,c.fill};affine_u8(in,out,ac,s);}
void transpose_u8(const std::uint8_t* in,std::uint8_t* out,const TransposeConfig& c,cudaStream_t s){if(!in||!out||c.input_width<=0||c.input_height<=0||c.channels<=0)throw std::invalid_argument("invalid transpose configuration");int n=c.input_width*c.input_height*c.channels;transpose_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"transpose_kernel");}
void rotate90_u8(const std::uint8_t* in,std::uint8_t* out,const Rotate90Config& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid rotate90 configuration");int n=c.width*c.height*c.channels;rotate_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"rotate_kernel");}
void make_random_grid_shuffle_permutation(std::uint32_t* permutation,int grid_rows,int grid_cols,std::uint64_t seed){
  if(!permutation||grid_rows<=0||grid_cols<=0)throw std::invalid_argument("invalid random grid shuffle permutation dimensions");
  const std::size_t count=static_cast<std::size_t>(grid_rows)*static_cast<std::size_t>(grid_cols);
  if(count>std::numeric_limits<std::uint32_t>::max())throw std::invalid_argument("random grid shuffle grid is too large");
  for(std::size_t i=0;i<count;++i)permutation[i]=static_cast<std::uint32_t>(i);
  for(std::size_t i=count-1;i>0;--i){
    std::uint64_t value=seed+static_cast<std::uint64_t>(i)+0x9e3779b97f4a7c15ULL;
    value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
    value=(value^(value>>27))*0x94d049bb133111ebULL;
    const std::size_t j=static_cast<std::size_t>((value^(value>>31))%(i+1));
    const std::uint32_t swap=permutation[i];permutation[i]=permutation[j];permutation[j]=swap;
  }
}
void random_grid_shuffle_u8(const std::uint8_t* in,std::uint8_t* out,const RandomGridShuffleConfig& c,cudaStream_t s){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.grid_rows<=0||c.grid_cols<=0||c.grid_rows>c.height||c.grid_cols>c.width)
    throw std::invalid_argument("invalid random grid shuffle configuration");
  const std::size_t count=static_cast<std::size_t>(c.grid_rows)*static_cast<std::size_t>(c.grid_cols);
  if(count>0xffffffffULL)throw std::invalid_argument("random grid shuffle grid is too large");
  const int n=c.width*c.height*c.channels;random_grid_shuffle_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"random_grid_shuffle_kernel");
}
}
