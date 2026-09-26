#include "augmatch/color/color.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__device__ unsigned char sat(int x){return (unsigned char)max(0,min(255,x));}
__global__ void shift_kernel(const unsigned char* in,unsigned char* out,RGBShiftConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int ch=i%c.channels,shift=ch==0?c.shift0:ch==1?c.shift1:ch==2?c.shift2:0;out[i]=sat((int)in[i]+shift);}
__global__ void shuffle_kernel(const unsigned char* in,unsigned char* out,ChannelShuffleConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int ch=i%c.channels,base=i-ch;int source=ch==0?c.order0:ch==1?c.order1:ch==2?c.order2:ch;out[i]=in[base+source];}
__device__ int round_u8(float x){if(x<=0.f)return 0;if(x>=255.f)return 255;return (int)floorf(x+0.5f);}
__global__ void uniform_color_quantization_kernel(const unsigned char* in,unsigned char* out,UniformColorQuantizationConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;
  const int ch=i%c.channels,quantized_channels=c.channels==4?3:c.channels;const float bin=256.0f/(float)c.levels;
  out[i]=c.levels==256?in[i]:(ch<quantized_channels?(unsigned char)round_u8(floorf((float)in[i]/bin)*bin+0.5f*bin):in[i]);
}
__global__ void uniform_color_quantization_to_n_bits_kernel(const unsigned char* in,unsigned char* out,UniformColorQuantizationToNBitsConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;
  const int ch=i%c.channels,quantized_channels=c.channels==4?3:c.channels;const unsigned char mask=(unsigned char)(0xffu<<(8-c.bits));
  out[i]=ch<quantized_channels?(unsigned char)(in[i]&mask):in[i];
}
void valid(int w,int h,int c);
std::uint64_t splitmix64(std::uint64_t value){value+=0x9e3779b97f4a7c15ULL;value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;value=(value^(value>>27))*0x94d049bb133111ebULL;return value^(value>>31);}
void validate_random_jitter(const RandomColorJitterConfig& c){
  valid(c.width,c.height,c.channels);
  const float values[]={c.brightness_min,c.brightness_max,c.contrast_min,c.contrast_max,c.saturation_min,c.saturation_max,c.hue_min,c.hue_max};
  for(float value:values)if(!std::isfinite(value))throw std::invalid_argument("random color jitter ranges must be finite");
  if(c.brightness_min>c.brightness_max||c.contrast_min>c.contrast_max||c.saturation_min>c.saturation_max||c.hue_min>c.hue_max||c.contrast_min<0.0f||c.saturation_min<0.0f)
    throw std::invalid_argument("invalid random color jitter range");
}
float random_jitter_value(float minimum,float maximum,std::uint64_t seed,int index){
  const float unit=static_cast<float>(splitmix64(seed+static_cast<std::uint64_t>(index))>>40)/16777216.0f;
  return minimum+(maximum-minimum)*unit;
}
__global__ void color_jitter_kernel(const unsigned char* in,unsigned char* out,ColorJitterConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;int rgb[3];for(int ch=0;ch<3;++ch)rgb[ch]=round_u8(c.contrast*(float)p[ch]+c.brightness*255.f);int red=rgb[0],green=rgb[1],blue=rgb[2],value=max(red,max(green,blue)),minimum=min(red,min(green,blue)),difference=value-minimum,hue_bin=0;if(difference>0){int numerator=value==red?green-blue+6*difference:value==green?blue-red+2*difference:red-green+4*difference;hue_bin=(numerator*30+difference/2)/difference;if(hue_bin<0)hue_bin+=180;}int hue_shift=(int)floorf(c.hue+0.5f);hue_bin=(hue_bin+hue_shift)%180;if(hue_bin<0)hue_bin+=180;int saturation_bin=value==0?0:(difference*255+value/2)/value,adjusted_saturation=round_u8(saturation_bin*c.saturation),remainder=hue_bin%30;int p_value=(value*(255-adjusted_saturation)+127)/255,q_value=(value*(255-(adjusted_saturation*remainder)/30)+127)/255,t_value=(value*(255-(adjusted_saturation*(30-remainder))/30)+127)/255;switch(hue_bin/30){case 0:q[0]=sat(value);q[1]=sat(t_value);q[2]=sat(p_value);break;case 1:q[0]=sat(q_value);q[1]=sat(value);q[2]=sat(p_value);break;case 2:q[0]=sat(p_value);q[1]=sat(value);q[2]=sat(t_value);break;case 3:q[0]=sat(p_value);q[1]=sat(q_value);q[2]=sat(value);break;case 4:q[0]=sat(t_value);q[1]=sat(p_value);q[2]=sat(value);break;default:q[0]=sat(value);q[1]=sat(p_value);q[2]=sat(q_value);break;}for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}
struct PlanckianRGB { float red,green,blue; };
__device__ PlanckianRGB planckian_rgb(float kelvin){
  const float t=kelvin/100.0f;PlanckianRGB rgb{};
  if(t<=66.0f){rgb.red=255.0f;rgb.green=99.4708025861f*logf(t)-161.1195681661f;rgb.blue=t<=19.0f?0.0f:138.5177312231f*logf(t-10.0f)-305.0447927307f;}
  else{rgb.red=329.698727446f*powf(t-60.0f,-0.1332047592f);rgb.green=288.1221695283f*powf(t-60.0f,-0.0755148492f);rgb.blue=255.0f;}
  rgb.red=fminf(255.0f,fmaxf(0.0f,rgb.red));rgb.green=fminf(255.0f,fmaxf(0.0f,rgb.green));rgb.blue=fminf(255.0f,fmaxf(0.0f,rgb.blue));return rgb;
}
__global__ void planckian_jitter_kernel(const unsigned char* in,unsigned char* out,PlanckianJitterConfig c){
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;
  const PlanckianRGB reference=planckian_rgb(6500.0f),color=planckian_rgb(c.temperature_kelvin);
  const float gains[3]={color.red/reference.red,color.green/reference.green,color.blue/reference.blue};
  const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;
  for(int ch=0;ch<3;++ch)q[ch]=sat((int)floorf((float)p[ch]*gains[ch]+0.5f));
  for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];
}
__device__ float photo_normal(std::uint64_t);
__global__ void gain_noise_kernel(const unsigned char* in,unsigned char* out,PerChannelGainNoiseConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;for(int k=0;k<3;++k)q[k]=sat((int)floorf((float)p[k]*(c.gains[k]+c.stddev*photo_normal(c.seed+(unsigned long long)i*3+k))+0.5f));for(int k=3;k<c.channels;++k)q[k]=p[k];}
__global__ void temp_error_kernel(const unsigned char* in,unsigned char* out,ColorTemperatureErrorConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;PlanckianRGB a=planckian_rgb(6500.0f),b=planckian_rgb(c.temperature_kelvin);float gain[3]={1+c.strength*(b.red/a.red-1),1+c.strength*(b.green/a.green-1),1+c.strength*(b.blue/a.blue-1)};const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;for(int k=0;k<3;++k)q[k]=sat((int)floorf((float)p[k]*gain[k]+0.5f));for(int k=3;k<c.channels;++k)q[k]=p[k];}
__device__ float chromatic_sample(const unsigned char* in,int w,int h,int c,float sx,float sy,int ch,unsigned char fill,Interpolation interpolation){
  if(!isfinite(sx)||!isfinite(sy)||fabsf(sx)>=2147483647.0f||fabsf(sy)>=2147483647.0f)return (float)fill;
  if(interpolation==Interpolation::Nearest){const int ix=(int)rintf(sx),iy=(int)rintf(sy);return ix>=0&&ix<w&&iy>=0&&iy<h?(float)in[(iy*w+ix)*c+ch]:(float)fill;}
  const int x0=(int)floorf(sx),y0=(int)floorf(sy);const float ax=sx-x0,ay=sy-y0;
  auto at=[&](int px,int py){return px>=0&&px<w&&py>=0&&py<h?(float)in[(py*w+px)*c+ch]:(float)fill;};
  return (1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1));
}
__global__ void chromatic_aberration_kernel(const unsigned char* in,unsigned char* out,ChromaticAberrationConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width;
  const float cx=(c.width-1)*0.5f,cy=(c.height-1)*0.5f,fx=fmaxf(1.0f,c.width*0.5f),fy=fmaxf(1.0f,c.height*0.5f),xn=(x-cx)/fx,yn=(y-cy)/fy,r2=xn*xn+yn*yn;
  const float radial[3]={c.radial0,c.radial1,c.radial2},gain[3]={c.gain0,c.gain1,c.gain2};const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;
  for(int ch=0;ch<3;++ch){const float scale=1.0f+radial[ch]*r2;const float sx=cx+xn*scale*fx,sy=cy+yn*scale*fy;q[ch]=sat(round_u8(chromatic_sample(in,c.width,c.height,c.channels,sx,sy,ch,c.fill,c.interpolation)*gain[ch]));}
  for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];
}
__global__ void longitudinal_chromatic_kernel(const unsigned char* in,unsigned char* out,LongitudinalChromaticAberrationConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width,radii[3]={c.radius0,c.radius1,c.radius2};const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;const float inv=1.0f/(2.0f*c.sigma*c.sigma);
  for(int ch=0;ch<3;++ch){const int r=radii[ch];if(r==0){q[ch]=p[ch];continue;}float sum=0.0f,norm=0.0f;for(int dy=-r;dy<=r;++dy)for(int dx=-r;dx<=r;++dx){const float w=expf(-(dx*dx+dy*dy)*inv);const int px=max(0,min(c.width-1,x+dx)),py=max(0,min(c.height-1,y+dy));sum+=w*in[(py*c.width+px)*c.channels+ch];norm+=w;}q[ch]=round_u8(sum/norm);}
  for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];
}
__global__ void fancy_pca_kernel(const unsigned char* in,unsigned char* out,FancyPCAConfig c){
  int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;
  float delta[3]={0.0f,0.0f,0.0f};for(int row=0;row<3;++row)for(int col=0;col<3;++col)delta[row]+=c.basis[row*3+col]*c.eigenvalues[col]*c.perturbation[col]*c.alpha;
  const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;
  for(int ch=0;ch<3;++ch)q[ch]=(unsigned char)round_u8((float)p[ch]+delta[ch]);for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];
}
__device__ std::uint64_t plasma_mix_device(std::uint64_t value){value^=value>>30;value*=0xbf58476d1ce4e5b9ULL;value^=value>>27;value*=0x94d049bb133111ebULL;return value^(value>>31);}
__device__ float plasma_value_device(int x,int y,std::uint64_t seed){const int scales[]={8,4,2,1};std::uint64_t weighted=0;for(int scale=0;scale<4;++scale){const std::uint64_t key=seed+static_cast<std::uint64_t>(x>>scale)*0x9e3779b97f4a7c15ULL+static_cast<std::uint64_t>(y>>scale)*0xd1b54a32d192ed03ULL+static_cast<std::uint64_t>(scale)*0x94d049bb133111ebULL;weighted+=((plasma_mix_device(key)>>48)&0xffffULL)*static_cast<std::uint64_t>(scales[scale]);}return static_cast<float>(weighted)/(65535.0f*15.0f);}
__global__ void plasma_contrast_kernel(const unsigned char* in,unsigned char* out,PlasmaContrastConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width;const float field=c.field?c.field[i]:plasma_value_device(x,y,c.seed);const float factor=c.contrast*(1.0f+c.plasma_strength*(2.0f*field-1.0f));const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;for(int ch=0;ch<c.channels;++ch)q[ch]=sat(round_u8(127.5f+(static_cast<float>(p[ch])-127.5f)*factor));}
__global__ void plasma_brightness_contrast_kernel(const unsigned char* in,unsigned char* out,PlasmaBrightnessContrastConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width;const float field=c.field?c.field[i]:plasma_value_device(x,y,c.seed);const float factor=c.contrast*(1.0f+c.plasma_strength*(2.0f*field-1.0f));const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;for(int ch=0;ch<c.channels;++ch)q[ch]=sat(round_u8(127.5f+(static_cast<float>(p[ch])-127.5f)*factor+c.brightness*255.0f));}
__global__ void plasma_shadow_kernel(const unsigned char* in,unsigned char* out,PlasmaShadowConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width;const float field=c.field?c.field[i]:plasma_value_device(x,y,c.seed);const float shadow=c.threshold>0.0f&&field<c.threshold?(c.threshold-field)/c.threshold:0.0f;const float factor=1.0f-c.strength*shadow;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;for(int ch=0;ch<c.channels;++ch)q[ch]=sat(round_u8((float)p[ch]*factor));}
__device__ std::uint64_t photo_mix(std::uint64_t value){value+=0x9e3779b97f4a7c15ULL;value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;value=(value^(value>>27))*0x94d049bb133111ebULL;return value^(value>>31);}
__device__ float photo_unit(std::uint64_t key){return (float)((photo_mix(key)>>40)&0xffffffULL)/16777216.0f;}
__device__ float photo_normal(std::uint64_t key){float u=fmaxf(1.0e-7f,photo_unit(key)),v=photo_unit(key+0x9e3779b97f4a7c15ULL);return sqrtf(-2.0f*logf(u))*cosf(6.2831853071795864769f*v);}
__device__ float photo_clip(float x,float lo,float hi){return fminf(hi,fmaxf(lo,isfinite(x)?x:lo));}
__device__ void photo_ycc(float r,float g,float b,float& y,float& cb,float& cr){y=.299f*r+.587f*g+.114f*b;cb=b-y;cr=r-y;}
__device__ void photo_rgb(float y,float cb,float cr,float& r,float& g,float& b){r=y+cr;b=y+cb;g=(y-.299f*r-.114f*b)/.587f;}
__global__ void photo_matrix_kernel(const unsigned char* in,unsigned char* out,ColorMatrixPerturbationConfig c,float g0,float g1,float g2,float noise,std::uint64_t seed){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;float x[3]={(float)p[0]/255.f,(float)p[1]/255.f,(float)p[2]/255.f};for(int r=0;r<3;++r){float y=c.offset[r];for(int k=0;k<3;++k)y+=c.matrix[r*3+k]*x[k];y*=r==0?g0:r==1?g1:g2;if(noise!=0)y+=noise*photo_normal(seed+(std::uint64_t)i*3ULL+(std::uint64_t)r);q[r]=(unsigned char)round_u8(photo_clip(y,c.clip_min,c.clip_max)*255.f);}for(int k=3;k<c.channels;++k)q[k]=p[k];}
__global__ void photo_spectral_kernel(const unsigned char* in,unsigned char* out,SensorSpectralResponseVariationConfig c){float m[9];for(int k=0;k<9;++k)m[k]=c.response[k]+c.response_stddev*photo_normal(c.seed+(std::uint64_t)k);int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;float x[3]={(float)p[0]/255.f,(float)p[1]/255.f,(float)p[2]/255.f};for(int r=0;r<3;++r){float y=c.offset[r];for(int k=0;k<3;++k)y+=m[r*3+k]*x[k];q[r]=(unsigned char)round_u8(photo_clip(y,c.clip_min,c.clip_max)*255.f);}for(int k=3;k<c.channels;++k)q[k]=p[k];}
__global__ void photo_clip_kernel(const unsigned char* in,unsigned char* out,ColorClippingConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;for(int k=0;k<3;++k)q[k]=(unsigned char)round_u8(photo_clip((float)p[k]/255.f,c.clip_min[k],c.clip_max[k])*255.f);for(int k=3;k<c.channels;++k)q[k]=p[k];}
__global__ void photo_wb_kernel(const unsigned char* in,unsigned char* out,WhiteBalanceClippingConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;for(int k=0;k<3;++k)q[k]=(unsigned char)round_u8(photo_clip((float)p[k]/255.f*c.gains[k],c.clip_min,c.clip_max)*255.f);for(int k=3;k<c.channels;++k)q[k]=p[k];}
__global__ void photo_noise_kernel(const unsigned char* in,unsigned char* out,int mode,ChromaNoiseConfig cc,LumaNoiseConfig lc,CorrelatedLumaChromaNoiseConfig xc){int width=mode==0?cc.width:mode==1?lc.width:xc.width,height=mode==0?cc.height:mode==1?lc.height:xc.height,channels=mode==0?cc.channels:mode==1?lc.channels:xc.channels;int i=blockIdx.x*blockDim.x+threadIdx.x,n=width*height;if(i>=n)return;int c=channels;std::uint64_t seed=mode==0?cc.seed:mode==1?lc.seed:xc.seed;float s=mode==0?cc.stddev:0,ls=mode==1?lc.stddev:xc.luma_stddev,cs=mode==2?0:xc.chroma_stddev,rho=xc.luma_chroma_correlation;const unsigned char* p=in+i*c;unsigned char* q=out+i*c;float y,cb,cr;photo_ycc((float)p[0]/255.f,(float)p[1]/255.f,(float)p[2]/255.f,y,cb,cr);if(mode==0){cb+=s*photo_normal(seed+(std::uint64_t)i*3);cr+=s*photo_normal(seed+(std::uint64_t)i*3+1);}else if(mode==1)y+=ls*photo_normal(seed+(std::uint64_t)i*3);else{float z0=photo_normal(seed+(std::uint64_t)i*3),z1=photo_normal(seed+(std::uint64_t)i*3+1),z2=photo_normal(seed+(std::uint64_t)i*3+2),root=sqrtf(fmaxf(0.f,1.f-rho*rho));y+=ls*z0;cb+=cs*(rho*z0+root*z1);cr+=cs*(rho*z0+root*z2);}float r,g,b;photo_rgb(y,cb,cr,r,g,b);q[0]=(unsigned char)round_u8(photo_clip(r,0,1)*255.f);q[1]=(unsigned char)round_u8(photo_clip(g,0,1)*255.f);q[2]=(unsigned char)round_u8(photo_clip(b,0,1)*255.f);for(int k=3;k<c;++k)q[k]=p[k];}
__global__ void chroma_subsampling_kernel(const unsigned char* in,unsigned char* out,ChromaSubsamplingArtifactsConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;const int mode=(int)c.subsampling;
  if(mode==0){for(int k=0;k<c.channels;++k)q[k]=p[k];return;}
  float yy,cb,cr;photo_ycc(p[0]/255.0f,p[1]/255.0f,p[2]/255.0f,yy,cb,cr);const int x0=(x/2)*2,y0=mode==2?(y/2)*2:y;float cb_sum=0.0f,cr_sum=0.0f;int count=0;
  for(int ny=y0;ny<c.height&&ny<=y0+(mode==2?1:0);++ny)for(int nx=x0;nx<c.width&&nx<=x0+1;++nx){float sy,scb,scr;const unsigned char* s=in+(ny*c.width+nx)*c.channels;photo_ycc(s[0]/255.0f,s[1]/255.0f,s[2]/255.0f,sy,scb,scr);cb_sum+=scb;cr_sum+=scr;++count;}
  cb=cb_sum/count;cr=cr_sum/count;float r,g,b;photo_rgb(yy,cb,cr,r,g,b);q[0]=(unsigned char)round_u8(photo_clip(r,0,1)*255.0f);q[1]=(unsigned char)round_u8(photo_clip(g,0,1)*255.0f);q[2]=(unsigned char)round_u8(photo_clip(b,0,1)*255.0f);for(int k=3;k<c.channels;++k)q[k]=p[k];
}
__global__ void local_tone_mapping_noise_kernel(const unsigned char* in,unsigned char* out,LocalToneMappingNoiseConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const int x=i%c.width,y=i/c.width;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;float yy,cb,cr;photo_ycc(p[0]/255.0f,p[1]/255.0f,p[2]/255.0f,yy,cb,cr);float mean=0.0f;int count=0;
  for(int ny=max(0,y-c.radius);ny<=min(c.height-1,y+c.radius);++ny)for(int nx=max(0,x-c.radius);nx<=min(c.width-1,x+c.radius);++nx){float sy,scb,scr;const unsigned char* s=in+(ny*c.width+nx)*c.channels;photo_ycc(s[0]/255.0f,s[1]/255.0f,s[2]/255.0f,sy,scb,scr);mean+=sy;++count;}
  yy+=c.tone_strength*(mean/count-yy)+c.noise_stddev*photo_normal(c.seed+(std::uint64_t)i);float r,g,b;photo_rgb(yy,cb,cr,r,g,b);q[0]=(unsigned char)round_u8(photo_clip(r,0,1)*255.0f);q[1]=(unsigned char)round_u8(photo_clip(g,0,1)*255.0f);q[2]=(unsigned char)round_u8(photo_clip(b,0,1)*255.0f);for(int k=3;k<c.channels;++k)q[k]=p[k];
}
__global__ void to_rgb_kernel(const unsigned char* in,unsigned char* out,ToRGBConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;if(c.input_channels==1)out[3*i]=out[3*i+1]=out[3*i+2]=in[i];else{out[3*i]=in[3*i];out[3*i+1]=in[3*i+1];out[3*i+2]=in[3*i+2];}}
void check(cudaError_t e,const char* p){if(e!=cudaSuccess)throw std::runtime_error(std::string(p)+": "+cudaGetErrorString(e));}
void valid(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid color dimensions");}
std::uint64_t plasma_mix_host(std::uint64_t value){value^=value>>30;value*=0xbf58476d1ce4e5b9ULL;value^=value>>27;value*=0x94d049bb133111ebULL;return value^(value>>31);}
float plasma_value_host(int x,int y,std::uint64_t seed){const int scales[]={8,4,2,1};std::uint64_t weighted=0;for(int scale=0;scale<4;++scale){const std::uint64_t key=seed+static_cast<std::uint64_t>(x>>scale)*0x9e3779b97f4a7c15ULL+static_cast<std::uint64_t>(y>>scale)*0xd1b54a32d192ed03ULL+static_cast<std::uint64_t>(scale)*0x94d049bb133111ebULL;weighted+=((plasma_mix_host(key)>>48)&0xffffULL)*static_cast<std::uint64_t>(scales[scale]);}return static_cast<float>(weighted)/(65535.0f*15.0f);}
}
void make_plasma_field(float* field,int width,int height,std::uint64_t seed){if(!field||width<=0||height<=0)throw std::invalid_argument("invalid plasma field dimensions or buffer");for(int y=0;y<height;++y)for(int x=0;x<width;++x)field[static_cast<std::size_t>(y)*width+x]=plasma_value_host(x,y,seed);}
void rgb_shift_u8(const std::uint8_t* in,std::uint8_t* out,const RGBShiftConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.channels<3)throw std::invalid_argument("RGB shift requires at least three channels");int n=c.width*c.height*c.channels;shift_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"shift_kernel");}
void channel_shuffle_u8(const std::uint8_t* in,std::uint8_t* out,const ChannelShuffleConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.order0<0||c.order0>=c.channels||c.order1<0||c.order1>=c.channels||c.order2<0||c.order2>=c.channels)throw std::invalid_argument("invalid channel permutation");int n=c.width*c.height*c.channels;shuffle_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"shuffle_kernel");}
void color_jitter_u8(const std::uint8_t* in,std::uint8_t* out,const ColorJitterConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.channels<3||!isfinite(c.brightness)||!isfinite(c.contrast)||!isfinite(c.saturation)||!isfinite(c.hue)||c.contrast<0.f||c.saturation<0.f)throw std::invalid_argument("invalid color jitter configuration");int n=c.width*c.height;color_jitter_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"color_jitter_kernel");}
ColorJitterConfig make_random_color_jitter_config(const RandomColorJitterConfig& c){
  validate_random_jitter(c);
  return {c.width,c.height,c.channels,
    random_jitter_value(c.brightness_min,c.brightness_max,c.seed,0),
    random_jitter_value(c.contrast_min,c.contrast_max,c.seed,1),
    random_jitter_value(c.saturation_min,c.saturation_max,c.seed,2),
    random_jitter_value(c.hue_min,c.hue_max,c.seed,3)};
}
void random_color_jitter_u8(const std::uint8_t* in,std::uint8_t* out,const RandomColorJitterConfig& c,cudaStream_t stream){
  color_jitter_u8(in,out,make_random_color_jitter_config(c),stream);
}
void plasma_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const PlasmaContrastConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!isfinite(c.contrast)||!isfinite(c.plasma_strength)||c.contrast<0.0f||c.plasma_strength<0.0f||c.plasma_strength>1.0f)throw std::invalid_argument("invalid PlasmaContrast scaling configuration");
  const int n=c.width*c.height;plasma_contrast_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"plasma_contrast_kernel");
}
void plasma_brightness_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const PlasmaBrightnessContrastConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!isfinite(c.brightness)||!isfinite(c.contrast)||!isfinite(c.plasma_strength)||c.contrast<0.0f||c.plasma_strength<0.0f||c.plasma_strength>1.0f)throw std::invalid_argument("invalid PlasmaBrightnessContrast scaling configuration");
  const int n=c.width*c.height;plasma_brightness_contrast_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"plasma_brightness_contrast_kernel");
}
void plasma_shadow_u8(const std::uint8_t* in,std::uint8_t* out,const PlasmaShadowConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||!isfinite(c.threshold)||!isfinite(c.strength)||c.threshold<0.0f||c.threshold>1.0f||c.strength<0.0f||c.strength>1.0f)throw std::invalid_argument("invalid PlasmaShadow threshold or strength");
  const int n=c.width*c.height;plasma_shadow_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"plasma_shadow_kernel");
}
void planckian_jitter_u8(const std::uint8_t* in,std::uint8_t* out,const PlanckianJitterConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.channels<3||!isfinite(c.temperature_kelvin)||c.temperature_kelvin<1000.0f||c.temperature_kelvin>40000.0f)throw std::invalid_argument("PlanckianJitter temperature must be finite and in [1000,40000] K");int n=c.width*c.height;planckian_jitter_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"planckian_jitter_kernel");}
void change_color_temperature_u8(const std::uint8_t* in,std::uint8_t* out,const ChangeColorTemperatureConfig& c,cudaStream_t s){planckian_jitter_u8(in,out,{c.width,c.height,c.channels,c.temperature_kelvin},s);}
void per_channel_gain_noise_u8(const std::uint8_t* in,std::uint8_t* out,const PerChannelGainNoiseConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.channels<3||!isfinite(c.stddev)||c.stddev<0)throw std::invalid_argument("invalid per-channel gain noise configuration");for(int k=0;k<3;++k)if(!isfinite(c.gains[k])||c.gains[k]<0)throw std::invalid_argument("invalid gain");int n=c.width*c.height;gain_noise_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"gain_noise_kernel");}
void color_temperature_error_u8(const std::uint8_t* in,std::uint8_t* out,const ColorTemperatureErrorConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.channels<3||!isfinite(c.temperature_kelvin)||c.temperature_kelvin<1000||c.temperature_kelvin>40000||!isfinite(c.strength)||c.strength<0||c.strength>1)throw std::invalid_argument("invalid color-temperature error configuration");int n=c.width*c.height;temp_error_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"color_temperature_error_kernel");}
void chromatic_aberration_u8(const std::uint8_t* in,std::uint8_t* out,const ChromaticAberrationConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.channels<3||c.interpolation==Interpolation::Area||static_cast<int>(c.interpolation)<0||static_cast<int>(c.interpolation)>1||!isfinite(c.radial0)||!isfinite(c.radial1)||!isfinite(c.radial2)||!isfinite(c.gain0)||!isfinite(c.gain1)||!isfinite(c.gain2)||c.gain0<0.0f||c.gain1<0.0f||c.gain2<0.0f)throw std::invalid_argument("invalid chromatic aberration configuration");int n=c.width*c.height;chromatic_aberration_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"chromatic_aberration_kernel");}
void optical_chromatic_aberration_u8(const std::uint8_t* in,std::uint8_t* out,const OpticalChromaticAberrationConfig& c,cudaStream_t s){chromatic_aberration_u8(in,out,{c.width,c.height,c.channels,c.radial0,c.radial1,c.radial2,c.gain0,c.gain1,c.gain2,c.interpolation,c.fill},s);}
void lateral_chromatic_aberration_u8(const std::uint8_t* in,std::uint8_t* out,const LateralChromaticAberrationConfig& c,cudaStream_t s){chromatic_aberration_u8(in,out,{c.width,c.height,c.channels,c.radial0,c.radial1,c.radial2,1.0f,1.0f,1.0f,c.interpolation,c.fill},s);}
void longitudinal_chromatic_aberration_u8(const std::uint8_t* in,std::uint8_t* out,const LongitudinalChromaticAberrationConfig& c,cudaStream_t s){valid(c.width,c.height,c.channels);if(!in||!out||c.channels<3||c.radius0<0||c.radius1<0||c.radius2<0||c.radius0>32||c.radius1>32||c.radius2>32||!isfinite(c.sigma)||c.sigma<=0.0f)throw std::invalid_argument("invalid longitudinal chromatic aberration configuration");int n=c.width*c.height;longitudinal_chromatic_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"longitudinal_chromatic_kernel");}
void fancy_pca_u8(const std::uint8_t* in,std::uint8_t* out,const FancyPCAConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||c.channels<3)throw std::invalid_argument("FancyPCA requires at least three channels");
  for(int i=0;i<9;++i)if(!isfinite(c.basis[i]))throw std::invalid_argument("FancyPCA basis must be finite");
  for(int i=0;i<3;++i)if(!isfinite(c.eigenvalues[i])||c.eigenvalues[i]<0.0f||!isfinite(c.perturbation[i]))throw std::invalid_argument("FancyPCA eigenvalues and perturbation must be finite; eigenvalues must be nonnegative");
  if(!isfinite(c.alpha))throw std::invalid_argument("FancyPCA alpha must be finite");
  for(int col=0;col<3;++col){float norm=0.0f;for(int row=0;row<3;++row)norm+=c.basis[row*3+col]*c.basis[row*3+col];if(fabsf(norm-1.0f)>1e-3f)throw std::invalid_argument("FancyPCA basis must be orthonormal");for(int other=0;other<col;++other){float dot=0.0f;for(int row=0;row<3;++row)dot+=c.basis[row*3+col]*c.basis[row*3+other];if(fabsf(dot)>1e-3f)throw std::invalid_argument("FancyPCA basis must be orthonormal");}}
  int n=c.width*c.height;fancy_pca_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"fancy_pca_kernel");
}
void uniform_color_quantization_u8(const std::uint8_t* in,std::uint8_t* out,const UniformColorQuantizationConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||c.levels<2||c.levels>256)throw std::invalid_argument("UniformColorQuantization levels must be in [2,256]");
  const int n=c.width*c.height*c.channels;uniform_color_quantization_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"uniform_color_quantization_kernel");
}
void uniform_color_quantization_to_n_bits_u8(const std::uint8_t* in,std::uint8_t* out,const UniformColorQuantizationToNBitsConfig& c,cudaStream_t s){
  valid(c.width,c.height,c.channels);
  if(!in||!out||c.bits<1||c.bits>8)throw std::invalid_argument("UniformColorQuantizationToNBits bits must be in [1,8]");
  const int n=c.width*c.height*c.channels;uniform_color_quantization_to_n_bits_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"uniform_color_quantization_to_n_bits_kernel");
}
void to_rgb_u8(const std::uint8_t* in,std::uint8_t* out,const ToRGBConfig& c,cudaStream_t s){if(!in||!out||c.width<=0||c.height<=0||(c.input_channels!=1&&c.input_channels!=3))throw std::invalid_argument("ToRGB requires one or three input channels");int n=c.width*c.height;to_rgb_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"to_rgb_kernel");}
void photo_valid(int w,int h,int ch,const void* in,const void* out){valid(w,h,ch);if(!in||!out||ch<3)throw std::invalid_argument("color photometric operations require RGB HWC buffers");}
void photo_clip_valid(float lo,float hi){if(!isfinite(lo)||!isfinite(hi)||lo>hi)throw std::invalid_argument("invalid normalized clipping range");}
void photo_matrix_valid(const ColorMatrixPerturbationConfig& c){photo_clip_valid(c.clip_min,c.clip_max);for(int i=0;i<9;++i)if(!isfinite(c.matrix[i]))throw std::invalid_argument("color matrix must be finite");for(int i=0;i<3;++i)if(!isfinite(c.offset[i]))throw std::invalid_argument("color offset must be finite");}
void photo_noise_valid(float s){if(!isfinite(s)||s<0)throw std::invalid_argument("noise standard deviation must be finite and nonnegative");}
void color_matrix_perturbation_u8(const std::uint8_t* in,std::uint8_t* out,const ColorMatrixPerturbationConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);photo_matrix_valid(c);int n=c.width*c.height;photo_matrix_kernel<<<(n+255)/256,256,0,s>>>(in,out,c,1,1,1,0,0);check(cudaGetLastError(),"photo_matrix_kernel");}
void camera_color_profile_variation_u8(const std::uint8_t* in,std::uint8_t* out,const CameraColorProfileVariationConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);ColorMatrixPerturbationConfig base;base.width=c.width;base.height=c.height;base.channels=c.channels;for(int i=0;i<9;++i)base.matrix[i]=c.matrix[i];for(int i=0;i<3;++i)base.offset[i]=c.offset[i];base.clip_min=c.clip_min;base.clip_max=c.clip_max;photo_matrix_valid(base);photo_noise_valid(c.noise_stddev);for(int i=0;i<3;++i)if(!isfinite(c.gains[i])||c.gains[i]<0)throw std::invalid_argument("profile gains must be finite and nonnegative");int n=c.width*c.height;photo_matrix_kernel<<<(n+255)/256,256,0,s>>>(in,out,base,c.gains[0],c.gains[1],c.gains[2],c.noise_stddev,c.seed);check(cudaGetLastError(),"photo_matrix_kernel");}
void rgb_channel_cross_talk_u8(const std::uint8_t* in,std::uint8_t* out,const RGBChannelCrossTalkConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);ColorMatrixPerturbationConfig base;base.width=c.width;base.height=c.height;base.channels=c.channels;for(int i=0;i<9;++i)base.matrix[i]=c.matrix[i];for(int i=0;i<3;++i)base.offset[i]=c.offset[i];base.clip_min=c.clip_min;base.clip_max=c.clip_max;photo_matrix_valid(base);int n=c.width*c.height;photo_matrix_kernel<<<(n+255)/256,256,0,s>>>(in,out,base,1,1,1,0,0);check(cudaGetLastError(),"photo_matrix_kernel");}
void sensor_spectral_response_variation_u8(const std::uint8_t* in,std::uint8_t* out,const SensorSpectralResponseVariationConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);photo_clip_valid(c.clip_min,c.clip_max);for(int i=0;i<9;++i)if(!isfinite(c.response[i]))throw std::invalid_argument("spectral response must be finite");for(int i=0;i<3;++i)if(!isfinite(c.offset[i]))throw std::invalid_argument("color offset must be finite");photo_noise_valid(c.response_stddev);int n=c.width*c.height;photo_spectral_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"photo_spectral_kernel");}
void color_clipping_u8(const std::uint8_t* in,std::uint8_t* out,const ColorClippingConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);for(int i=0;i<3;++i)photo_clip_valid(c.clip_min[i],c.clip_max[i]);int n=c.width*c.height;photo_clip_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"photo_clip_kernel");}
void white_balance_clipping_u8(const std::uint8_t* in,std::uint8_t* out,const WhiteBalanceClippingConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);photo_clip_valid(c.clip_min,c.clip_max);for(int i=0;i<3;++i)if(!isfinite(c.gains[i])||c.gains[i]<0)throw std::invalid_argument("white-balance gains must be finite and nonnegative");int n=c.width*c.height;photo_wb_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"photo_wb_kernel");}
void chroma_noise_u8(const std::uint8_t* in,std::uint8_t* out,const ChromaNoiseConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);photo_noise_valid(c.stddev);int n=c.width*c.height;photo_noise_kernel<<<(n+255)/256,256,0,s>>>(in,out,0,c,{},{});check(cudaGetLastError(),"photo_noise_kernel");}
void luma_noise_u8(const std::uint8_t* in,std::uint8_t* out,const LumaNoiseConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);photo_noise_valid(c.stddev);int n=c.width*c.height;photo_noise_kernel<<<(n+255)/256,256,0,s>>>(in,out,1,{},c,{});check(cudaGetLastError(),"photo_noise_kernel");}
void correlated_luma_chroma_noise_u8(const std::uint8_t* in,std::uint8_t* out,const CorrelatedLumaChromaNoiseConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);photo_noise_valid(c.luma_stddev);photo_noise_valid(c.chroma_stddev);if(!isfinite(c.luma_chroma_correlation)||c.luma_chroma_correlation<-1||c.luma_chroma_correlation>1)throw std::invalid_argument("luma/chroma correlation must be in [-1,1]");int n=c.width*c.height;photo_noise_kernel<<<(n+255)/256,256,0,s>>>(in,out,2,{},{},c);check(cudaGetLastError(),"photo_noise_kernel");}
void chroma_subsampling_artifacts_u8(const std::uint8_t* in,std::uint8_t* out,const ChromaSubsamplingArtifactsConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);const int mode=(int)c.subsampling;if(mode<0||mode>2)throw std::invalid_argument("chroma subsampling must be Y444, Y422, or Y420");int n=c.width*c.height;chroma_subsampling_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"chroma_subsampling_kernel");}
void local_tone_mapping_noise_u8(const std::uint8_t* in,std::uint8_t* out,const LocalToneMappingNoiseConfig& c,cudaStream_t s){photo_valid(c.width,c.height,c.channels,in,out);if(c.radius<0||c.radius>32||!isfinite(c.tone_strength)||c.tone_strength<0.0f||c.tone_strength>1.0f)throw std::invalid_argument("invalid local tone-mapping strength or radius");photo_noise_valid(c.noise_stddev);int n=c.width*c.height;local_tone_mapping_noise_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"local_tone_mapping_noise_kernel");}
}

