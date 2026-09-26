#include "augmatch/color/tone.hpp"
#include "augmatch/color/arithmetic.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
namespace augmatch { namespace {
__device__ unsigned char sat(float x){return (unsigned char)fminf(255.f,fmaxf(0.f,rintf(x)));}
__host__ __device__ unsigned long long curve_splitmix64(unsigned long long value){value+=0x9e3779b97f4a7c15ULL;value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;value=(value^(value>>27))*0x94d049bb133111ebULL;return value^(value>>31);}
__host__ __device__ unsigned char curve_value(int channel,int input,unsigned long long seed){
  const int segment=input==255?3:input/64;
  const int x0=segment==0?0:segment==1?64:segment==2?128:192;
  const int x1=segment==0?64:segment==1?128:segment==2?192:255;
  int y[5]={0,0,0,0,255};
  for(int point=1;point<4;++point){
    const int offset=(int)(curve_splitmix64(seed+(unsigned long long)(channel*4+point))%65ULL)-32;
    const int candidate=point*64+offset;
    y[point]=candidate<y[point-1]?y[point-1]:(candidate>255?255:candidate);
  }
  const int distance=input-x0,span=x1-x0;
  return (unsigned char)((y[segment]*(span-distance)+y[segment+1]*distance+span/2)/span);
}
__global__ void random_tone_curve_kernel(const unsigned char* in,unsigned char* out,RandomToneCurveConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;
  const int channel=i%c.channels;const int input=in[i];
  out[i]=c.lut?c.lut[channel*256+input]:curve_value(channel,input,c.seed);
}
__device__ int clahe_reflect101(int p,int length){
  if(length<=1)return 0;
  while(p<0||p>=length)p=p<0?-p:2*length-p-2;
  return p;
}
__global__ void clahe_histogram_kernel(const unsigned char* in,unsigned char* luts,
    int width,int height,int channels,float clip_limit,int tiles_x,int tiles_y,
    int tile_width,int tile_height,int tile_area){
  // One block owns one channel/tile histogram. Shared memory keeps the complete
  // 256-bin histogram on the device; this deliberately favors parity over speed.
  __shared__ unsigned int histogram[256];
  for(int value=threadIdx.x;value<256;value+=blockDim.x)histogram[value]=0;
  __syncthreads();
  const int tile_x=static_cast<int>(blockIdx.x),tile_y=static_cast<int>(blockIdx.y);
  const int channel=static_cast<int>(blockIdx.z);
  for(int index=threadIdx.x;index<tile_area;index+=blockDim.x){
    const int local_y=index/tile_width,local_x=index-local_y*tile_width;
    const int source_y=clahe_reflect101(tile_y*tile_height+local_y,height);
    const int source_x=clahe_reflect101(tile_x*tile_width+local_x,width);
    const std::size_t source=(static_cast<std::size_t>(source_y)*width+source_x)*channels+channel;
    atomicAdd(&histogram[in[source]],1U);
  }
  __syncthreads();
  if(threadIdx.x==0){
    // This is the OpenCV CLAHE clip-limit normalization and redistribution
    // order used by clahe_u8_host, including the residual-bin stepping rule.
    const int clip=clip_limit*tile_area/256.0f<1.0f?1:static_cast<int>(clip_limit*tile_area/256.0f);
    int clipped=0;
    for(int value=0;value<256;++value){
      if(histogram[value]>static_cast<unsigned int>(clip)){
        clipped+=static_cast<int>(histogram[value])-clip;
        histogram[value]=static_cast<unsigned int>(clip);
      }
    }
    const int redist_batch=clipped/256;
    const int residual=clipped-redist_batch*256;
    for(int value=0;value<256;++value)histogram[value]+=static_cast<unsigned int>(redist_batch);
    if(residual>0){
      const int residual_step=256/residual>1?256/residual:1;
      int remaining=residual;
      for(int value=0;value<256&&remaining>0;value+=residual_step,--remaining)++histogram[value];
    }
    const std::size_t lut_offset=(static_cast<std::size_t>(channel)*tiles_y*tiles_x+
      static_cast<std::size_t>(tile_y)*tiles_x+tile_x)*256;
    int cumulative=0;
    for(int value=0;value<256;++value){
      cumulative+=static_cast<int>(histogram[value]);
      double mapped=static_cast<double>(cumulative)*255.0/tile_area;
      double rounded=rint(mapped);
      if(rounded<0.0)rounded=0.0;
      if(rounded>255.0)rounded=255.0;
      luts[lut_offset+value]=static_cast<unsigned char>(rounded);
    }
  }
}
__global__ void clahe_apply_kernel(const unsigned char* in,unsigned char* out,
    const unsigned char* luts,int width,int height,int channels,int tiles_x,int tiles_y,
    int tile_width,int tile_height){
  const std::size_t total=static_cast<std::size_t>(width)*height*channels;
  for(std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
      index<total;index+=static_cast<std::size_t>(blockDim.x)*gridDim.x){
    const int channel=static_cast<int>(index%channels);
    const std::size_t pixel=index/channels;
    const int y=static_cast<int>(pixel/width),x=static_cast<int>(pixel%width);
    const double inv_tile_width=1.0/tile_width;
    const double inv_tile_height=1.0/tile_height;
    const double tyf=y*inv_tile_height-0.5;
    const double txf=x*inv_tile_width-0.5;
    int ty1=static_cast<int>(floor(tyf)),ty2=ty1+1;
    int tx1=static_cast<int>(floor(txf)),tx2=tx1+1;
    double ya=tyf-ty1,xa=txf-tx1;
    if(ty1<0){ty1=ty2=0;ya=0.0;}
    if(ty2>=tiles_y){ty1=ty2=tiles_y-1;ya=0.0;}
    if(tx1<0){tx1=tx2=0;xa=0.0;}
    if(tx2>=tiles_x){tx1=tx2=tiles_x-1;xa=0.0;}
    const int value=in[index];
    const std::size_t channel_offset=static_cast<std::size_t>(channel)*tiles_y*tiles_x*256;
    const std::size_t lut_row_11=channel_offset+(static_cast<std::size_t>(ty1)*tiles_x+tx1)*256;
    const std::size_t lut_row_21=channel_offset+(static_cast<std::size_t>(ty1)*tiles_x+tx2)*256;
    const std::size_t lut_row_12=channel_offset+(static_cast<std::size_t>(ty2)*tiles_x+tx1)*256;
    const std::size_t lut_row_22=channel_offset+(static_cast<std::size_t>(ty2)*tiles_x+tx2)*256;
    const double top=luts[lut_row_11+value]*(1.0-xa)+luts[lut_row_21+value]*xa;
    const double bottom=luts[lut_row_12+value]*(1.0-xa)+luts[lut_row_22+value]*xa;
    double rounded=rint(top*(1.0-ya)+bottom*ya);
    if(rounded<0.0)rounded=0.0;
    if(rounded>255.0)rounded=255.0;
    out[index]=static_cast<unsigned char>(rounded);
  }
}
__global__ void sepia_kernel(const unsigned char* in,unsigned char* out,ToneConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height;if(i>=n)return;const unsigned char* p=in+i*c.channels;unsigned char* q=out+i*c.channels;q[0]=sat(.393f*p[0]+.769f*p[1]+.189f*p[2]);q[1]=sat(.349f*p[0]+.686f*p[1]+.168f*p[2]);q[2]=sat(.272f*p[0]+.534f*p[1]+.131f*p[2]);for(int ch=3;ch<c.channels;++ch)q[ch]=p[ch];}
__global__ void contrast_kernel(const unsigned char* in,unsigned char* out,ToneConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int ch=i%c.channels, pixels=c.width*c.height,lo=255,hi=0;for(int p=0;p<pixels;++p){int v=in[p*c.channels+ch];lo=min(lo,v);hi=max(hi,v);}out[i]=hi==lo?in[i]:(unsigned char)fminf(255.f,fmaxf(0.f,rintf((in[i]-lo)*255.f/(hi-lo))));}
__global__ void equalize_kernel(const unsigned char* in,unsigned char* out,ToneConfig c){int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i>=n)return;int ch=i%c.channels,pixels=c.width*c.height,hist[256]={};for(int p=0;p<pixels;++p)++hist[in[p*c.channels+ch]];int first=0;while(first<256&&hist[first]==0)++first;int cumulative=0;for(int v=0;v<=in[i];++v)cumulative+=hist[v];int base=hist[first],den=pixels-base;out[i]=den<=0?in[i]:(unsigned char)fminf(255.f,fmaxf(0.f,rintf((cumulative-base)*255.f/(float)den)));}
__device__ unsigned char contrast_round(float value){return (unsigned char)floorf(fminf(255.0f,fmaxf(0.0f,value))+0.5f);}
__global__ void sigmoid_contrast_kernel(const unsigned char* in,unsigned char* out,SigmoidContrastConfig c){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int n=c.width*c.height*c.channels;
  if(i>=n)return;
  const float x=in[i]/255.0f;
  const float y=1.0f/(1.0f+expf(c.gain*(c.cutoff-x)));
  out[i]=contrast_round(y*255.0f);
}
__global__ void log_contrast_kernel(const unsigned char* in,unsigned char* out,LogContrastConfig c,float scale,float denominator){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int n=c.width*c.height*c.channels;
  if(i>=n)return;
  const float x=in[i]/255.0f;
  const float y=log1pf(x*scale)/denominator;
  out[i]=contrast_round(y*255.0f);
}
__device__ unsigned char variation_round(float y){return (unsigned char)floorf(fminf(1.0f,fmaxf(0.0f,y))*255.0f+0.5f);}
__global__ void gamma_variation_kernel(const unsigned char* in,unsigned char* out,GammaVariationConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=variation_round(powf(in[i]/255.0f,c.gamma));}
__global__ void tone_curve_variation_kernel(const unsigned char* in,unsigned char* out,ToneCurveVariationConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n)out[i]=c.lut[(i%c.channels)*256+in[i]];}
__global__ void s_curve_kernel(const unsigned char* in,unsigned char* out,SCurveContrastVariationConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n){const float x=in[i]/255.0f;out[i]=variation_round(x+c.amount*4.0f*x*(1.0f-x)*(2.0f*x-1.0f));}}
__global__ void highlight_rolloff_kernel(const unsigned char* in,unsigned char* out,HighlightRolloffVariationConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n){const float x=in[i]/255.0f,d=fmaxf(0.0f,x-c.threshold),den=fmaxf(1e-12f,1.0f-c.threshold);const float y=x<=c.threshold?x:c.threshold+d/(1.0f+c.strength*d/den);out[i]=variation_round(y);}}
__global__ void shadow_lift_kernel(const unsigned char* in,unsigned char* out,ShadowLiftConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n){const float x=in[i]/255.0f,w=fminf(1.0f,fmaxf(0.0f,(c.threshold-x)/c.threshold));out[i]=variation_round(x+c.amount*w*(1.0f-x));}}
__global__ void shadow_crush_kernel(const unsigned char* in,unsigned char* out,ShadowCrushConfig c){const int i=blockIdx.x*blockDim.x+threadIdx.x,n=c.width*c.height*c.channels;if(i<n){const float x=in[i]/255.0f,w=fminf(1.0f,fmaxf(0.0f,(c.threshold-x)/c.threshold));out[i]=variation_round(x*(1.0f-c.amount*w));}}
void check(cudaError_t e,const char* p){if(e!=cudaSuccess)throw std::runtime_error(std::string(p)+": "+cudaGetErrorString(e));}
void valid_channels(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("tone transforms require positive image dimensions and channels");}
void valid_rgb(int w,int h,int c){if(w<=0||h<=0||c<3)throw std::invalid_argument("tone transforms require RGB dimensions");}
}
void clahe_u8(const std::uint8_t* in,std::uint8_t* out,const CLAHEConfig& c,cudaStream_t s){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.tiles_x<=0||c.tiles_y<=0)
    throw std::invalid_argument("invalid CLAHE configuration");
  if(!std::isfinite(c.clip_limit)||c.clip_limit<=0.0f)
    throw std::invalid_argument("CLAHE clip limit must be finite and positive");
  // Match the CPU/OpenCV padding contract: if either dimension needs padding,
  // both tile dimensions use the rounded-up grid size (including zero remainders).
  const bool needs_padding=c.width%c.tiles_x!=0||c.height%c.tiles_y!=0;
  const int tile_width=needs_padding?(c.width+c.tiles_x-c.width%c.tiles_x)/c.tiles_x:c.width/c.tiles_x;
  const int tile_height=needs_padding?(c.height+c.tiles_y-c.height%c.tiles_y)/c.tiles_y:c.height/c.tiles_y;
  const int tile_area=tile_width*tile_height;
  const std::size_t tile_count=static_cast<std::size_t>(c.tiles_x)*c.tiles_y;
  const std::size_t lut_bytes=tile_count*static_cast<std::size_t>(c.channels)*256;
  const std::size_t element_count=static_cast<std::size_t>(c.width)*c.height*c.channels;
  unsigned char* luts=nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&luts),lut_bytes),"CLAHE LUT allocation");
  try{
    const dim3 histogram_grid(static_cast<unsigned>(c.tiles_x),static_cast<unsigned>(c.tiles_y),static_cast<unsigned>(c.channels));
    clahe_histogram_kernel<<<histogram_grid,256,0,s>>>(in,luts,c.width,c.height,c.channels,c.clip_limit,
      c.tiles_x,c.tiles_y,tile_width,tile_height,tile_area);
    check(cudaGetLastError(),"clahe_histogram_kernel");
    const std::size_t required_blocks=(element_count+255)/256;
    const unsigned blocks=static_cast<unsigned>(required_blocks<65535?required_blocks:65535);
    clahe_apply_kernel<<<blocks,256,0,s>>>(in,out,luts,c.width,c.height,c.channels,c.tiles_x,c.tiles_y,tile_width,tile_height);
    check(cudaGetLastError(),"clahe_apply_kernel");
    // cudaFree synchronizes with the supplied stream, retaining the existing
    // CLAHE API's synchronous completion behavior without copying image data.
    const cudaError_t release_error=cudaFree(luts);
    luts=nullptr;
    check(release_error,"CLAHE LUT release");
  }catch(...){
    if(luts)cudaFree(luts);
    throw;
  }
}
void make_random_tone_curve_lut(std::uint8_t* lut,int channels,std::uint64_t seed){
  if(!lut||channels<=0)throw std::invalid_argument("random tone curve LUT requires positive channels and non-null storage");
  for(int channel=0;channel<channels;++channel)for(int input=0;input<256;++input)lut[static_cast<std::size_t>(channel)*256+input]=curve_value(channel,input,seed);
}
void random_tone_curve_u8(const std::uint8_t* in,std::uint8_t* out,const RandomToneCurveConfig& c,cudaStream_t s){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid random tone curve configuration");
  const int n=c.width*c.height*c.channels;random_tone_curve_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"random_tone_curve_kernel");
}
void all_channels_clahe_u8(const std::uint8_t* in,std::uint8_t* out,const AllChannelsCLAHEConfig& c,cudaStream_t s){clahe_u8(in,out,{c.width,c.height,c.channels,c.clip_limit,c.tiles_x,c.tiles_y},s);}
void sepia_u8(const std::uint8_t* in,std::uint8_t* out,const ToneConfig& c,cudaStream_t s){valid_rgb(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null sepia buffer");int n=c.width*c.height;sepia_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"sepia_kernel");}
void autocontrast_u8(const std::uint8_t* in,std::uint8_t* out,const ToneConfig& c,cudaStream_t s){valid_channels(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null autocontrast buffer");int n=c.width*c.height*c.channels;contrast_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"contrast_kernel");}
void equalize_u8(const std::uint8_t* in,std::uint8_t* out,const ToneConfig& c,cudaStream_t s){valid_channels(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null equalize buffer");int n=c.width*c.height*c.channels;equalize_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"equalize_kernel");}
void all_channels_histogram_equalization_u8(const std::uint8_t* in,std::uint8_t* out,const AllChannelsHistogramEqualizationConfig& c,cudaStream_t s){equalize_u8(in,out,{c.width,c.height,c.channels},s);}
void sigmoid_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const SigmoidContrastConfig& c,cudaStream_t s){
  if(c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid contrast dimensions");
  if(!in||!out||!std::isfinite(c.cutoff)||c.cutoff<0.0f||c.cutoff>1.0f||!std::isfinite(c.gain)||c.gain<0.0f)
    throw std::invalid_argument("invalid sigmoid contrast configuration");
  const int n=c.width*c.height*c.channels;
  sigmoid_contrast_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);
  check(cudaGetLastError(),"sigmoid_contrast_kernel");
}
void log_contrast_u8(const std::uint8_t* in,std::uint8_t* out,const LogContrastConfig& c,cudaStream_t s){
  if(c.width<=0||c.height<=0||c.channels<=0)throw std::invalid_argument("invalid contrast dimensions");
  if(!in||!out||!std::isfinite(c.gain)||c.gain<0.0f||!std::isfinite(c.base)||c.base<=1.0f)
    throw std::invalid_argument("invalid log contrast configuration");
  const int n=c.width*c.height*c.channels;
  const float scale=powf(c.base,c.gain)-1.0f;
  const float denominator=logf(c.base);
  log_contrast_kernel<<<(n+255)/256,256,0,s>>>(in,out,c,scale,denominator);
  check(cudaGetLastError(),"log_contrast_kernel");
}
namespace {
void variation_valid(const std::uint8_t* in,const std::uint8_t* out,int w,int h,int channels){if(!in||!out||w<=0||h<=0||channels<=0)throw std::invalid_argument("invalid tone variation dimensions or buffers");}

}
void gamma_variation_u8(const std::uint8_t* in,std::uint8_t* out,const GammaVariationConfig& c,cudaStream_t s){variation_valid(in,out,c.width,c.height,c.channels);if(!std::isfinite(c.gamma)||c.gamma<=0.0f)throw std::invalid_argument("gamma must be positive");const int n=c.width*c.height*c.channels;gamma_variation_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"gamma_variation_kernel");}
void tone_curve_variation_u8(const std::uint8_t* in,std::uint8_t* out,const ToneCurveVariationConfig& c,cudaStream_t s){variation_valid(in,out,c.width,c.height,c.channels);if(!c.lut)throw std::invalid_argument("tone curve LUT must not be null");const int n=c.width*c.height*c.channels;tone_curve_variation_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"tone_curve_variation_kernel");}
void s_curve_contrast_variation_u8(const std::uint8_t* in,std::uint8_t* out,const SCurveContrastVariationConfig& c,cudaStream_t s){variation_valid(in,out,c.width,c.height,c.channels);if(!std::isfinite(c.amount)||c.amount<-1.0f||c.amount>1.0f)throw std::invalid_argument("S-curve amount must be in [-1,1]");const int n=c.width*c.height*c.channels;s_curve_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"s_curve_kernel");}
void highlight_rolloff_variation_u8(const std::uint8_t* in,std::uint8_t* out,const HighlightRolloffVariationConfig& c,cudaStream_t s){variation_valid(in,out,c.width,c.height,c.channels);if(!std::isfinite(c.threshold)||!std::isfinite(c.strength)||c.threshold<0.0f||c.threshold>1.0f||c.strength<0.0f)throw std::invalid_argument("invalid highlight roll-off configuration");const int n=c.width*c.height*c.channels;highlight_rolloff_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"highlight_rolloff_kernel");}
void shadow_lift_u8(const std::uint8_t* in,std::uint8_t* out,const ShadowLiftConfig& c,cudaStream_t s){variation_valid(in,out,c.width,c.height,c.channels);if(!std::isfinite(c.amount)||!std::isfinite(c.threshold)||c.amount<0.0f||c.amount>1.0f||c.threshold<=0.0f||c.threshold>1.0f)throw std::invalid_argument("invalid shadow lift configuration");const int n=c.width*c.height*c.channels;shadow_lift_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"shadow_lift_kernel");}
void shadow_crush_u8(const std::uint8_t* in,std::uint8_t* out,const ShadowCrushConfig& c,cudaStream_t s){variation_valid(in,out,c.width,c.height,c.channels);if(!std::isfinite(c.amount)||!std::isfinite(c.threshold)||c.amount<0.0f||c.amount>1.0f||c.threshold<=0.0f||c.threshold>1.0f)throw std::invalid_argument("invalid shadow crush configuration");const int n=c.width*c.height*c.channels;shadow_crush_kernel<<<(n+255)/256,256,0,s>>>(in,out,c);check(cudaGetLastError(),"shadow_crush_kernel");}
void posterization_u8(const std::uint8_t* in,std::uint8_t* out,const PosterizationConfig& c,cudaStream_t s){posterize_u8(in,out,{c.width,c.height,c.channels,c.bits},s);}
void low_bit_depth_banding_u8(const std::uint8_t* in,std::uint8_t* out,const LowBitDepthBandingConfig& c,cudaStream_t s){posterization_u8(in,out,{c.width,c.height,c.channels,c.bits},s);}
}
