#include "augmatch/arithmetic.hpp"
#include "augmatch/bayer.hpp"
#include "augmatch/blend.hpp"
#include "augmatch/canny.hpp"
#include "augmatch/color.hpp"
#include "augmatch/colorspace.hpp"
#include "augmatch/convert.hpp"
#include "augmatch/dithering.hpp"
#include "augmatch/derivative.hpp"
#include "augmatch/dropout.hpp"
#include "augmatch/environmental.hpp"
#include "augmatch/filter.hpp"
#include "augmatch/geometric.hpp"
#include "augmatch/hsv.hpp"
#include "augmatch/iso_profile.hpp"
#include "augmatch/meta.hpp"
#include "augmatch/mixing.hpp"
#include "augmatch/imgcorruptlike.hpp"
#include "augmatch/isp_artifacts.hpp"
#include "augmatch/jpeg.hpp"
#include "augmatch/noise.hpp"
#include "augmatch/remaining_catalog.hpp"
#include "augmatch/pixel.hpp"
#include "augmatch/pillike.hpp"
#include "augmatch/size.hpp"
#include "augmatch/signal.hpp"
#include "augmatch/superpixels.hpp"
#include "augmatch/voronoi.hpp"
#include "augmatch/tone.hpp"
#include "augmatch/transport.hpp"
#include "augmatch/transforms.hpp"
#include "augmatch/weather.hpp"
#include "augmatch/webp.hpp"
#include "augmatch/video_codec.hpp"
#if AUGMATCH_HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if AUGMATCH_HAS_CUDA
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#endif
static std::vector<augmatch::BoxXYXY> read_target_boxes(const std::string& path){
  std::ifstream file(path); if(!file) throw std::runtime_error("failed to open target box file");
  std::vector<augmatch::BoxXYXY> boxes; augmatch::BoxXYXY box{};
  while(file>>box.x1>>box.y1>>box.x2>>box.y2) boxes.push_back(box);
  if(!file.eof()) throw std::runtime_error("invalid target box file"); return boxes;
}
static std::vector<augmatch::XYMaskInterval> parse_xy_intervals(const std::string& spec,int limit,const char* axis){
  std::vector<augmatch::XYMaskInterval> intervals;
  if(spec.empty()||spec=="-")return intervals;
  std::size_t start=0;
  while(start<spec.size()){
    const std::size_t comma=spec.find(',',start);
    const std::string token=spec.substr(start,comma==std::string::npos?std::string::npos:comma-start);
    const std::size_t colon=token.find(':');
    if(token.empty()||colon==std::string::npos||token.find(':',colon+1)!=std::string::npos)throw std::invalid_argument(std::string("invalid ")+axis+" masking interval: "+token);
    std::size_t begin_end=0,end_end=0;
    const int begin=std::stoi(token.substr(0,colon),&begin_end);
    const int end=std::stoi(token.substr(colon+1),&end_end);
    if(begin_end!=colon||end_end!=token.size()-colon-1||begin<0||begin>end||end>limit)throw std::invalid_argument(std::string(axis)+" masking interval is outside the image");
    intervals.push_back({begin,end});
    if(comma==std::string::npos)break;
    start=comma+1;
    if(start==spec.size())throw std::invalid_argument(std::string("invalid ")+axis+" masking interval list");
  }
  return intervals;
}
static std::vector<augmatch::SaltPepperRectangle> parse_salt_rectangles(const std::string& spec,int width,int height){
  std::vector<augmatch::SaltPepperRectangle> rectangles;
  if(spec.empty()||spec=="-")return rectangles;
  std::size_t start=0;
  while(start<spec.size()){
    const std::size_t comma=spec.find(',',start); const std::string token=spec.substr(start,comma==std::string::npos?std::string::npos:comma-start);
    std::vector<int> values; std::size_t part=0;
    while(part<=token.size()){
      const std::size_t colon=token.find(':',part); const std::string value=token.substr(part,colon==std::string::npos?std::string::npos:colon-part); std::size_t end=0;
      if(value.empty()) throw std::invalid_argument("invalid salt and pepper rectangle: "+token);
      values.push_back(std::stoi(value,&end)); if(end!=value.size()) throw std::invalid_argument("invalid salt and pepper rectangle: "+token);
      if(colon==std::string::npos) break; part=colon+1;
    }
    if(values.size()!=4||values[0]<0||values[1]<0||values[2]<values[0]||values[3]<values[1]||values[2]>width||values[3]>height) throw std::invalid_argument("salt and pepper rectangle is outside the image: "+token);
    rectangles.push_back({values[0],values[1],values[2],values[3]});
    if(comma==std::string::npos) break; start=comma+1; if(start==spec.size()) throw std::invalid_argument("invalid salt and pepper rectangle list");
  }
  return rectangles;
}
static augmatch::ChromaSubsampling parse_chroma_subsampling(const std::string& spec){
  if(spec=="444"||spec=="4:4:4"||spec=="0")return augmatch::ChromaSubsampling::Y444;
  if(spec=="422"||spec=="4:2:2"||spec=="1")return augmatch::ChromaSubsampling::Y422;
  if(spec=="420"||spec=="4:2:0"||spec=="2")return augmatch::ChromaSubsampling::Y420;
  throw std::invalid_argument("chroma subsampling must be 444, 422, or 420");
}
static augmatch::JpegSubsampling parse_jpeg_subsampling(const std::string& spec){
  if(spec=="444"||spec=="4:4:4"||spec=="0")return augmatch::JpegSubsampling::Y444;
  if(spec=="422"||spec=="4:2:2"||spec=="1")return augmatch::JpegSubsampling::Y422;
  if(spec=="420"||spec=="4:2:0"||spec=="2")return augmatch::JpegSubsampling::Y420;
  throw std::invalid_argument("JPEG subsampling must be 444, 422, or 420");
}
static std::vector<augmatch::CutoutRectangle> parse_cutout_rectangles(const std::string& spec,int width,int height){
  std::vector<augmatch::CutoutRectangle> rectangles;
  if(spec.empty()||spec=="-")return rectangles;
  std::size_t start=0;
  while(start<spec.size()){
    const std::size_t comma=spec.find(',',start);
    const std::string token=spec.substr(start,comma==std::string::npos?std::string::npos:comma-start);
    std::vector<int> values; std::size_t part=0;
    while(part<=token.size()){
      const std::size_t colon=token.find(':',part);
      const std::string value=token.substr(part,colon==std::string::npos?std::string::npos:colon-part);
      std::size_t end=0; if(value.empty()) throw std::invalid_argument("invalid Cutout rectangle: "+token);
      values.push_back(std::stoi(value,&end)); if(end!=value.size()) throw std::invalid_argument("invalid Cutout rectangle: "+token);
      if(colon==std::string::npos) break; part=colon+1;
    }
    if(values.size()!=4||values[0]<0||values[1]<0||values[2]<values[0]||values[3]<values[1]||values[2]>width||values[3]>height)
      throw std::invalid_argument("Cutout rectangle is outside the image: "+token);
    rectangles.push_back({values[0],values[1],values[2],values[3]});
    if(comma==std::string::npos)break;
    start=comma+1; if(start==spec.size())throw std::invalid_argument("invalid Cutout rectangle list");
  }
  return rectangles;
}
template<typename Config>
static void run_f32_cli(const char* operation,const char* input_path,const char* output_path,const Config& config,void (*filter)(const float*,float*,const Config&,cudaStream_t)){
  const std::size_t n=static_cast<std::size_t>(config.width)*config.height;std::vector<float> in(n),out(n);std::ifstream fi(input_path,std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read raw float input");
#if AUGMATCH_HAS_CUDA
  float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));filter(di,doo,config,nullptr);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
  filter(in.data(),out.data(),config,nullptr);
#endif
  std::ofstream fo(output_path,std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error(std::string("failed to write ")+operation+" output");
}
template<typename Config>
static void run_raw_rgb_cli(const char* operation,const char* input_path,const char* output_path,const Config& config,void (*filter)(const std::uint8_t*,std::uint8_t*,const Config&,cudaStream_t)){
  const std::size_t ni=static_cast<std::size_t>(config.width)*config.height,no=ni*3;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(input_path,std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(ni));if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw Bayer input");
#if AUGMATCH_HAS_CUDA
  std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));filter(di,doo,config,nullptr);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
  filter(in.data(),out.data(),config,nullptr);
#endif
  std::ofstream fo(output_path,std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(no));if(!fo)throw std::runtime_error(std::string("failed to write ")+operation+" output");
}
template<typename Config>
static void run_filter_cli(const char* operation,const char* input_path,const char* output_path,const Config& config,void (*filter)(const std::uint8_t*,std::uint8_t*,const Config&,cudaStream_t)){
  const std::size_t n=static_cast<std::size_t>(config.width)*config.height*config.channels;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(input_path,std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
  std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));filter(di,doo,config,nullptr);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
  filter(in.data(),out.data(),config,nullptr);
#endif
  std::ofstream fo(output_path,std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error(std::string("failed to write ")+operation+" output");
}
template<typename Config>
static void run_jpeg_cli(const char* operation,const char* input_path,const char* output_path,
                         const Config& config,void (*filter)(const std::uint8_t*,std::uint8_t*,const Config&,cudaStream_t)){
  const std::size_t n=static_cast<std::size_t>(config.width)*config.height*config.channels;
  std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(input_path,std::ios::binary);
  fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));
  if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
  std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));
  ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));filter(di,doo,config,nullptr);
  ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
  filter(in.data(),out.data(),config,nullptr);
#endif
  std::ofstream fo(output_path,std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));
  if(!fo)throw std::runtime_error(std::string("failed to write ")+operation+" output");
}
template<typename Config>
static void run_residual_cli(const char* operation,const char* input_path,const char* output_path,Config config,const char* map_path,void (*filter)(const std::uint8_t*,std::uint8_t*,const Config&,cudaStream_t)){
  const std::size_t n=static_cast<std::size_t>(config.width)*config.height*config.channels;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(input_path,std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
  std::vector<float> map; if(map_path){const std::size_t mn=static_cast<std::size_t>(config.width)*config.height*config.map_channels;map.resize(mn);std::ifstream fm(map_path,std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(mn*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(mn*sizeof(float)))throw std::runtime_error("failed to read residual map");config.residual_map=map.data();}
#if AUGMATCH_HAS_CUDA
  std::uint8_t *di=nullptr,*doo=nullptr;float* dm=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(config.residual_map){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));config.residual_map=dm;}filter(di,doo,config,nullptr);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dm)ck(cudaFree(dm));ck(cudaFree(di));ck(cudaFree(doo));
#else
  filter(in.data(),out.data(),config,nullptr);
#endif
  std::ofstream fo(output_path,std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error(std::string("failed to write ")+operation+" output");
}
template<typename Config>
static void run_transport_frame_cli(const char* op,const char* ip,const char* opath,const Config& c,void (*fn)(const std::uint8_t*,std::uint8_t*,const Config&,cudaStream_t)) {
  const std::size_t n=static_cast<std::size_t>(c.width)*c.height*c.channels; std::vector<std::uint8_t> in(n),out(n); std::ifstream fi(ip,std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),n); if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)) throw std::runtime_error("failed to read transport frame");
#if AUGMATCH_HAS_CUDA
  std::uint8_t *di=nullptr,*doo=nullptr; ck(cudaMalloc(&di,n)); ck(cudaMalloc(&doo,n)); ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice)); fn(di,doo,c,nullptr); ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost)); ck(cudaFree(di)); ck(cudaFree(doo));
#else
  fn(in.data(),out.data(),c,nullptr);
#endif
  std::ofstream fo(opath,std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),n); if(!fo) throw std::runtime_error(std::string("failed to write ")+op);
}
static void run_transport_bytes_cli(const char* op,const char* ip,const char* opath,const augmatch::TransportByteConfig& c,void (*fn)(const std::uint8_t*,std::uint8_t*,const augmatch::TransportByteConfig&,cudaStream_t)) {
  std::vector<std::uint8_t> in(c.bytes),out(c.bytes); std::ifstream fi(ip,std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),c.bytes); if(!fi||fi.gcount()!=static_cast<std::streamsize>(c.bytes)) throw std::runtime_error("failed to read transport bytes");
#if AUGMATCH_HAS_CUDA
  std::uint8_t *di=nullptr,*doo=nullptr; ck(cudaMalloc(&di,c.bytes)); ck(cudaMalloc(&doo,c.bytes)); ck(cudaMemcpy(di,in.data(),c.bytes,cudaMemcpyHostToDevice)); fn(di,doo,c,nullptr); ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,c.bytes,cudaMemcpyDeviceToHost)); ck(cudaFree(di)); ck(cudaFree(doo));
#else
  fn(in.data(),out.data(),c,nullptr);
#endif
  std::ofstream fo(opath,std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),c.bytes); if(!fo) throw std::runtime_error(std::string("failed to write ")+op);
}
int main(int argc,char** argv){
  const std::string isp_op=argc>1?argv[1]:"";
  // Codec-independent transport corruption CLI. Inputs and outputs are raw bytes.
  try {
    // Remaining imgaug catalog commands intentionally use the documented host-only
    // API contract, including when this CLI was built with CUDA enabled.
    if (isp_op == "remaining" && argc >= 3) {
      const std::string op=argv[2];
      auto read_bytes=[](const char* path,std::size_t n){std::vector<std::uint8_t> v(n);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(n));if(!f||f.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read remaining-catalog input");return v;};
      auto write_bytes=[](const char* path,const std::vector<std::uint8_t>& v){std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()));if(!f)throw std::runtime_error("failed to write remaining-catalog output");};
      if (op=="fda" || op=="fourier_domain_adaptation") {
        if(argc!=12) throw std::invalid_argument("remaining fda IN REF OUT W H C RW RH RADIUS");
        const int w=std::stoi(argv[6]),h=std::stoi(argv[7]),ch=std::stoi(argv[8]),rw=std::stoi(argv[9]),rh=std::stoi(argv[10]); const std::size_t n=static_cast<std::size_t>(w)*h*ch,rn=static_cast<std::size_t>(rw)*rh*ch; auto in=read_bytes(argv[3],n); auto ref=read_bytes(argv[4],rn); std::vector<std::uint8_t> out(n); augmatch::fda_u8(in.data(),out.data(),{w,h,ch,ref.data(),rw,rh,std::stof(argv[11])}); write_bytes(argv[5],out); return 0;
      }
      if(argc<9) throw std::invalid_argument("remaining OP IN OUT W H C PARAMS");
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),ch=std::stoi(argv[7]); const std::size_t n=static_cast<std::size_t>(w)*h*ch; auto in=read_bytes(argv[3],n); std::vector<std::uint8_t> out(n);
      if(op=="additive_laplace") augmatch::additive_laplace_noise_u8(in.data(),out.data(),{w,h,ch,std::stof(argv[8]),argc>9?std::stoull(argv[9]):0});
      else if(op=="additive_poisson") augmatch::additive_poisson_noise_u8(in.data(),out.data(),{w,h,ch,std::stof(argv[8]),argc>9?std::stoull(argv[9]):0});
      else if(op=="cartoon") augmatch::cartoon_u8(in.data(),out.data(),{w,h,ch,std::stoi(argv[8]),argc>9?std::stof(argv[9]):24.f,0});
      else if(op=="randaugment") augmatch::rand_augment_u8(in.data(),out.data(),{w,h,ch,std::stoi(argv[8]),argc>9?std::stof(argv[9]):.25f,argc>10?std::stoull(argv[10]):0});
      else if(op=="changecolorspace") augmatch::change_colorspace_u8(in.data(),out.data(),{w,h,ch,static_cast<augmatch::CatalogColorSpace>(std::stoi(argv[8])),static_cast<augmatch::CatalogColorSpace>(std::stoi(argv[9]))});
      else if(op=="kmeans") augmatch::kmeans_color_quantization_u8(in.data(),out.data(),{w,h,ch,std::stoi(argv[8]),argc>9?std::stoi(argv[9]):8,argc>10?std::stoull(argv[10]):0});
      else if(op=="convolve"){const float k[9]={0,-1,0,-1,5,-1,0,-1,0};augmatch::convolve_u8(in.data(),out.data(),{w,h,ch,3,3,k,1,0});}
      else if(op=="debug") augmatch::save_debug_image_every_n_batches_u8(in.data(),out.data(),{w,h,ch,std::stoi(argv[8]),std::stoi(argv[9]),argc>10?argv[10]:nullptr});
      else if(op=="polar") augmatch::with_polar_warping_u8(in.data(),out.data(),{w,h,ch,.5f,.5f,std::stof(argv[8]),argc>9?std::stof(argv[9]):1.f,0});
      else if(op=="jigsaw") augmatch::jigsaw_u8(in.data(),out.data(),{w,h,ch,std::stoi(argv[8]),std::stoi(argv[9]),nullptr,argc>10?std::stoull(argv[10]):0});
      else throw std::invalid_argument("unknown remaining catalog operation: "+op);
      write_bytes(argv[4],out); return 0;
    }
    // Mixing commands use explicit raw HWC uint8 records. Their compact CLI forms
    // are documented in README.md; all dimensions are validated by the API.
    if ((isp_op=="mixup" && argc==10) || (isp_op=="cutmix" && argc==12) || (isp_op=="mosaic" && argc==12) || (isp_op=="template_transform" && argc==10) || (isp_op=="overlay_elements" && argc==11)) {
      auto read_raw=[](const char* path,std::size_t n){std::vector<std::uint8_t> v(n);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(n));if(!f||f.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read mixing input");return v;};
      auto write_raw=[](const char* path,const std::vector<std::uint8_t>& v){std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()));if(!f)throw std::runtime_error("failed to write mixing output");};
      int w=0,h=0,ch=0; const char* out_path=nullptr; std::vector<std::uint8_t> out;
      if(isp_op=="mixup") {
        w=std::stoi(argv[5]);h=std::stoi(argv[6]);ch=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*ch;auto a=read_raw(argv[2],n),b=read_raw(argv[3],n);out.resize(n);augmatch::ImageSourceU8 sa{a.data(),w,h,ch},sb{b.data(),w,h,ch};augmatch::MixUpConfig z{w,h,ch,std::stof(argv[8]),std::stoull(argv[9]),false};
#if AUGMATCH_HAS_CUDA
        std::uint8_t *da=nullptr,*db=nullptr,*do_=nullptr;ck(cudaMalloc(&da,n));ck(cudaMalloc(&db,n));ck(cudaMalloc(&do_,n));ck(cudaMemcpy(da,a.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),n,cudaMemcpyHostToDevice));sa.data=da;sb.data=db;augmatch::mixup_u8(sa,sb,do_,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost));cudaFree(da);cudaFree(db);cudaFree(do_);
#else
        augmatch::mixup_u8(sa,sb,out.data(),z);
#endif
        write_raw(argv[4],out);return 0;
      }
      if(isp_op=="cutmix") {
        w=std::stoi(argv[5]);h=std::stoi(argv[6]);ch=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*ch;auto a=read_raw(argv[2],n),b=read_raw(argv[3],n);out.resize(n);augmatch::ImageSourceU8 sa{a.data(),w,h,ch},sb{b.data(),w,h,ch};augmatch::CutMixConfig z{w,h,ch,{std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11])},0,false,.5f};
#if AUGMATCH_HAS_CUDA
        std::uint8_t *da=nullptr,*db=nullptr,*do_=nullptr;ck(cudaMalloc(&da,n));ck(cudaMalloc(&db,n));ck(cudaMalloc(&do_,n));ck(cudaMemcpy(da,a.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),n,cudaMemcpyHostToDevice));sa.data=da;sb.data=db;augmatch::cutmix_u8(sa,sb,do_,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost));cudaFree(da);cudaFree(db);cudaFree(do_);
#else
        augmatch::cutmix_u8(sa,sb,out.data(),z);
#endif
        write_raw(argv[4],out);return 0;
      }
      if(isp_op=="mosaic") {
        w=std::stoi(argv[7]);h=std::stoi(argv[8]);ch=std::stoi(argv[9]);const std::size_t n=static_cast<std::size_t>(w)*h*ch;std::vector<std::vector<std::uint8_t>> data(4);std::vector<augmatch::ImageSourceU8> s(4);for(int i=0;i<4;++i){data[i]=read_raw(argv[2+i],n);s[i]={data[i].data(),w,h,ch};}out.resize(n);augmatch::MosaicConfig z{w,h,ch,std::stoi(argv[10]),std::stoi(argv[11]),0};
#if AUGMATCH_HAS_CUDA
        std::vector<std::uint8_t*> d(4);for(int i=0;i<4;++i){ck(cudaMalloc(&d[i],n));ck(cudaMemcpy(d[i],data[i].data(),n,cudaMemcpyHostToDevice));s[i].data=d[i];}augmatch::ImageSourceU8* ds=nullptr;ck(cudaMalloc(&ds,4*sizeof(augmatch::ImageSourceU8)));ck(cudaMemcpy(ds,s.data(),4*sizeof(augmatch::ImageSourceU8),cudaMemcpyHostToDevice));std::uint8_t* do_=nullptr;ck(cudaMalloc(&do_,n));augmatch::mosaic_u8({ds,4},do_,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost));for(auto p:d)cudaFree(p);cudaFree(ds);cudaFree(do_);
#else
        augmatch::mosaic_u8({s.data(),s.size()},out.data(),z);
#endif
        write_raw(argv[6],out);return 0;
      }
      if(isp_op=="template_transform") {
        w=std::stoi(argv[5]);h=std::stoi(argv[6]);ch=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*ch;auto a=read_raw(argv[2],n),t=read_raw(argv[3],n);out.resize(n);augmatch::ImageSourceU8 sa{a.data(),w,h,ch},st{t.data(),w,h,ch};float tw=std::stof(argv[9]);augmatch::TemplateTransformConfig z{w,h,ch,std::stof(argv[8]),&tw,1};
#if AUGMATCH_HAS_CUDA
        std::uint8_t *da=nullptr,*dt=nullptr,*do_=nullptr;ck(cudaMalloc(&da,n));ck(cudaMalloc(&dt,n));ck(cudaMalloc(&do_,n));ck(cudaMemcpy(da,a.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dt,t.data(),n,cudaMemcpyHostToDevice));sa.data=da;st.data=dt;augmatch::ImageSourceU8 ds{dt,w,h,ch};augmatch::ImageSourceU8* dts=nullptr;ck(cudaMalloc(&dts,sizeof(ds)));ck(cudaMemcpy(dts,&ds,sizeof(ds),cudaMemcpyHostToDevice));float* dw=nullptr;ck(cudaMalloc(&dw,sizeof(float)));ck(cudaMemcpy(dw,&tw,sizeof(float),cudaMemcpyHostToDevice));augmatch::template_transform_u8({da,w,h,ch},{dts,1},do_,{w,h,ch,std::stof(argv[8]),dw,1});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost));cudaFree(da);cudaFree(dt);cudaFree(do_);cudaFree(dts);cudaFree(dw);
#else
        augmatch::template_transform_u8(sa,{&st,1},out.data(),z);
#endif
        write_raw(argv[4],out);return 0;
      }
      w=std::stoi(argv[5]);h=std::stoi(argv[6]);ch=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*ch;auto a=read_raw(argv[2],n),e=read_raw(argv[3],n);out.resize(n);augmatch::ImageSourceU8 sa{a.data(),w,h,ch},se{e.data(),w,h,ch};augmatch::OverlayElementU8 element{se,std::stoi(argv[8]),std::stoi(argv[9]),std::stof(argv[10]),nullptr};augmatch::OverlayElementsConfig z{w,h,ch,&element,1};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *da=nullptr,*de=nullptr,*do_=nullptr;ck(cudaMalloc(&da,n));ck(cudaMalloc(&de,n));ck(cudaMalloc(&do_,n));ck(cudaMemcpy(da,a.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(de,e.data(),n,cudaMemcpyHostToDevice));sa.data=da;element.image.data=de;augmatch::OverlayElementU8* delem=nullptr;ck(cudaMalloc(&delem,sizeof(element)));ck(cudaMemcpy(delem,&element,sizeof(element),cudaMemcpyHostToDevice));augmatch::overlay_elements_u8(sa,do_,{w,h,ch,delem,1});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),do_,n,cudaMemcpyDeviceToHost));cudaFree(da);cudaFree(de);cudaFree(do_);cudaFree(delem);
#else
      augmatch::overlay_elements_u8(sa,out.data(),z);
#endif
      write_raw(argv[4],out);return 0;
    }
    // Metadata-only meta commands use whitespace-separated x1 y1 x2 y2 and
    // x y records. Arrays are host-owned because callbacks/target metadata do
    // not have a safe CUDA device-callback ABI.
    if (isp_op == "meta_clip" && argc == 8) {
      std::vector<augmatch::BoxXYXY> boxes; std::vector<augmatch::PointXY> points;
      { std::ifstream f(argv[2]); augmatch::BoxXYXY b{}; while (f >> b.x1 >> b.y1 >> b.x2 >> b.y2) boxes.push_back(b); if (!f.eof()) throw std::runtime_error("invalid box metadata"); }
      { std::ifstream f(argv[4]); augmatch::PointXY p{}; while (f >> p.x >> p.y) points.push_back(p); if (!f.eof()) throw std::runtime_error("invalid keypoint metadata"); }
      augmatch::MutableTargetAnnotations targets{boxes.data(), boxes.size(), points.data(), points.size()};
      augmatch::clip_cbas_to_image_planes(targets, std::stoi(argv[6]), std::stoi(argv[7]));
      { std::ofstream f(argv[3]); for (const auto& b : boxes) f << b.x1 << ' ' << b.y1 << ' ' << b.x2 << ' ' << b.y2 << '\n'; }
      { std::ofstream f(argv[5]); for (const auto& p : points) f << p.x << ' ' << p.y << '\n'; }
      return 0;
    }
    if (isp_op == "meta_remove" && argc == 9) {
      std::vector<augmatch::BoxXYXY> boxes; std::vector<augmatch::PointXY> points;
      { std::ifstream f(argv[2]); augmatch::BoxXYXY b{}; while (f >> b.x1 >> b.y1 >> b.x2 >> b.y2) boxes.push_back(b); if (!f.eof()) throw std::runtime_error("invalid box metadata"); }
      { std::ifstream f(argv[4]); augmatch::PointXY p{}; while (f >> p.x >> p.y) points.push_back(p); if (!f.eof()) throw std::runtime_error("invalid keypoint metadata"); }
      augmatch::MutableTargetAnnotations targets{boxes.data(), boxes.size(), points.data(), points.size()};
      const auto kept = augmatch::remove_cbas_by_out_of_image_fraction(targets, std::stoi(argv[6]), std::stoi(argv[7]), std::stof(argv[8]));
      { std::ofstream f(argv[3]); for (std::size_t i = 0; i < kept.box_count; ++i) { const auto& b = boxes[i]; f << b.x1 << ' ' << b.y1 << ' ' << b.x2 << ' ' << b.y2 << '\n'; } }
      { std::ofstream f(argv[5]); for (std::size_t i = 0; i < kept.keypoint_count; ++i) f << points[i].x << ' ' << points[i].y << '\n'; }
      return 0;
    }
    // imgcorruptlike commands use raw HWC uint8 files. The nested form
    // `imgcorruptlike OP ...` is accepted alongside the short OP spelling.
    const bool ic_nested=isp_op=="imgcorruptlike";
    const std::string ic_op=ic_nested?(argc>2?argv[2]:""):isp_op;
    const int ic_base=ic_nested?3:2;
    const bool ic_name=ic_op=="speckle_noise"||ic_op=="fog"||ic_op=="frost"||ic_op=="snow"||ic_op=="contrast"||ic_op=="brightness"||ic_op=="saturate"||ic_op=="pixelate";
    if(ic_name){
      const int expected=ic_op=="speckle_noise"||ic_op=="fog"||ic_op=="frost"||ic_op=="snow"?ic_base+8:ic_base+6;
      if(argc!=expected) throw std::invalid_argument("imgcorruptlike command has invalid argument count");
      const int w=std::stoi(argv[ic_base+2]),h=std::stoi(argv[ic_base+3]),ch=std::stoi(argv[ic_base+4]);
      const char* in_path=argv[ic_base],*out_path=argv[ic_base+1];
      if(ic_op=="speckle_noise") { augmatch::SpeckleNoiseConfig c{w,h,ch,std::stof(argv[ic_base+5]),std::stof(argv[ic_base+6]),std::stoull(argv[ic_base+7])}; run_filter_cli(ic_op.c_str(),in_path,out_path,c,augmatch::speckle_noise_u8); }
      else if(ic_op=="fog") { augmatch::FogConfig c{w,h,ch,std::stof(argv[ic_base+5]),std::stof(argv[ic_base+6]),nullptr,std::stoull(argv[ic_base+7])}; run_filter_cli(ic_op.c_str(),in_path,out_path,c,augmatch::fog_u8); }
      else if(ic_op=="frost") { augmatch::FrostConfig c{w,h,ch,std::stof(argv[ic_base+5]),std::stof(argv[ic_base+6]),nullptr,std::stoull(argv[ic_base+7])}; run_filter_cli(ic_op.c_str(),in_path,out_path,c,augmatch::frost_u8); }
      else if(ic_op=="snow") { augmatch::SnowConfig c{w,h,ch,std::stof(argv[ic_base+5]),std::stof(argv[ic_base+6]),nullptr,std::stoull(argv[ic_base+7])}; run_filter_cli(ic_op.c_str(),in_path,out_path,c,augmatch::snow_u8); }
      else if(ic_op=="contrast") { run_filter_cli(ic_op.c_str(),in_path,out_path,augmatch::ContrastConfig{w,h,ch,std::stof(argv[ic_base+5])},augmatch::contrast_u8); }
      else if(ic_op=="brightness") { run_filter_cli(ic_op.c_str(),in_path,out_path,augmatch::BrightnessConfig{w,h,ch,std::stof(argv[ic_base+5])},augmatch::brightness_u8); }
      else if(ic_op=="saturate") { run_filter_cli(ic_op.c_str(),in_path,out_path,augmatch::SaturateConfig{w,h,ch,std::stof(argv[ic_base+5])},augmatch::saturate_u8); }
      else { run_filter_cli(ic_op.c_str(),in_path,out_path,augmatch::PixelateConfig{w,h,ch,std::stoi(argv[ic_base+5])},augmatch::pixelate_u8); }
      return 0;
    }
    if(isp_op=="optical_defocus" && argc==9){ augmatch::OpticalDefocusConfig c{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),std::stof(argv[8])}; run_filter_cli(isp_op.c_str(),argv[2],argv[3],c,augmatch::optical_defocus_u8); return 0; }
    if(isp_op=="optical_motion_blur" && argc==10){ augmatch::OpticalMotionBlurConfig c{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9])}; run_filter_cli(isp_op.c_str(),argv[2],argv[3],c,augmatch::optical_motion_blur_u8); return 0; }
    if(isp_op=="optical_zoom_blur" && argc==10){ augmatch::OpticalZoomBlurConfig c{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stof(argv[7]),std::stof(argv[8]),std::stoi(argv[9])}; run_filter_cli(isp_op.c_str(),argv[2],argv[3],c,augmatch::optical_zoom_blur_u8); return 0; }
    if(isp_op=="optical_chromatic_aberration" && argc==14){ augmatch::OpticalChromaticAberrationConfig c{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11]),std::stof(argv[12]),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[13]))}; run_filter_cli(isp_op.c_str(),argv[2],argv[3],c,augmatch::optical_chromatic_aberration_u8); return 0; }
    if(isp_op=="lateral_chromatic_aberration" && argc==10){ augmatch::LateralChromaticAberrationConfig c{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),augmatch::Interpolation::Linear,0}; run_filter_cli(isp_op.c_str(),argv[2],argv[3],c,augmatch::lateral_chromatic_aberration_u8); return 0; }
    if(isp_op=="longitudinal_chromatic_aberration" && argc==11){ augmatch::LongitudinalChromaticAberrationConfig c{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),std::stoi(argv[9]),std::stof(argv[10])}; run_filter_cli(isp_op.c_str(),argv[2],argv[3],c,augmatch::longitudinal_chromatic_aberration_u8); return 0; }
    if(isp_op=="thin_prism_distortion" && argc==12){ augmatch::ThinPrismDistortionConfig c{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[11]))}; run_filter_cli(isp_op.c_str(),argv[2],argv[3],c,augmatch::thin_prism_distortion_u8); return 0; }
    if(isp_op=="rolling_shutter_geometric_distortion" && argc==10){
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),ch=std::stoi(argv[6]); const std::size_t n=static_cast<std::size_t>(w)*h*ch; std::vector<std::uint8_t> in(n),out(n); std::vector<float> sx(h),sy(h); std::ifstream fi(argv[2],std::ios::binary),fx(argv[7],std::ios::binary),fy(argv[8],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),n); fx.read(reinterpret_cast<char*>(sx.data()),static_cast<std::streamsize>(h*sizeof(float))); fy.read(reinterpret_cast<char*>(sy.data()),static_cast<std::streamsize>(h*sizeof(float))); if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!fx||fx.gcount()!=static_cast<std::streamsize>(h*sizeof(float))||!fy||fy.gcount()!=static_cast<std::streamsize>(h*sizeof(float))) throw std::runtime_error("failed to read rolling-shutter geometric input");
      augmatch::RollingShutterGeometricDistortionConfig c{w,h,ch,sx.data(),sy.data(),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[9]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *dx=nullptr,*dy=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMalloc(&dx,h*sizeof(float)));ck(cudaMalloc(&dy,h*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dx,sx.data(),h*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dy,sy.data(),h*sizeof(float),cudaMemcpyHostToDevice));c.row_shift_x=dx;c.row_shift_y=dy;augmatch::rolling_shutter_geometric_distortion_u8(di,doo,c);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));ck(cudaFree(dx));ck(cudaFree(dy));
#else
      augmatch::rolling_shutter_geometric_distortion_u8(in.data(),out.data(),c);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));return fo?0:1;
    }
    // BlendAlpha CLI: blend_alpha SOURCE.raw OVERLAY.raw OUT.raw WIDTH HEIGHT CHANNELS ALPHA.
    // Inputs and output are contiguous HWC uint8; the explicit overlay is never owned by the library.
    if(isp_op=="blend_alpha" && argc==9) {
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),ch=std::stoi(argv[7]); const std::size_t n=static_cast<std::size_t>(w)*h*ch;
      std::vector<std::uint8_t> source(n),overlay(n),out(n); std::ifstream fs(argv[2],std::ios::binary),fo(argv[3],std::ios::binary);
      fs.read(reinterpret_cast<char*>(source.data()),static_cast<std::streamsize>(n)); fo.read(reinterpret_cast<char*>(overlay.data()),static_cast<std::streamsize>(n));
      if(!fs||fs.gcount()!=static_cast<std::streamsize>(n)||!fo||fo.gcount()!=static_cast<std::streamsize>(n)) throw std::runtime_error("failed to read blend input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *ds=nullptr,*dv=nullptr,*doo=nullptr; ck(cudaMalloc(&ds,n));ck(cudaMalloc(&dv,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(ds,source.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,overlay.data(),n,cudaMemcpyHostToDevice));augmatch::blend_alpha_u8(ds,dv,doo,{w,h,ch,std::stof(argv[8])});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(ds));ck(cudaFree(dv));ck(cudaFree(doo));
#else
      augmatch::blend_alpha_u8(source.data(),overlay.data(),out.data(),{w,h,ch,std::stof(argv[8])});
#endif
      std::ofstream result(argv[4],std::ios::binary); result.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n)); return result?0:1;
    }
    if(isp_op=="truncated_frame" && argc==7) { augmatch::TransportByteConfig c; c.bytes=std::stoull(argv[4]); c.preserve_bytes=std::stoull(argv[5]); c.fill=static_cast<std::uint8_t>(std::stoi(argv[6])); run_transport_bytes_cli(argv[1],argv[2],argv[3],c,augmatch::truncated_frame_u8); return 0; }
    if(isp_op=="bit_flips" && argc==8) { augmatch::TransportByteConfig c; c.bytes=std::stoull(argv[4]); c.probability=std::stof(argv[5]); c.bit_mask=static_cast<std::uint32_t>(std::stoul(argv[6])); c.seed=std::stoull(argv[7]); run_transport_bytes_cli(argv[1],argv[2],argv[3],c,augmatch::bit_flips_u8); return 0; }
    if((isp_op=="block_quantization"||isp_op=="deblocking_filter_mismatch"||isp_op=="packet_loss_macroblocks") && argc>=9) { augmatch::TransportFrameConfig c; c.width=std::stoi(argv[4]); c.height=std::stoi(argv[5]); c.channels=std::stoi(argv[6]); c.block_size=std::stoi(argv[7]); if(isp_op=="block_quantization") { c.quant_step=std::stoi(argv[8]); run_transport_frame_cli(argv[1],argv[2],argv[3],c,augmatch::block_quantization_u8); } else if(isp_op=="deblocking_filter_mismatch") { c.strength=std::stof(argv[8]); run_transport_frame_cli(argv[1],argv[2],argv[3],c,augmatch::deblocking_filter_mismatch_u8); } else if(argc==11) { c.probability=std::stof(argv[8]); c.fill=static_cast<std::uint8_t>(std::stoi(argv[9])); c.seed=std::stoull(argv[10]); run_transport_frame_cli(argv[1],argv[2],argv[3],c,augmatch::packet_loss_macroblocks_u8); } else throw std::invalid_argument("packet_loss_macroblocks requires probability, fill, and seed"); return 0; }
    if(isp_op=="raw_bayer_transport_corruption" && argc==12) { augmatch::RawBayerTransportCorruptionConfig c; c.width=std::stoi(argv[4]); c.height=std::stoi(argv[5]); c.stride=std::stoi(argv[6]); c.bit_depth=std::stoi(argv[7]); c.probability=std::stof(argv[8]); c.fill=static_cast<std::uint16_t>(std::stoul(argv[9])); c.xor_mask=static_cast<std::uint16_t>(std::stoul(argv[10])); c.seed=std::stoull(argv[11]); const std::size_t n=static_cast<std::size_t>(c.stride)*c.height; std::vector<std::uint16_t> in(n),out(n); std::ifstream fi(argv[2],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(std::uint16_t))); if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(std::uint16_t))) throw std::runtime_error("failed to read raw Bayer plane");
#if AUGMATCH_HAS_CUDA
      std::uint16_t *di=nullptr,*doo=nullptr; ck(cudaMalloc(&di,n*sizeof(std::uint16_t))); ck(cudaMalloc(&doo,n*sizeof(std::uint16_t))); ck(cudaMemcpy(di,in.data(),n*sizeof(std::uint16_t),cudaMemcpyHostToDevice)); augmatch::raw_bayer_transport_corruption_u16(di,doo,c,nullptr); ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,n*sizeof(std::uint16_t),cudaMemcpyDeviceToHost)); ck(cudaFree(di)); ck(cudaFree(doo));
#else
      augmatch::raw_bayer_transport_corruption_u16(in.data(),out.data(),c,nullptr);
#endif
      std::ofstream fo(argv[3],std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(std::uint16_t))); return 0; }
    if(isp_op=="chroma_plane_misalignment" && argc==10) { const int w=std::stoi(argv[4]),h=std::stoi(argv[5]); augmatch::TransportPlaneConfig c{w,h,std::stoi(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),static_cast<std::uint8_t>(std::stoi(argv[9]))}; const std::size_t n=static_cast<std::size_t>(c.stride)*h; std::vector<std::uint8_t> in(n),out(n); std::ifstream fi(argv[2],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),n); if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)) throw std::runtime_error("failed to read transport plane");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr; ck(cudaMalloc(&di,n)); ck(cudaMalloc(&doo,n)); ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice)); augmatch::chroma_plane_misalignment_u8(di,doo,c,nullptr); ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost)); ck(cudaFree(di)); ck(cudaFree(doo));
#else
      augmatch::chroma_plane_misalignment_u8(in.data(),out.data(),c,nullptr);
#endif
      std::ofstream fo(argv[3],std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),n); return 0; }
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
  // Color/photometric operations consume raw interleaved HWC uint8 RGB(A).
  if(argc>=7 && (isp_op=="color_matrix_perturbation"||isp_op=="rgb_channel_cross_talk"||isp_op=="camera_color_profile_variation"||isp_op=="sensor_spectral_response_variation"||isp_op=="color_clipping"||isp_op=="white_balance_clipping"||isp_op=="chroma_noise"||isp_op=="luma_noise"||isp_op=="correlated_luma_chroma_noise"||isp_op=="chroma_subsampling_artifacts"||isp_op=="local_tone_mapping_noise"||isp_op=="per_channel_gain_noise"||isp_op=="color_temperature_error")) {
    try {
      using namespace augmatch;
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),ch=std::stoi(argv[6]);
      auto f=[&](const char* name){return isp_op==name;};
      if(f("color_matrix_perturbation") && (argc==16||argc==19)){ ColorMatrixPerturbationConfig c{w,h,ch}; for(int i=0;i<9;++i)c.matrix[i]=std::stof(argv[7+i]); if(argc==19)for(int i=0;i<3;++i)c.offset[i]=std::stof(argv[16+i]); run_filter_cli(argv[1],argv[2],argv[3],c,color_matrix_perturbation_u8); return 0; }
      if(f("rgb_channel_cross_talk") && argc==16){ RGBChannelCrossTalkConfig c; c.width=w;c.height=h;c.channels=ch;for(int i=0;i<9;++i)c.matrix[i]=std::stof(argv[7+i]);run_filter_cli(argv[1],argv[2],argv[3],c,rgb_channel_cross_talk_u8);return 0; }
      if(f("camera_color_profile_variation") && argc==11){ CameraColorProfileVariationConfig c{w,h,ch};c.gains[0]=std::stof(argv[7]);c.gains[1]=std::stof(argv[8]);c.gains[2]=std::stof(argv[9]);c.noise_stddev=std::stof(argv[10]);run_filter_cli(argv[1],argv[2],argv[3],c,camera_color_profile_variation_u8);return 0; }
      if(f("sensor_spectral_response_variation") && argc==9){ SensorSpectralResponseVariationConfig c{w,h,ch};c.response_stddev=std::stof(argv[7]);c.seed=std::stoull(argv[8]);run_filter_cli(argv[1],argv[2],argv[3],c,sensor_spectral_response_variation_u8);return 0; }
      if(f("color_clipping") && argc==9){ ColorClippingConfig c{w,h,ch};for(int i=0;i<3;++i){c.clip_min[i]=std::stof(argv[7]);c.clip_max[i]=std::stof(argv[8]);}run_filter_cli(argv[1],argv[2],argv[3],c,color_clipping_u8);return 0; }
      if(f("white_balance_clipping") && argc==12){ WhiteBalanceClippingConfig c{w,h,ch};for(int i=0;i<3;++i)c.gains[i]=std::stof(argv[7+i]);c.clip_min=std::stof(argv[10]);c.clip_max=std::stof(argv[11]);run_filter_cli(argv[1],argv[2],argv[3],c,white_balance_clipping_u8);return 0; }
      if((f("chroma_noise")||f("luma_noise")) && argc==9){ if(f("chroma_noise")){ChromaNoiseConfig c{w,h,ch,std::stof(argv[7]),std::stoull(argv[8])};run_filter_cli(argv[1],argv[2],argv[3],c,chroma_noise_u8);}else{LumaNoiseConfig c{w,h,ch,std::stof(argv[7]),std::stoull(argv[8])};run_filter_cli(argv[1],argv[2],argv[3],c,luma_noise_u8);}return 0; }
      if(f("correlated_luma_chroma_noise") && argc==11){ CorrelatedLumaChromaNoiseConfig c{w,h,ch,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stoull(argv[10])};run_filter_cli(argv[1],argv[2],argv[3],c,correlated_luma_chroma_noise_u8);return 0; }
      if(f("chroma_subsampling_artifacts") && argc==8){ ChromaSubsamplingArtifactsConfig c{w,h,ch,parse_chroma_subsampling(argv[7])};run_filter_cli(argv[1],argv[2],argv[3],c,chroma_subsampling_artifacts_u8);return 0; }
      if(f("local_tone_mapping_noise") && argc==11){ LocalToneMappingNoiseConfig c{w,h,ch,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stoull(argv[10])};run_filter_cli(argv[1],argv[2],argv[3],c,local_tone_mapping_noise_u8);return 0; }
      if(f("per_channel_gain_noise") && argc==12){ PerChannelGainNoiseConfig c{w,h,ch,{std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9])},std::stof(argv[10]),std::stoull(argv[11])};run_filter_cli(argv[1],argv[2],argv[3],c,per_channel_gain_noise_u8);return 0; }
      if(f("color_temperature_error") && argc==9){ ColorTemperatureErrorConfig c{w,h,ch,std::stof(argv[7]),std::stof(argv[8])};run_filter_cli(argv[1],argv[2],argv[3],c,color_temperature_error_u8);return 0; }
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if ((isp_op=="laplacian_residual_injection"||isp_op=="high_pass_residual_injection"||isp_op=="highpass_residual_injection") && (argc==8||argc==9)) {
    try { const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),ch=std::stoi(argv[6]); const char* map=argc==9?argv[8]:nullptr;
      if(isp_op=="laplacian_residual_injection"){augmatch::LaplacianResidualInjectionConfig c{w,h,ch,nullptr,map?ch:1,std::stof(argv[7])};run_residual_cli(argv[1],argv[2],argv[3],c,map,augmatch::laplacian_residual_injection_u8);}
      else {augmatch::HighPassResidualInjectionConfig c{w,h,ch,nullptr,map?ch:1,std::stof(argv[7])};run_residual_cli(argv[1],argv[2],argv[3],c,map,augmatch::high_pass_residual_injection_u8);}
      return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if ((isp_op=="sobel_residual_injection") && (argc==9||argc==10)) {
    try { const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),ch=std::stoi(argv[6]); const char* map=argc==10?argv[9]:nullptr; augmatch::SobelResidualInjectionConfig c{w,h,ch,nullptr,map?ch:1,std::stof(argv[7]),std::stoi(argv[8])};run_residual_cli(argv[1],argv[2],argv[3],c,map,augmatch::sobel_residual_injection_u8);return 0; }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if ((argc==8&&(isp_op=="edge_oversharpening"||isp_op=="laplacian_halos")) ||
      (argc==10&&(isp_op=="unsharp_mask_halos"||isp_op=="local_sharpening_noise_amplification"||isp_op=="local_contrast_enhancement_artifacts")) ||
      (argc==10&&isp_op=="ringing_near_strong_edges") ||
      (argc==11&&isp_op=="haloing_from_tone_mapping") ||
      (argc==9&&isp_op=="high_frequency_attenuation") ||
      (argc==11&&isp_op=="detail_smearing") ||
      (argc==12&&isp_op=="overshoot_and_undershoot") ||
      (argc==11&&(isp_op=="gradient_plus_laplacian_noise"||isp_op=="edge_dependent_compression_error")) ||
      (argc==10&&isp_op=="deblocking_halos") ||
      (argc==9&&(isp_op=="demosaicing_edge_artifacts"||isp_op=="edge_dependent_quantization"||isp_op=="gradient_reversal")) ||
      (argc==10&&isp_op=="clipped_edge_ringing")) {
    try {
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),ch=std::stoi(argv[6]);
      if(isp_op=="edge_oversharpening") run_filter_cli(argv[1],argv[2],argv[3],augmatch::EdgeOversharpeningConfig{w,h,ch,std::stof(argv[7])},augmatch::edge_oversharpening_u8);
      else if(isp_op=="laplacian_halos") run_filter_cli(argv[1],argv[2],argv[3],augmatch::LaplacianHalosConfig{w,h,ch,std::stof(argv[7])},augmatch::laplacian_halos_u8);
      else if(isp_op=="unsharp_mask_halos") run_filter_cli(argv[1],argv[2],argv[3],augmatch::UnsharpMaskHalosConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9])},augmatch::unsharp_mask_halos_u8);
      else if(isp_op=="local_sharpening_noise_amplification") run_filter_cli(argv[1],argv[2],argv[3],augmatch::LocalSharpeningNoiseAmplificationConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9])},augmatch::local_sharpening_noise_amplification_u8);
      else if(isp_op=="local_contrast_enhancement_artifacts") run_filter_cli(argv[1],argv[2],argv[3],augmatch::LocalContrastEnhancementArtifactsConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9])},augmatch::local_contrast_enhancement_artifacts_u8);
      else if(isp_op=="ringing_near_strong_edges") run_filter_cli(argv[1],argv[2],argv[3],augmatch::RingingNearStrongEdgesConfig{w,h,ch,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9])},augmatch::ringing_near_strong_edges_u8);
      else if(isp_op=="haloing_from_tone_mapping") run_filter_cli(argv[1],argv[2],argv[3],augmatch::HaloingFromToneMappingConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10])},augmatch::haloing_from_tone_mapping_u8);
      else if(isp_op=="high_frequency_attenuation") run_filter_cli(argv[1],argv[2],argv[3],augmatch::HighFrequencyAttenuationConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8])},augmatch::high_frequency_attenuation_u8);
      else if(isp_op=="detail_smearing") run_filter_cli(argv[1],argv[2],argv[3],augmatch::DetailSmearingConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stoi(argv[10])!=0},augmatch::detail_smearing_u8);
      else if(isp_op=="gradient_plus_laplacian_noise") run_filter_cli(argv[1],argv[2],argv[3],augmatch::GradientPlusLaplacianNoiseConfig{w,h,ch,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stoull(argv[10])},augmatch::gradient_plus_laplacian_noise_u8);
      else if(isp_op=="deblocking_halos") run_filter_cli(argv[1],argv[2],argv[3],augmatch::DeblockingHalosConfig{w,h,ch,std::stoi(argv[7]),std::stoi(argv[8]),std::stof(argv[9])},augmatch::deblocking_halos_u8);
      else if(isp_op=="demosaicing_edge_artifacts") run_filter_cli(argv[1],argv[2],argv[3],augmatch::DemosaicingEdgeArtifactsConfig{w,h,ch,std::stof(argv[7]),std::stof(argv[8])},augmatch::demosaicing_edge_artifacts_u8);
      else if(isp_op=="edge_dependent_quantization") run_filter_cli(argv[1],argv[2],argv[3],augmatch::EdgeDependentQuantizationConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8])},augmatch::edge_dependent_quantization_u8);
      else if(isp_op=="edge_dependent_compression_error") run_filter_cli(argv[1],argv[2],argv[3],augmatch::EdgeDependentCompressionErrorConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stoull(argv[10])},augmatch::edge_dependent_compression_error_u8);
      else if(isp_op=="gradient_reversal") run_filter_cli(argv[1],argv[2],argv[3],augmatch::GradientReversalConfig{w,h,ch,std::stof(argv[7]),std::stof(argv[8])},augmatch::gradient_reversal_u8);
      else if(isp_op=="clipped_edge_ringing") run_filter_cli(argv[1],argv[2],argv[3],augmatch::ClippedEdgeRingingConfig{w,h,ch,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9])},augmatch::clipped_edge_ringing_u8);
      else run_filter_cli(argv[1],argv[2],argv[3],augmatch::OvershootAndUndershootConfig{w,h,ch,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11])},augmatch::overshoot_and_undershoot_u8);
      return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
  }
  if (argc==10 && (std::string(argv[1])=="cfa_channel_response" || std::string(argv[1])=="bayer_plane_noise" || std::string(argv[1])=="bayer_plane_gain")) {
    try { const int w=std::stoi(argv[4]),h=std::stoi(argv[5]); const auto map=augmatch::bayer_plane_map(augmatch::BayerPattern::RGGB); const std::string op=argv[1];
      if(op=="cfa_channel_response"){augmatch::CfaChannelResponseVariationConfig c; c.width=w;c.height=h;c.plane_map=map;for(int p=0;p<4;++p)c.response[p]=std::stof(argv[6+p]);run_f32_cli(op.c_str(),argv[2],argv[3],c,augmatch::cfa_channel_response_variation_f32);}
      else if(op=="bayer_plane_noise"){augmatch::BayerPlaneNoiseConfig c; c.width=w;c.height=h;c.plane_map=map;for(int p=0;p<4;++p)c.stddev[p]=std::stof(argv[6+p]);run_f32_cli(op.c_str(),argv[2],argv[3],c,augmatch::bayer_plane_noise_f32);}
      else {augmatch::BayerPlaneGainConfig c; c.width=w;c.height=h;c.plane_map=map;for(int p=0;p<4;++p)c.gain[p]=std::stof(argv[6+p]);run_f32_cli(op.c_str(),argv[2],argv[3],c,augmatch::bayer_plane_gain_f32);} return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if (argc==14 && std::string(argv[1])=="cfa_misregistration") {
    try { augmatch::CfaMisregistrationConfig c; c.width=std::stoi(argv[4]);c.height=std::stoi(argv[5]);c.plane_map=augmatch::bayer_plane_map(augmatch::BayerPattern::RGGB);for(int p=0;p<4;++p)c.dx[p]=std::stoi(argv[6+p]);for(int p=0;p<4;++p)c.dy[p]=std::stoi(argv[10+p]);run_f32_cli("cfa_misregistration",argv[2],argv[3],c,augmatch::cfa_misregistration_f32);return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if (argc==22 && std::string(argv[1])=="cfa_leakage") {
    try { augmatch::CfaLeakageConfig c; c.width=std::stoi(argv[4]);c.height=std::stoi(argv[5]);c.plane_map=augmatch::bayer_plane_map(augmatch::BayerPattern::RGGB);for(int i=0;i<16;++i)c.leakage[i]=std::stof(argv[6+i]);run_f32_cli("cfa_leakage",argv[2],argv[3],c,augmatch::cfa_leakage_f32);return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if (argc==10 && std::string(argv[1])=="cfa_missing_samples") {
    try { augmatch::CfaMissingSamplesConfig c; c.width=std::stoi(argv[4]);c.height=std::stoi(argv[5]);c.plane_map=augmatch::bayer_plane_map(augmatch::BayerPattern::RGGB);c.missing_probability=std::stof(argv[6]);c.replacement=static_cast<augmatch::CfaMissingReplacement>(std::stoi(argv[7]));c.replacement_value=std::stof(argv[8]);c.seed=std::stoull(argv[9]);run_f32_cli("cfa_missing_samples",argv[2],argv[3],c,augmatch::cfa_missing_samples_f32);return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==8&&(std::string(argv[1])=="enhance_color"||std::string(argv[1])=="enhance_contrast"||std::string(argv[1])=="enhance_brightness"||std::string(argv[1])=="enhance_sharpness")){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const float factor=std::stof(argv[7]);const std::string op=argv[1];
      if(op=="enhance_color")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::EnhanceColorConfig{w,h,c,factor},augmatch::enhance_color_u8);
      else if(op=="enhance_contrast")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::EnhanceContrastConfig{w,h,c,factor},augmatch::enhance_contrast_u8);
      else if(op=="enhance_brightness")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::EnhanceBrightnessConfig{w,h,c,factor},augmatch::enhance_brightness_u8);
      else run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::EnhanceSharpnessConfig{w,h,c,factor},augmatch::enhance_sharpness_u8);
      return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==7&&(std::string(argv[1])=="filter_blur"||std::string(argv[1])=="filter_smooth"||std::string(argv[1])=="filter_smooth_more"||std::string(argv[1])=="filter_edge_enhance"||std::string(argv[1])=="filter_edge_enhance_more"||std::string(argv[1])=="filter_find_edges"||std::string(argv[1])=="filter_contour"||std::string(argv[1])=="filter_emboss"||std::string(argv[1])=="filter_sharpen"||std::string(argv[1])=="filter_detail")){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::string op=argv[1];
      if(op=="filter_blur")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterBlurConfig{w,h,c},augmatch::filter_blur_u8);
      else if(op=="filter_smooth")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterSmoothConfig{w,h,c},augmatch::filter_smooth_u8);
      else if(op=="filter_smooth_more")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterSmoothMoreConfig{w,h,c},augmatch::filter_smooth_more_u8);
      else if(op=="filter_edge_enhance")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterEdgeEnhanceConfig{w,h,c},augmatch::filter_edge_enhance_u8);
      else if(op=="filter_edge_enhance_more")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterEdgeEnhanceMoreConfig{w,h,c},augmatch::filter_edge_enhance_more_u8);
      else if(op=="filter_find_edges")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterFindEdgesConfig{w,h,c},augmatch::filter_find_edges_u8);
      else if(op=="filter_contour")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterContourConfig{w,h,c},augmatch::filter_contour_u8);
      else if(op=="filter_emboss")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterEmbossConfig{w,h,c},augmatch::filter_emboss_u8);
      else if(op=="filter_sharpen")run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterSharpenConfig{w,h,c},augmatch::filter_sharpen_u8);
      else run_filter_cli(op.c_str(),argv[2],argv[3],augmatch::FilterDetailConfig{w,h,c},augmatch::filter_detail_u8);
      return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="emboss"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);run_filter_cli("emboss",argv[2],argv[3],augmatch::EmbossConfig{w,h,c,std::stof(argv[7]),std::stof(argv[8])},augmatch::emboss_u8);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==8&&std::string(argv[1])=="edge_detect"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);run_filter_cli("edge_detect",argv[2],argv[3],augmatch::EdgeDetectConfig{w,h,c,std::stof(argv[7])},augmatch::edge_detect_u8);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="canny"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const auto border=static_cast<augmatch::BorderPolicy>(std::stoi(argv[10]));run_filter_cli("canny",argv[2],argv[3],augmatch::CannyConfig{w,h,c,std::stof(argv[7]),std::stof(argv[8]),std::stoi(argv[9]),border,static_cast<std::uint8_t>(std::stoi(argv[11]))},augmatch::canny_u8);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="directed_edge_detect"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);run_filter_cli("directed_edge_detect",argv[2],argv[3],augmatch::DirectedEdgeDetectConfig{w,h,c,std::stof(argv[7]),std::stof(argv[8])},augmatch::directed_edge_detect_u8);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==8&&std::string(argv[1])=="derivative"){
    try{const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::DerivativeConfig cfg{w,h,c,1.0f};augmatch::LocalStatsConfig lcfg{w,h,c,1,1.0f,augmatch::BorderPolicy::Clamp,0};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(op=="variance")augmatch::local_variance_u8(di,doo,lcfg);else if(op=="entropy")augmatch::local_entropy_u8(di,doo,lcfg);else if(op=="sobel_x")augmatch::sobel_x_u8(di,doo,cfg);else if(op=="sobel_y")augmatch::sobel_y_u8(di,doo,cfg);else if(op=="scharr_x")augmatch::scharr_x_u8(di,doo,cfg);else if(op=="scharr_y")augmatch::scharr_y_u8(di,doo,cfg);else if(op=="laplacian")augmatch::laplacian_u8(di,doo,cfg);else if(op=="magnitude")augmatch::gradient_magnitude_u8(di,doo,cfg);else throw std::invalid_argument("unknown derivative operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="variance")augmatch::local_variance_u8(in.data(),out.data(),lcfg);else if(op=="entropy")augmatch::local_entropy_u8(in.data(),out.data(),lcfg);else if(op=="sobel_x")augmatch::sobel_x_u8(in.data(),out.data(),cfg);else if(op=="sobel_y")augmatch::sobel_y_u8(in.data(),out.data(),cfg);else if(op=="scharr_x")augmatch::scharr_x_u8(in.data(),out.data(),cfg);else if(op=="scharr_y")augmatch::scharr_y_u8(in.data(),out.data(),cfg);else if(op=="laplacian")augmatch::laplacian_u8(in.data(),out.data(),cfg);else if(op=="magnitude")augmatch::gradient_magnitude_u8(in.data(),out.data(),cfg);else throw std::invalid_argument("unknown derivative operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==9||argc==8||argc==10)&& (std::string(argv[1])=="demosaic_directional"||std::string(argv[1])=="demosaic_zipper"||std::string(argv[1])=="demosaic_aliasing"||std::string(argv[1])=="demosaic_ringing"||std::string(argv[1])=="demosaic_noise_amplification")){
    try { const std::string op=argv[1]; const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),pattern=std::stoi(argv[6]);
      const auto p=static_cast<augmatch::BayerPattern>(pattern);
      if(op=="demosaic_directional"&&argc==9){augmatch::DirectionalDemosaicingArtifactsConfig c{w,h,p,std::stoi(argv[7]),std::stof(argv[8])};run_raw_rgb_cli(op.c_str(),argv[2],argv[3],c,augmatch::bayer_directional_demosaicing_artifacts_u8);}
      else if(op=="demosaic_zipper"&&argc==8){augmatch::FalseColorZipperArtifactsConfig c{w,h,p,std::stof(argv[7])};run_raw_rgb_cli(op.c_str(),argv[2],argv[3],c,augmatch::bayer_false_color_zipper_artifacts_u8);}
      else if(op=="demosaic_aliasing"&&argc==10){augmatch::DemosaicingAliasingConfig c{w,h,p,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9])};run_raw_rgb_cli(op.c_str(),argv[2],argv[3],c,augmatch::bayer_demosaicing_aliasing_u8);}
      else if(op=="demosaic_ringing"&&argc==8){augmatch::DemosaicingRingingConfig c{w,h,p,std::stof(argv[7])};run_raw_rgb_cli(op.c_str(),argv[2],argv[3],c,augmatch::bayer_demosaicing_ringing_u8);}
      else if(op=="demosaic_noise_amplification"&&argc==10){augmatch::DemosaicingNoiseAmplificationConfig c{w,h,p,std::stof(argv[7]),std::stof(argv[8]),std::stoull(argv[9])};run_raw_rgb_cli(op.c_str(),argv[2],argv[3],c,augmatch::bayer_demosaicing_noise_amplification_u8);}
      else throw std::invalid_argument("invalid demosaicing artifact CLI arguments");
      return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==7&&(std::string(argv[1])=="demosaic"||std::string(argv[1])=="demosaic_bilinear"||std::string(argv[1])=="demosaic_malvar"||std::string(argv[1])=="demosaic_malvar_he_cutler"||std::string(argv[1])=="demosaic_edge_aware"||std::string(argv[1])=="demosaic_edgeaware") ){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),pattern=std::stoi(argv[6]);std::size_t ni=static_cast<std::size_t>(w)*h,no=ni*3;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");augmatch::BayerConfig cfg{w,h,static_cast<augmatch::BayerPattern>(pattern)};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));const std::string op=argv[1];if(op=="demosaic_bilinear")augmatch::bayer_demosaic_bilinear_u8(di,doo,cfg);else if(op=="demosaic_malvar"||op=="demosaic_malvar_he_cutler")augmatch::bayer_demosaic_malvar_he_cutler_u8(di,doo,cfg);else if(op=="demosaic_edge_aware"||op=="demosaic_edgeaware")augmatch::bayer_demosaic_edge_aware_u8(di,doo,cfg);else augmatch::bayer_demosaic_nearest_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      const std::string op=argv[1];if(op=="demosaic_bilinear")augmatch::bayer_demosaic_bilinear_u8(in.data(),out.data(),cfg);else if(op=="demosaic_malvar"||op=="demosaic_malvar_he_cutler")augmatch::bayer_demosaic_malvar_he_cutler_u8(in.data(),out.data(),cfg);else if(op=="demosaic_edge_aware"||op=="demosaic_edgeaware")augmatch::bayer_demosaic_edge_aware_u8(in.data(),out.data(),cfg);else augmatch::bayer_demosaic_nearest_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="channel_gain"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::ChannelGainConfig cfg{w,h,c,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9])};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::channel_gain_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::channel_gain_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="scale"){
    try{int iw=std::stoi(argv[4]),ih=std::stoi(argv[5]),ch=std::stoi(argv[6]),ow=std::stoi(argv[7]),oh=std::stoi(argv[8]);float factor=std::stof(argv[9]);std::size_t ni=static_cast<std::size_t>(iw)*ih*ch,no=static_cast<std::size_t>(ow)*oh*ch;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");augmatch::ScaleConfig cfg{iw,ih,ow,oh,ch,factor,augmatch::Interpolation::Linear};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::random_scale_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_scale_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc>=7&&argc<=10&&std::string(argv[1])=="downscale"){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      const float scale=argc>=8?std::stof(argv[7]):0.5f;
      const auto interpolation=[&](int index,augmatch::Interpolation fallback){return argc>index?static_cast<augmatch::Interpolation>(std::stoi(argv[index])):fallback;};
      const augmatch::DownscaleConfig cfg{w,h,c,scale,interpolation(8,augmatch::Interpolation::Linear),interpolation(9,augmatch::Interpolation::Linear)};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::downscale_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::downscale_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="axis_scale"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),axis=std::stoi(argv[7]);float factor=std::stof(argv[8]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::AxisScaleConfig cfg{w,h,c,factor,augmatch::Interpolation::Linear,0};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(axis==0)augmatch::scale_x_u8(di,doo,cfg);else augmatch::scale_y_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(axis==0)augmatch::scale_x_u8(in.data(),out.data(),cfg);else augmatch::scale_y_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="axis_translate"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),axis=std::stoi(argv[7]);float offset=std::stof(argv[8]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::AxisTranslateConfig cfg{w,h,c,offset,augmatch::Interpolation::Linear,0};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(axis==0)augmatch::translate_x_u8(di,doo,cfg);else augmatch::translate_y_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(axis==0)augmatch::translate_x_u8(in.data(),out.data(),cfg);else augmatch::translate_y_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="axis_shear"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),axis=std::stoi(argv[7]);float shear=std::stof(argv[8]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::AxisShearConfig cfg{w,h,c,shear,augmatch::Interpolation::Linear,0};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(axis==0)augmatch::shear_x_u8(di,doo,cfg);else augmatch::shear_y_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(axis==0)augmatch::shear_x_u8(in.data(),out.data(),cfg);else augmatch::shear_y_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==7&&std::string(argv[1])=="bayer"){  
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),pattern=std::stoi(argv[6]);std::size_t ni=static_cast<std::size_t>(w)*h*3,no=static_cast<std::size_t>(w)*h;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");augmatch::BayerConfig cfg{w,h,static_cast<augmatch::BayerPattern>(pattern)};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::bayer_sample_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::bayer_sample_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==7&&(std::string(argv[1])=="quad_bayer"||std::string(argv[1])=="rgbw")){  
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),pattern=std::stoi(argv[6]);
      const bool rgbw=std::string(argv[1])=="rgbw";
      const std::size_t ni=static_cast<std::size_t>(w)*h*(rgbw?4:3),no=static_cast<std::size_t>(w)*h;
      std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));
      if(rgbw)augmatch::rgbw_sample_u8(di,doo,{w,h,static_cast<augmatch::RGBWPattern>(pattern)});else augmatch::quad_bayer_sample_u8(di,doo,{w,h,static_cast<augmatch::QuadBayerPattern>(pattern)});
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(rgbw)augmatch::rgbw_sample_u8(in.data(),out.data(),{w,h,static_cast<augmatch::RGBWPattern>(pattern)});else augmatch::quad_bayer_sample_u8(in.data(),out.data(),{w,h,static_cast<augmatch::QuadBayerPattern>(pattern)});
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="custom_cfa"){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),channels=std::stoi(argv[6]),mw=std::stoi(argv[7]),mh=std::stoi(argv[8]);
      const std::size_t ni=static_cast<std::size_t>(w)*h*channels,no=static_cast<std::size_t>(w)*h,nm=static_cast<std::size_t>(mw)*mh;
      std::vector<std::uint8_t> in(ni),out(no),mask(nm);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");
      std::ifstream fm(argv[9],std::ios::binary);fm.read(reinterpret_cast<char*>(mask.data()),nm);if(!fm||fm.gcount()!=static_cast<std::streamsize>(nm))throw std::runtime_error("failed to read CFA mask");
      const augmatch::CustomCfaConfig cfg{w,h,channels,mw,mh,mask.data()};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMalloc(&dm,nm));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));ck(cudaMemcpy(dm,mask.data(),nm,cudaMemcpyHostToDevice));
      augmatch::custom_cfa_sample_u8(di,doo,{w,h,channels,mw,mh,dm});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));ck(cudaFree(dm));
#else
      augmatch::custom_cfa_sample_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="signal"){ 
    try{const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);augmatch::SignalGainConfig cfg{w,h,c,std::stof(argv[8])};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(op=="exposure")augmatch::exposure_scale_u8(di,doo,cfg);else if(op=="analog")augmatch::analog_gain_u8(di,doo,cfg);else if(op=="digital")augmatch::digital_gain_u8(di,doo,cfg);else if(op=="saturation")augmatch::pixel_saturation_u8(di,doo,{w,h,c,std::stoi(argv[8])});else if(op=="black")augmatch::black_level_u8(di,doo,{w,h,c,std::stoi(argv[8])});else if(op=="adc")augmatch::adc_quantize_u8(di,doo,{w,h,c,std::stoi(argv[8])});else if(op=="bits")augmatch::bit_reduce_u8(di,doo,{w,h,c,std::stoi(argv[8])});else if(op=="cross")augmatch::pixel_cross_talk_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="leak")augmatch::charge_leakage_u8(di,doo,{w,h,c,std::stof(argv[8])});else throw std::invalid_argument("unknown signal operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="exposure")augmatch::exposure_scale_u8(in.data(),out.data(),cfg);else if(op=="analog")augmatch::analog_gain_u8(in.data(),out.data(),cfg);else if(op=="digital")augmatch::digital_gain_u8(in.data(),out.data(),cfg);else if(op=="saturation")augmatch::pixel_saturation_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});else if(op=="black")augmatch::black_level_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});else if(op=="adc")augmatch::adc_quantize_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});else if(op=="bits")augmatch::bit_reduce_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});else if(op=="cross")augmatch::pixel_cross_talk_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="leak")augmatch::charge_leakage_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else throw std::invalid_argument("unknown signal operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==8||argc==9||argc==15)&&(std::string(argv[1])=="to_tensor_3d"||std::string(argv[1])=="tensor3d")){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),d=std::stoi(argv[6]),c=std::stoi(argv[7]);
      const std::size_t n=static_cast<std::size_t>(w)*h*d*c;
      augmatch::ToTensor3DConfig cfg{w,h,d,c,false};
      if(argc==9){
        const std::string flag=argv[8];
        if(flag=="1"||flag=="true"||flag=="normalize") cfg.normalize=true;
        else if(flag!="0"&&flag!="false"&&flag!="none") throw std::invalid_argument("ToTensor3D normalization flag must be 0 or 1");
      } else if(argc==15) {
        cfg.normalize=std::stoi(argv[8])!=0;cfg.mean0=std::stof(argv[9]);cfg.mean1=std::stof(argv[10]);cfg.mean2=std::stof(argv[11]);cfg.std0=std::stof(argv[12]);cfg.std1=std::stof(argv[13]);cfg.std2=std::stof(argv[14]);
      }
      std::vector<std::uint8_t> in(n);std::vector<float> out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t* di=nullptr;float* doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::to_tensor_3d_u8_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::to_tensor_3d_u8_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write tensor output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==7||argc==8||argc==14)&&(std::string(argv[1])=="to_tensor_v2"||std::string(argv[1])=="tensor")){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;
      augmatch::ToTensorV2Config cfg{w,h,c,false};
      if(argc==8){
        const std::string flag=argv[7];
        if(flag=="1"||flag=="true"||flag=="normalize") cfg.normalize=true;
        else if(flag!="0"&&flag!="false"&&flag!="none") throw std::invalid_argument("ToTensorV2 normalization flag must be 0 or 1");
      } else if(argc==14) {
        cfg.normalize=std::stoi(argv[7])!=0;cfg.mean0=std::stof(argv[8]);cfg.mean1=std::stof(argv[9]);cfg.mean2=std::stof(argv[10]);cfg.std0=std::stof(argv[11]);cfg.std1=std::stof(argv[12]);cfg.std2=std::stof(argv[13]);
      }
      std::vector<std::uint8_t> in(n);std::vector<float> out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t* di=nullptr;float* doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::to_tensor_v2_u8_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::to_tensor_v2_u8_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write tensor output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==8&&std::string(argv[1])=="convert"){ 
    try{const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::ifstream fi(argv[3],std::ios::binary);
#if AUGMATCH_HAS_CUDA
      if(op=="to_float"){std::vector<std::uint8_t> in(n);std::vector<float> out(n);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");std::uint8_t* di=nullptr;float* doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::to_float_u8(di,doo,{w,h,c});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n*sizeof(float));}
      else if(op=="from_float"){std::vector<float> in(n);std::vector<std::uint8_t> out(n);fi.read(reinterpret_cast<char*>(in.data()),n*sizeof(float));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");float* di=nullptr;std::uint8_t* doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));augmatch::from_float_u8(di,doo,{w,h,c});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);}
      else throw std::invalid_argument("unknown conversion operation");
#else
      if(op=="to_float"){std::vector<std::uint8_t> in(n);std::vector<float> out(n);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::to_float_u8(in.data(),out.data(),{w,h,c});std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n*sizeof(float));}else if(op=="from_float"){std::vector<float> in(n);std::vector<std::uint8_t> out(n);fi.read(reinterpret_cast<char*>(in.data()),n*sizeof(float));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");augmatch::from_float_u8(in.data(),out.data(),{w,h,c});std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);}else throw std::invalid_argument("unknown conversion operation");
#endif
      return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==13&&std::string(argv[1])=="normalize"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::NormalizeConfig cfg{w,h,c,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11]),std::stof(argv[12])};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n);std::vector<float> out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t* di=nullptr;float* doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::normalize_u8_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::normalize_u8_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n*sizeof(float));return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==7&&std::string(argv[1])=="to_rgb"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);std::size_t ni=static_cast<std::size_t>(w)*h*c,no=static_cast<std::size_t>(w)*h*3;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");augmatch::ToRGBConfig cfg{w,h,c};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::to_rgb_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::to_rgb_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==8&&std::string(argv[1])=="flip"){ 
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),axis=std::stoi(argv[7]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::flip_u8(di,doo,w,h,c,axis);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::flip_u8(in.data(),out.data(),w,h,c,axis);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==11&&std::string(argv[1])=="tone"&&(std::string(argv[2])=="clahe"||std::string(argv[2])=="all_channels_clahe"))||(argc==10&&(std::string(argv[1])=="clahe"||std::string(argv[1])=="all_channels_clahe"))){ 
    try{
      const bool nested=std::string(argv[1])=="tone";
      const int in_arg=nested?3:2,out_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,clip_arg=nested?8:7,tx_arg=nested?9:8,ty_arg=nested?10:9;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);
      const bool all_channels=(std::string(argv[1])=="all_channels_clahe"||(nested&&std::string(argv[2])=="all_channels_clahe"));
      const augmatch::CLAHEConfig cfg{w,h,c,std::stof(argv[clip_arg]),std::stoi(argv[tx_arg]),std::stoi(argv[ty_arg])};
      const augmatch::AllChannelsCLAHEConfig all_channels_cfg{w,h,c,std::stof(argv[clip_arg]),std::stoi(argv[tx_arg]),std::stoi(argv[ty_arg])};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[in_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(all_channels)augmatch::all_channels_clahe_u8(di,doo,all_channels_cfg);else augmatch::clahe_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(all_channels)augmatch::all_channels_clahe_u8(in.data(),out.data(),all_channels_cfg);else augmatch::clahe_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[out_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  const auto tone_variation_name=[](const std::string& op){return op=="gamma_variation"||op=="s_curve_contrast_variation"||op=="s_curve"||op=="highlight_rolloff_variation"||op=="highlight_rolloff"||op=="shadow_lift"||op=="shadow_crush"||op=="posterization"||op=="posterize_variation"||op=="low_bit_depth_banding"||op=="banding";};
  if(argc>=8 && ((std::string(argv[1])=="tone"&&tone_variation_name(argv[2]))||tone_variation_name(argv[1]))){
    try{
      const bool nested=std::string(argv[1])=="tone";const std::string op=nested?argv[2]:argv[1];const int in_arg=nested?3:2,out_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,p0_arg=nested?8:7,p1_arg=nested?9:8;const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);
      if(op=="gamma_variation") {if(argc!=p0_arg+1)throw std::invalid_argument("gamma_variation expects GAMMA");run_filter_cli(argv[1],argv[in_arg],argv[out_arg],augmatch::GammaVariationConfig{w,h,c,std::stof(argv[p0_arg])},augmatch::gamma_variation_u8);}
      else if(op=="s_curve_contrast_variation"||op=="s_curve") {if(argc!=p0_arg+1)throw std::invalid_argument("s_curve_contrast_variation expects AMOUNT");run_filter_cli(argv[1],argv[in_arg],argv[out_arg],augmatch::SCurveContrastVariationConfig{w,h,c,std::stof(argv[p0_arg])},augmatch::s_curve_contrast_variation_u8);}
      else if(op=="highlight_rolloff_variation"||op=="highlight_rolloff") {if(argc!=p1_arg+1)throw std::invalid_argument("highlight_rolloff expects THRESHOLD STRENGTH");run_filter_cli(argv[1],argv[in_arg],argv[out_arg],augmatch::HighlightRolloffVariationConfig{w,h,c,std::stof(argv[p0_arg]),std::stof(argv[p1_arg])},augmatch::highlight_rolloff_variation_u8);}
      else if(op=="shadow_lift") {if(argc!=p1_arg+1)throw std::invalid_argument("shadow_lift expects AMOUNT THRESHOLD");run_filter_cli(argv[1],argv[in_arg],argv[out_arg],augmatch::ShadowLiftConfig{w,h,c,std::stof(argv[p0_arg]),std::stof(argv[p1_arg])},augmatch::shadow_lift_u8);}
      else if(op=="shadow_crush") {if(argc!=p1_arg+1)throw std::invalid_argument("shadow_crush expects AMOUNT THRESHOLD");run_filter_cli(argv[1],argv[in_arg],argv[out_arg],augmatch::ShadowCrushConfig{w,h,c,std::stof(argv[p0_arg]),std::stof(argv[p1_arg])},augmatch::shadow_crush_u8);}
      else if(op=="posterization"||op=="posterize_variation") {if(argc!=p0_arg+1)throw std::invalid_argument("posterization expects BITS");run_filter_cli(argv[1],argv[in_arg],argv[out_arg],augmatch::PosterizationConfig{w,h,c,std::stoi(argv[p0_arg])},augmatch::posterization_u8);}
      else {if(argc!=p0_arg+1)throw std::invalid_argument("low_bit_depth_banding expects BITS");run_filter_cli(argv[1],argv[in_arg],argv[out_arg],augmatch::LowBitDepthBandingConfig{w,h,c,std::stoi(argv[p0_arg])},augmatch::low_bit_depth_banding_u8);}
      return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  const bool nested_tone_curve=argc==9&&std::string(argv[1])=="tone"&&std::string(argv[2])=="tone_curve_variation";
  const bool top_tone_curve=argc==8&&std::string(argv[1])=="tone_curve_variation";
  if(nested_tone_curve||top_tone_curve){
    try{
      const int in_arg=nested_tone_curve?3:2,out_arg=nested_tone_curve?4:3,w_arg=nested_tone_curve?5:4,h_arg=nested_tone_curve?6:5,c_arg=nested_tone_curve?7:6,lut_arg=nested_tone_curve?8:7;const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);const std::size_t n=static_cast<std::size_t>(w)*h*c,lut_size=static_cast<std::size_t>(c)*256;std::vector<std::uint8_t> in(n),out(n),lut(lut_size);std::ifstream fi(argv[in_arg],std::ios::binary),fl(argv[lut_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);fl.read(reinterpret_cast<char*>(lut.data()),static_cast<std::streamsize>(lut_size));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!fl||fl.gcount()!=static_cast<std::streamsize>(lut_size))throw std::runtime_error("failed to read tone curve input or LUT");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr,*dl=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMalloc(&dl,lut_size));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dl,lut.data(),lut_size,cudaMemcpyHostToDevice));augmatch::tone_curve_variation_u8(di,doo,{w,h,c,dl});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(dl));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::tone_curve_variation_u8(in.data(),out.data(),{w,h,c,lut.data()});
#endif
      std::ofstream fo(argv[out_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  const bool nested_sigmoid=argc==10&&std::string(argv[1])=="tone"&&(std::string(argv[2])=="sigmoid_contrast"||std::string(argv[2])=="sigmoid");
  const bool top_sigmoid=argc==9&&(std::string(argv[1])=="sigmoid_contrast"||std::string(argv[1])=="sigmoid");
  const bool nested_log=argc==10&&std::string(argv[1])=="tone"&&(std::string(argv[2])=="log_contrast"||std::string(argv[2])=="log");
  const bool top_log=argc==9&&(std::string(argv[1])=="log_contrast"||std::string(argv[1])=="log");
  if(nested_sigmoid||top_sigmoid||nested_log||top_log){
    try{
      const bool sigmoid=nested_sigmoid||top_sigmoid;
      const bool nested=nested_sigmoid||nested_log;
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,p0_arg=nested?8:7,p1_arg=nested?9:8;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(sigmoid)augmatch::sigmoid_contrast_u8(di,doo,{w,h,c,std::stof(argv[p0_arg]),std::stof(argv[p1_arg])});
      else augmatch::log_contrast_u8(di,doo,{w,h,c,std::stof(argv[p0_arg]),std::stof(argv[p1_arg])});
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(sigmoid)augmatch::sigmoid_contrast_u8(in.data(),out.data(),{w,h,c,std::stof(argv[p0_arg]),std::stof(argv[p1_arg])});
      else augmatch::log_contrast_u8(in.data(),out.data(),{w,h,c,std::stof(argv[p0_arg]),std::stof(argv[p1_arg])});
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));
      if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(std::string(argv[1])=="tone"&&(argc==8||(argc>=8&&argc<=10&&std::string(argv[2])=="random_tone_curve"))){   
    try{const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      std::vector<std::uint8_t> lut;const bool has_lut=op=="random_tone_curve"&&argc>=9&&std::string(argv[8])!="-";std::uint64_t seed=0;if(has_lut){lut.resize(static_cast<std::size_t>(c)*256);std::ifstream fl(argv[8],std::ios::binary);fl.read(reinterpret_cast<char*>(lut.data()),static_cast<std::streamsize>(lut.size()));if(!fl||fl.gcount()!=static_cast<std::streamsize>(lut.size()))throw std::runtime_error("failed to read random tone curve LUT");}if(op=="random_tone_curve"&&argc==10)seed=static_cast<std::uint64_t>(std::stoull(argv[9]));
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr,*dl=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(op=="sepia")augmatch::sepia_u8(di,doo,{w,h,c});else if(op=="autocontrast")augmatch::autocontrast_u8(di,doo,{w,h,c});else if(op=="equalize")augmatch::equalize_u8(di,doo,{w,h,c});else if(op=="all_channels_histogram_equalization")augmatch::all_channels_histogram_equalization_u8(di,doo,{w,h,c});else if(op=="random_tone_curve"){if(has_lut){ck(cudaMalloc(&dl,lut.size()));ck(cudaMemcpy(dl,lut.data(),lut.size(),cudaMemcpyHostToDevice));}augmatch::random_tone_curve_u8(di,doo,{w,h,c,dl,seed});}else throw std::invalid_argument("unknown tone operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dl)ck(cudaFree(dl));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="sepia")augmatch::sepia_u8(in.data(),out.data(),{w,h,c});else if(op=="autocontrast")augmatch::autocontrast_u8(in.data(),out.data(),{w,h,c});else if(op=="equalize")augmatch::equalize_u8(in.data(),out.data(),{w,h,c});else if(op=="all_channels_histogram_equalization")augmatch::all_channels_histogram_equalization_u8(in.data(),out.data(),{w,h,c});else if(op=="random_tone_curve")augmatch::random_tone_curve_u8(in.data(),out.data(),{w,h,c,has_lut?lut.data():nullptr,seed});else throw std::invalid_argument("unknown tone operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="color"&&(std::string(argv[2])=="uniform_color_quantization"||std::string(argv[2])=="quantize_uniform"||std::string(argv[2])=="uniform_color_quantization_to_n_bits"||std::string(argv[2])=="quantize_uniform_to_n_bits")){
    try{
      const std::string op=argv[2];const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(op=="uniform_color_quantization"||op=="quantize_uniform")augmatch::uniform_color_quantization_u8(di,doo,{w,h,c,std::stoi(argv[8])});else augmatch::uniform_color_quantization_to_n_bits_u8(di,doo,{w,h,c,std::stoi(argv[8])});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="uniform_color_quantization"||op=="quantize_uniform")augmatch::uniform_color_quantization_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});else augmatch::uniform_color_quantization_to_n_bits_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  const bool nested_temperature=argc==9&&std::string(argv[1])=="color"&&(std::string(argv[2])=="planckian_jitter"||std::string(argv[2])=="planckian"||std::string(argv[2])=="change_color_temperature"||std::string(argv[2])=="change_temperature");
  const bool top_level_temperature=argc==8&&(std::string(argv[1])=="change_color_temperature"||std::string(argv[1])=="change_temperature");
  if(nested_temperature||top_level_temperature){
    try{
      const std::string op=nested_temperature?argv[2]:argv[1];
      if(nested_temperature&&op!="planckian_jitter"&&op!="planckian"&&op!="change_color_temperature"&&op!="change_temperature")throw std::invalid_argument("unknown color temperature operation");
      const int input_arg=nested_temperature?3:2,output_arg=nested_temperature?4:3,w_arg=nested_temperature?5:4,h_arg=nested_temperature?6:5,c_arg=nested_temperature?7:6,temp_arg=nested_temperature?8:7;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(op=="planckian_jitter"||op=="planckian")augmatch::planckian_jitter_u8(di,doo,{w,h,c,std::stof(argv[temp_arg])});else augmatch::change_color_temperature_u8(di,doo,{w,h,c,std::stof(argv[temp_arg])});
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="planckian_jitter"||op=="planckian")augmatch::planckian_jitter_u8(in.data(),out.data(),{w,h,c,std::stof(argv[temp_arg])});else augmatch::change_color_temperature_u8(in.data(),out.data(),{w,h,c,std::stof(argv[temp_arg])});
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==14&&std::string(argv[1])=="chromatic_aberration")||(argc==15&&std::string(argv[1])=="color"&&std::string(argv[2])=="chromatic_aberration")){
    try{
      const bool nested=std::string(argv[1])=="color";const int o=nested?1:0;
      const int w=std::stoi(argv[4+o]),h=std::stoi(argv[5+o]),c=std::stoi(argv[6+o]);
      const augmatch::ChromaticAberrationConfig cfg{w,h,c,std::stof(argv[7+o]),std::stof(argv[8+o]),std::stof(argv[9+o]),std::stof(argv[10+o]),std::stof(argv[11+o]),std::stof(argv[12+o]),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[13+o]))};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2+o],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::chromatic_aberration_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::chromatic_aberration_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3+o],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc>=2&&std::string(argv[1])=="fancy_pca")||(argc>=3&&std::string(argv[1])=="color"&&std::string(argv[2])=="fancy_pca")){
    try{
      const bool nested=std::string(argv[1])=="color";const int first=nested?8:7;const int input_arg=nested?3:2,output_arg=nested?4:3;
      if(argc!=first+15&&argc!=first+16)throw std::invalid_argument("FancyPCA expects 15 parameters (basis, eigenvalues, perturbation) and optional alpha");
      const int w=std::stoi(argv[nested?5:4]),h=std::stoi(argv[nested?6:5]),c=std::stoi(argv[nested?7:6]);augmatch::FancyPCAConfig cfg{w,h,c};
      for(int i=0;i<9;++i)cfg.basis[i]=std::stof(argv[first+i]);for(int i=0;i<3;++i)cfg.eigenvalues[i]=std::stof(argv[first+9+i]);for(int i=0;i<3;++i)cfg.perturbation[i]=std::stof(argv[first+12+i]);if(argc==first+16)cfg.alpha=std::stof(argv[first+15]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::fancy_pca_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::fancy_pca_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10&&std::string(argv[1])=="color"&&std::string(argv[2])=="dithering")||(argc==9&&std::string(argv[1])=="dithering")){ 
    try{
      const bool nested=std::string(argv[1])=="color";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,bits_arg=nested?8:7,mode_arg=nested?9:8;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);
      const std::string mode=argv[mode_arg];
      augmatch::DitheringMode selected;
      if(mode=="ordered"||mode=="bayer"||mode=="ordered_bayer4x4") selected=augmatch::DitheringMode::OrderedBayer4x4;
      else if(mode=="error_diffusion"||mode=="floyd"||mode=="fs"||mode=="floyd_steinberg"||mode=="error_diffusion_floyd_steinberg") selected=augmatch::DitheringMode::ErrorDiffusionFloydSteinberg;
      else throw std::invalid_argument("unknown dithering mode; use ordered or error_diffusion");
      const augmatch::DitheringConfig cfg{w,h,c,std::stoi(argv[bits_arg]),selected};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::dithering_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::dithering_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==11||argc==12)&&std::string(argv[1])=="color"&&std::string(argv[2])=="plasma_contrast"){
    try{
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t pixels=static_cast<std::size_t>(w)*h,n=pixels*c;
      std::vector<std::uint8_t> in(n),out(n);std::vector<float> field;
      std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const bool explicit_field=argc==12;if(explicit_field){field.resize(pixels);std::ifstream ff(argv[11],std::ios::binary);ff.read(reinterpret_cast<char*>(field.data()),static_cast<std::streamsize>(field.size()*sizeof(float)));if(!ff||ff.gcount()!=static_cast<std::streamsize>(field.size()*sizeof(float)))throw std::runtime_error("failed to read plasma field");}
      augmatch::PlasmaContrastConfig cfg{w,h,c,std::stof(argv[8]),std::stof(argv[9]),explicit_field?field.data():nullptr,static_cast<std::uint64_t>(std::stoull(argv[10]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *df=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(explicit_field){ck(cudaMalloc(&df,field.size()*sizeof(float)));ck(cudaMemcpy(df,field.data(),field.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.field=df;}augmatch::plasma_contrast_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(df)ck(cudaFree(df));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::plasma_contrast_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==12||argc==13)&&std::string(argv[1])=="color"&&std::string(argv[2])=="plasma_brightness_contrast"){
    try{
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t pixels=static_cast<std::size_t>(w)*h,n=pixels*c;
      std::vector<std::uint8_t> in(n),out(n);std::vector<float> field;
      std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const bool explicit_field=argc==13;if(explicit_field){field.resize(pixels);std::ifstream ff(argv[12],std::ios::binary);ff.read(reinterpret_cast<char*>(field.data()),static_cast<std::streamsize>(field.size()*sizeof(float)));if(!ff||ff.gcount()!=static_cast<std::streamsize>(field.size()*sizeof(float)))throw std::runtime_error("failed to read plasma field");}
      augmatch::PlasmaBrightnessContrastConfig cfg{w,h,c,std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),explicit_field?field.data():nullptr,static_cast<std::uint64_t>(std::stoull(argv[11]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *df=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(explicit_field){ck(cudaMalloc(&df,field.size()*sizeof(float)));ck(cudaMemcpy(df,field.data(),field.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.field=df;}augmatch::plasma_brightness_contrast_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(df)ck(cudaFree(df));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::plasma_brightness_contrast_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==11||argc==12)&&std::string(argv[1])=="color"&&std::string(argv[2])=="plasma_shadow"){
    try{
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t pixels=static_cast<std::size_t>(w)*h,n=pixels*c;
      std::vector<std::uint8_t> in(n),out(n);std::vector<float> field;
      std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const bool explicit_field=argc==12;if(explicit_field){field.resize(pixels);std::ifstream ff(argv[11],std::ios::binary);ff.read(reinterpret_cast<char*>(field.data()),static_cast<std::streamsize>(field.size()*sizeof(float)));if(!ff||ff.gcount()!=static_cast<std::streamsize>(field.size()*sizeof(float)))throw std::runtime_error("failed to read plasma field");}
      augmatch::PlasmaShadowConfig cfg{w,h,c,std::stof(argv[8]),std::stof(argv[9]),explicit_field?field.data():nullptr,static_cast<std::uint64_t>(std::stoull(argv[10]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *df=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(explicit_field){ck(cudaMalloc(&df,field.size()*sizeof(float)));ck(cudaMemcpy(df,field.data(),field.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.field=df;}augmatch::plasma_shadow_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(df)ck(cudaFree(df));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::plasma_shadow_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="color"){   
    try{
      const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);std::vector<std::uint8_t> in(size_t(w)*h*c),out(in.size());std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));if(op=="rgb_shift")augmatch::rgb_shift_u8(di,doo,{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10])});else if(op=="hsv_shift")augmatch::hsv_shift_u8(di,doo,{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10])});else if(op=="shuffle")augmatch::channel_shuffle_u8(di,doo,{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10])});else throw std::invalid_argument("unknown color operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="rgb_shift")augmatch::rgb_shift_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10])});else if(op=="hsv_shift")augmatch::hsv_shift_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10])});else if(op=="shuffle")augmatch::channel_shuffle_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10])});else throw std::invalid_argument("unknown color operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==17&&std::string(argv[1])=="color"&&std::string(argv[2])=="random_color_jitter")||(argc==16&&std::string(argv[1])=="random_color_jitter")){
    try{
      const bool nested=std::string(argv[1])=="color";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6;
      const int p=nested?8:7;const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);
      augmatch::RandomColorJitterConfig cfg{w,h,c,std::stof(argv[p]),std::stof(argv[p+1]),std::stof(argv[p+2]),std::stof(argv[p+3]),std::stof(argv[p+4]),std::stof(argv[p+5]),std::stof(argv[p+6]),std::stof(argv[p+7]),static_cast<std::uint64_t>(std::stoull(argv[p+8]))};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::random_color_jitter_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_color_jitter_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(std::string(argv[1])=="color"&&argc>=8&&argc<=10){
    const std::string op=argv[2];
    const bool one=op=="multiply_brightness"||op=="add_to_brightness"||op=="multiply_hue"||op=="multiply_saturation"||op=="add_to_hue"||op=="add_to_saturation";
    const bool multiply_add=op=="multiply_and_add_to_brightness";
    const bool two=op=="with_hue_and_saturation"||op=="multiply_hue_and_saturation"||op=="add_to_hue_and_saturation";
    const bool zero=op=="remove_saturation";
    if((one&&argc==9)||(two&&argc==10)||(zero&&argc==8)||(multiply_add&&argc==10)) try{
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(op=="multiply_brightness")augmatch::multiply_brightness_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="add_to_brightness")augmatch::add_to_brightness_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="multiply_and_add_to_brightness")augmatch::multiply_and_add_to_brightness_u8(di,doo,{w,h,c,std::stof(argv[8]),std::stof(argv[9])});else if(op=="with_hue_and_saturation")augmatch::with_hue_and_saturation_u8(di,doo,{w,h,c,std::stof(argv[8]),std::stof(argv[9])});else if(op=="multiply_hue_and_saturation")augmatch::multiply_hue_and_saturation_u8(di,doo,{w,h,c,std::stof(argv[8]),std::stof(argv[9])});else if(op=="multiply_hue")augmatch::multiply_hue_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="multiply_saturation")augmatch::multiply_saturation_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="remove_saturation")augmatch::remove_saturation_u8(di,doo,{w,h,c});else if(op=="add_to_hue_and_saturation")augmatch::add_to_hue_and_saturation_u8(di,doo,{w,h,c,std::stof(argv[8]),std::stof(argv[9])});else if(op=="add_to_hue")augmatch::add_to_hue_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="add_to_saturation")augmatch::add_to_saturation_u8(di,doo,{w,h,c,std::stof(argv[8])});else throw std::invalid_argument("unknown HSV color operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="multiply_brightness")augmatch::multiply_brightness_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="add_to_brightness")augmatch::add_to_brightness_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="multiply_and_add_to_brightness")augmatch::multiply_and_add_to_brightness_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),std::stof(argv[9])});else if(op=="with_hue_and_saturation")augmatch::with_hue_and_saturation_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),std::stof(argv[9])});else if(op=="multiply_hue_and_saturation")augmatch::multiply_hue_and_saturation_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),std::stof(argv[9])});else if(op=="multiply_hue")augmatch::multiply_hue_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="multiply_saturation")augmatch::multiply_saturation_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="remove_saturation")augmatch::remove_saturation_u8(in.data(),out.data(),{w,h,c});else if(op=="add_to_hue_and_saturation")augmatch::add_to_hue_and_saturation_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),std::stof(argv[9])});else if(op=="add_to_hue")augmatch::add_to_hue_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="add_to_saturation")augmatch::add_to_saturation_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else throw std::invalid_argument("unknown HSV color operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==15||argc==14)&&std::string(argv[1])=="color"&&(std::string(argv[2])=="with_colorspace"||std::string(argv[2])=="with_brightness_channels")){ 
    try{
      const std::string op=argv[2];const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const std::string space=argv[8];augmatch::ColorSpace cs;if(space=="rgb"||space=="RGB")cs=augmatch::ColorSpace::RGB;else if(space=="hsv"||space=="HSV")cs=augmatch::ColorSpace::HSV;else if(space=="lab"||space=="LAB")cs=augmatch::ColorSpace::LAB;else throw std::invalid_argument("colorspace must be rgb, hsv, or lab");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(op=="with_colorspace"){if(argc!=15)throw std::invalid_argument("with_colorspace expects six parameters");augmatch::with_colorspace_u8(di,doo,{w,h,c,cs,std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11]),std::stof(argv[12]),std::stof(argv[13]),std::stof(argv[14])});}
      else{if(argc!=14)throw std::invalid_argument("with_brightness_channels expects five parameters");augmatch::with_brightness_channels_u8(di,doo,{w,h,c,cs,std::stoi(argv[9]),std::stoi(argv[10]),std::stoi(argv[11]),std::stof(argv[12]),std::stof(argv[13])});}
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="with_colorspace"){if(argc!=15)throw std::invalid_argument("with_colorspace expects six parameters");augmatch::with_colorspace_u8(in.data(),out.data(),{w,h,c,cs,std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11]),std::stof(argv[12]),std::stof(argv[13]),std::stof(argv[14])});}
      else{if(argc!=14)throw std::invalid_argument("with_brightness_channels expects five parameters");augmatch::with_brightness_channels_u8(in.data(),out.data(),{w,h,c,cs,std::stoi(argv[9]),std::stoi(argv[10]),std::stoi(argv[11]),std::stof(argv[12]),std::stof(argv[13])});}
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="color"&&std::string(argv[2])=="color_jitter"){ 
    try{
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const augmatch::ColorJitterConfig cfg{w,h,c,std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::color_jitter_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::color_jitter_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10&&std::string(argv[1])=="xy_masking")||(argc==11&&std::string(argv[1])=="dropout"&&std::string(argv[2])=="xy_masking")){
    try{
      const bool nested=std::string(argv[1])=="dropout";
      const int in_arg=nested?3:2,out_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,fill_arg=nested?8:7,rows_arg=nested?9:8,columns_arg=nested?10:9;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);
      const auto rows=parse_xy_intervals(argv[rows_arg],h,"row");
      const auto columns=parse_xy_intervals(argv[columns_arg],w,"column");
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[in_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const std::uint8_t fill=static_cast<std::uint8_t>(std::stoi(argv[fill_arg]));
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::XYMaskInterval *dr=nullptr,*dc=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(!rows.empty()){ck(cudaMalloc(&dr,rows.size()*sizeof(augmatch::XYMaskInterval)));ck(cudaMemcpy(dr,rows.data(),rows.size()*sizeof(augmatch::XYMaskInterval),cudaMemcpyHostToDevice));}
      if(!columns.empty()){ck(cudaMalloc(&dc,columns.size()*sizeof(augmatch::XYMaskInterval)));ck(cudaMemcpy(dc,columns.data(),columns.size()*sizeof(augmatch::XYMaskInterval),cudaMemcpyHostToDevice));}
      augmatch::xy_masking_u8(di,doo,{w,h,c,dr,static_cast<int>(rows.size()),dc,static_cast<int>(columns.size()),fill});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dc)ck(cudaFree(dc));if(dr)ck(cudaFree(dr));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::xy_masking_u8(in.data(),out.data(),{w,h,c,rows.data(),static_cast<int>(rows.size()),columns.data(),static_cast<int>(columns.size()),fill});
#endif
      std::ofstream fo(argv[out_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==9&&std::string(argv[1])=="cutout")||(argc==10&&std::string(argv[1])=="dropout"&&std::string(argv[2])=="cutout")){
    try{
      const bool nested=std::string(argv[1])=="dropout";
      const int in_arg=nested?3:2,out_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,fill_arg=nested?8:7,rect_arg=nested?9:8;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);
      const auto rectangles=parse_cutout_rectangles(argv[rect_arg],w,h);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[in_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const augmatch::CutoutConfig cfg{w,h,c,rectangles.data(),static_cast<int>(rectangles.size()),static_cast<std::uint8_t>(std::stoi(argv[fill_arg])),0};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::CutoutRectangle *dr=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(!rectangles.empty()){ck(cudaMalloc(&dr,rectangles.size()*sizeof(augmatch::CutoutRectangle)));ck(cudaMemcpy(dr,rectangles.data(),rectangles.size()*sizeof(augmatch::CutoutRectangle),cudaMemcpyHostToDevice));}augmatch::cutout_u8(di,doo,{w,h,c,dr,static_cast<int>(rectangles.size()),cfg.fill,0});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dr)ck(cudaFree(dr));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::cutout_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[out_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc==10||argc==11)&&std::string(argv[1])=="total_dropout")||((argc==11||argc==12)&&std::string(argv[1])=="dropout"&&std::string(argv[2])=="total")){ 
    try{
      const bool nested=std::string(argv[1])=="dropout";const bool explicit_mask=nested?argc==12:argc==11;
      const int in_arg=nested?3:2,out_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6;
      const int first_arg=nested?8:7;const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),c=std::stoi(argv[c_arg]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[in_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const std::uint8_t fill=static_cast<std::uint8_t>(std::stoi(argv[first_arg+(explicit_mask?0:1)]));const float probability=std::stof(argv[first_arg+(explicit_mask?2:0)]);const std::uint64_t seed=static_cast<std::uint64_t>(std::stoull(argv[first_arg+(explicit_mask?3:2)]));const std::uint8_t mask_value=explicit_mask?static_cast<std::uint8_t>(std::stoi(argv[first_arg+1])):0;const augmatch::TotalDropoutConfig cfg{w,h,c,probability,fill,seed,explicit_mask?&mask_value:nullptr};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(explicit_mask){ck(cudaMalloc(&dm,1));ck(cudaMemcpy(dm,&mask_value,1,cudaMemcpyHostToDevice));}augmatch::total_dropout_u8(di,doo,{w,h,c,probability,fill,seed,dm});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dm)ck(cudaFree(dm));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::total_dropout_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[out_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc>=11&&argc<=13)&&std::string(argv[1])=="dropout"){    
    try{
      const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);std::vector<std::uint8_t> in(size_t(w)*h*c),out(in.size());std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));
      if(op=="pixel")augmatch::pixel_dropout_u8(di,doo,{w,h,c,std::stof(argv[8]),static_cast<std::uint8_t>(std::stoi(argv[9])),static_cast<std::uint64_t>(std::stoull(argv[10]))});else if(op=="channel")augmatch::channel_dropout_u8(di,doo,{w,h,c,std::stof(argv[8]),static_cast<std::uint8_t>(std::stoi(argv[9])),static_cast<std::uint64_t>(std::stoull(argv[10]))});else if(op=="grid")augmatch::grid_dropout_u8(di,doo,{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stof(argv[10]),static_cast<std::uint8_t>(std::stoi(argv[11]))});else if(op=="coarse")augmatch::coarse_dropout_u8(di,doo,{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10]),static_cast<std::uint8_t>(std::stoi(argv[11])),static_cast<std::uint64_t>(std::stoull(argv[12]))});else throw std::invalid_argument("unknown dropout operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="pixel")augmatch::pixel_dropout_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),static_cast<std::uint8_t>(std::stoi(argv[9])),static_cast<std::uint64_t>(std::stoull(argv[10]))});else if(op=="channel")augmatch::channel_dropout_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),static_cast<std::uint8_t>(std::stoi(argv[9])),static_cast<std::uint64_t>(std::stoull(argv[10]))});else if(op=="grid")augmatch::grid_dropout_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stof(argv[10]),static_cast<std::uint8_t>(std::stoi(argv[11]))});else if(op=="coarse")augmatch::coarse_dropout_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10]),static_cast<std::uint8_t>(std::stoi(argv[11])),static_cast<std::uint64_t>(std::stoull(argv[12]))});else throw std::invalid_argument("unknown dropout operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="dropout"&&std::string(argv[2])=="mask"){
    try{
      const int w=std::stoi(argv[6]),h=std::stoi(argv[7]),c=std::stoi(argv[8]);const std::size_t n=static_cast<std::size_t>(w)*h*c,mask_n=static_cast<std::size_t>(w)*h;std::vector<std::uint8_t> in(n),out(n),mask(mask_n);std::ifstream fi(argv[3],std::ios::binary),fm(argv[5],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);fm.read(reinterpret_cast<char*>(mask.data()),mask_n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!fm||fm.gcount()!=static_cast<std::streamsize>(mask_n))throw std::runtime_error("failed to read mask dropout input");const augmatch::MaskDropoutConfig cfg{w,h,c,static_cast<std::uint8_t>(std::stoi(argv[9]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*dm=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&dm,mask_n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dm,mask.data(),mask_n,cudaMemcpyHostToDevice));augmatch::mask_dropout_u8(di,dm,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(dm));ck(cudaFree(doo));
#else
      augmatch::mask_dropout_u8(in.data(),mask.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc==9&&std::string(argv[1])=="fast_snowy_landscape")||
      (argc==10&&std::string(argv[1])=="weather"&&std::string(argv[2])=="fast_snowy_landscape"))){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int ia=nested?3:2,oa=nested?4:3,wa=nested?5:4,ha=nested?6:5,ca=nested?7:6,sa=nested?8:7,aa=nested?9:8;
      const int w=std::stoi(argv[wa]),h=std::stoi(argv[ha]),c=std::stoi(argv[ca]); const std::size_t n=static_cast<std::size_t>(w)*h*c;
      std::vector<std::uint8_t> in(n),out(n); std::ifstream fi(argv[ia],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)) throw std::runtime_error("failed to read raw input");
      augmatch::FastSnowyLandscapeConfig config{w,h,c,std::stof(argv[sa]),std::stof(argv[aa])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr; ck(cudaMalloc(&di,n)); ck(cudaMalloc(&doo,n)); ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice)); augmatch::fast_snowy_landscape_u8(di,doo,config); ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost)); ck(cudaFree(di)); ck(cudaFree(doo));
#else
      augmatch::fast_snowy_landscape_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[oa],std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n)); if(!fo) throw std::runtime_error("failed to write output"); return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc==10&&std::string(argv[1])=="clouds")||
      (argc==11&&std::string(argv[1])=="weather"&&std::string(argv[2])=="clouds"))){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int ia=nested?3:2,oa=nested?4:3,wa=nested?5:4,ha=nested?6:5,ca=nested?7:6,aa=nested?8:7,da=nested?9:8,sea=nested?10:9;
      const int w=std::stoi(argv[wa]),h=std::stoi(argv[ha]),c=std::stoi(argv[ca]); const std::size_t n=static_cast<std::size_t>(w)*h*c;
      std::vector<std::uint8_t> in(n),out(n); std::ifstream fi(argv[ia],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)) throw std::runtime_error("failed to read raw input");
      augmatch::CloudsConfig config{w,h,c,std::stof(argv[aa]),std::stof(argv[da]),nullptr,static_cast<std::uint64_t>(std::stoull(argv[sea]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr; ck(cudaMalloc(&di,n)); ck(cudaMalloc(&doo,n)); ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice)); augmatch::clouds_u8(di,doo,config); ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost)); ck(cudaFree(di)); ck(cudaFree(doo));
#else
      augmatch::clouds_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[oa],std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n)); if(!fo) throw std::runtime_error("failed to write output"); return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc==9&& (std::string(argv[1])=="cloud_layer"||std::string(argv[1])=="snowflakes_layer"||std::string(argv[1])=="rain_layer"))||
      (argc==10&&std::string(argv[1])=="weather"&&(std::string(argv[2])=="cloud_layer"||std::string(argv[2])=="snowflakes_layer"||std::string(argv[2])=="rain_layer")))){
    try{
      const bool nested=std::string(argv[1])=="weather"; const std::string op=nested?argv[2]:argv[1];
      const int ia=nested?3:2,oa=nested?4:3,wa=nested?5:4,ha=nested?6:5,ca=nested?7:6,opa=nested?8:7,fa=nested?9:8;
      const int w=std::stoi(argv[wa]),h=std::stoi(argv[ha]),c=std::stoi(argv[ca]); const std::size_t pixels=static_cast<std::size_t>(w)*h,n=pixels*c;
      std::vector<std::uint8_t> in(n),out(n); std::vector<float> layer(pixels); std::ifstream fi(argv[ia],std::ios::binary),ff(argv[fa],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n)); ff.read(reinterpret_cast<char*>(layer.data()),static_cast<std::streamsize>(pixels*sizeof(float)));
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!ff||ff.gcount()!=static_cast<std::streamsize>(pixels*sizeof(float))) throw std::runtime_error("failed to read weather layer input");
      augmatch::WeatherLayerConfig config{w,h,c,layer.data(),std::stof(argv[opa])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr; float *dl=nullptr; ck(cudaMalloc(&di,n)); ck(cudaMalloc(&doo,n)); ck(cudaMalloc(&dl,pixels*sizeof(float))); ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice)); ck(cudaMemcpy(dl,layer.data(),pixels*sizeof(float),cudaMemcpyHostToDevice)); config.layer=dl;
      if(op=="cloud_layer") augmatch::cloud_layer_u8(di,doo,config); else if(op=="snowflakes_layer") augmatch::snowflakes_layer_u8(di,doo,config); else augmatch::rain_layer_u8(di,doo,config); ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost)); ck(cudaFree(dl)); ck(cudaFree(di)); ck(cudaFree(doo));
#else
      if(op=="cloud_layer") augmatch::cloud_layer_u8(in.data(),out.data(),config); else if(op=="snowflakes_layer") augmatch::snowflakes_layer_u8(in.data(),out.data(),config); else augmatch::rain_layer_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[oa],std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n)); if(!fo) throw std::runtime_error("failed to write output"); return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc>=9&&argc<=16&&(std::string(argv[1])=="random_rain"||std::string(argv[1])=="rain"))||(argc>=10&&argc<=17&&std::string(argv[1])=="weather"&&(std::string(argv[2])=="random_rain"||std::string(argv[2])=="rain")))){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,count_arg=nested?8:7;
      const int alpha_arg=count_arg+1,seed_arg=count_arg+2;
      if(argc<=seed_arg) throw std::invalid_argument("RandomRain expects count, alpha, and seed");
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),channels=std::stoi(argv[c_arg]);
      const int count=std::stoi(argv[count_arg]);
      const float alpha=std::stof(argv[alpha_arg]);
      const std::uint64_t seed=static_cast<std::uint64_t>(std::stoull(argv[seed_arg]));
      const float length_min=argc>seed_arg+1?std::stof(argv[seed_arg+1]):8.0f;
      const float length_max=argc>seed_arg+2?std::stof(argv[seed_arg+2]):24.0f;
      const float angle_min=argc>seed_arg+3?std::stof(argv[seed_arg+3]):75.0f;
      const float angle_max=argc>seed_arg+4?std::stof(argv[seed_arg+4]):105.0f;
      const float streak_width=argc>seed_arg+5?std::stof(argv[seed_arg+5]):1.0f;
      const bool explicit_streaks=argc>seed_arg+6;
      std::vector<augmatch::RainStreak> streaks;
      if(explicit_streaks){
        if(count<0) throw std::invalid_argument("RandomRain streak count must be nonnegative");
        streaks.resize(static_cast<std::size_t>(count));
        std::ifstream fs(argv[seed_arg+6],std::ios::binary);
        fs.read(reinterpret_cast<char*>(streaks.data()),static_cast<std::streamsize>(streaks.size()*sizeof(augmatch::RainStreak)));
        if(!fs||fs.gcount()!=static_cast<std::streamsize>(streaks.size()*sizeof(augmatch::RainStreak)))throw std::runtime_error("failed to read RandomRain streak records");
      }
      augmatch::RandomRainConfig config{w,h,channels,explicit_streaks?streaks.data():nullptr,count,alpha,length_min,length_max,angle_min,angle_max,streak_width,seed};
      const std::size_t n=static_cast<std::size_t>(w)*h*channels;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::RainStreak *ds=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(explicit_streaks&&count>0){ck(cudaMalloc(&ds,streaks.size()*sizeof(augmatch::RainStreak)));ck(cudaMemcpy(ds,streaks.data(),streaks.size()*sizeof(augmatch::RainStreak),cudaMemcpyHostToDevice));config.streaks=ds;}
      augmatch::random_rain_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(ds)ck(cudaFree(ds));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_rain_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc>=9&&argc<=13&&(std::string(argv[1])=="random_snow"||std::string(argv[1])=="snowflakes"))||(argc>=10&&argc<=14&&std::string(argv[1])=="weather"&&(std::string(argv[2])=="random_snow"||std::string(argv[2])=="snowflakes")))){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,count_arg=nested?8:7;
      const int alpha_arg=count_arg+1,seed_arg=count_arg+2;
      if(argc<=seed_arg) throw std::invalid_argument("RandomSnow expects count, alpha, and seed");
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),channels=std::stoi(argv[c_arg]);
      const int count=std::stoi(argv[count_arg]);
      const float alpha=std::stof(argv[alpha_arg]);
      const std::uint64_t seed=static_cast<std::uint64_t>(std::stoull(argv[seed_arg]));
      const float radius_min=argc>seed_arg+1?std::stof(argv[seed_arg+1]):1.0f;
      const float radius_max=argc>seed_arg+2?std::stof(argv[seed_arg+2]):3.0f;
      const bool explicit_snowflakes=argc>seed_arg+3;
      std::vector<augmatch::Snowflake> snowflakes;
      if(explicit_snowflakes){
        if(count<0) throw std::invalid_argument("RandomSnow snowflake count must be nonnegative");
        snowflakes.resize(static_cast<std::size_t>(count));
        std::ifstream fs(argv[seed_arg+3],std::ios::binary);
        fs.read(reinterpret_cast<char*>(snowflakes.data()),static_cast<std::streamsize>(snowflakes.size()*sizeof(augmatch::Snowflake)));
        if(!fs||fs.gcount()!=static_cast<std::streamsize>(snowflakes.size()*sizeof(augmatch::Snowflake)))throw std::runtime_error("failed to read RandomSnow snowflake records");
      }
      augmatch::RandomSnowConfig config{w,h,channels,explicit_snowflakes?snowflakes.data():nullptr,count,alpha,radius_min,radius_max,seed};
      const std::size_t n=static_cast<std::size_t>(w)*h*channels;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::Snowflake *ds=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(explicit_snowflakes&&count>0){ck(cudaMalloc(&ds,snowflakes.size()*sizeof(augmatch::Snowflake)));ck(cudaMemcpy(ds,snowflakes.data(),snowflakes.size()*sizeof(augmatch::Snowflake),cudaMemcpyHostToDevice));config.snowflakes=ds;}
      augmatch::random_snow_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(ds)ck(cudaFree(ds));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_snow_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc==15&&std::string(argv[1])=="snow_stamp")||(argc==16&&std::string(argv[1])=="weather"&&std::string(argv[2])=="snow_stamp"))){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,sw_arg=nested?8:7,sh_arg=nested?9:8,count_arg=nested?10:9;
      const int alpha_arg=count_arg+1,fill_arg=count_arg+2,image_arg=count_arg+3,mask_arg=count_arg+4,records_arg=count_arg+5;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),channels=std::stoi(argv[c_arg]);
      const int stamp_width=std::stoi(argv[sw_arg]),stamp_height=std::stoi(argv[sh_arg]),count=std::stoi(argv[count_arg]);
      const float alpha=std::stof(argv[alpha_arg]);
      const int fill_value=std::stoi(argv[fill_arg]);
      if(fill_value<0||fill_value>255)throw std::invalid_argument("SnowStamp fill must be in [0,255]");
      if(stamp_width<=0||stamp_height<=0||count<0)throw std::invalid_argument("invalid SnowStamp dimensions or count");
      const std::size_t image_size=static_cast<std::size_t>(stamp_width)*stamp_height*channels,mask_size=static_cast<std::size_t>(stamp_width)*stamp_height;
      const std::size_t n=static_cast<std::size_t>(w)*h*channels;std::vector<std::uint8_t> in(n),out(n),image(image_size),mask(mask_size);std::vector<augmatch::SnowStampPlacement> placements(static_cast<std::size_t>(count));
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      if(std::string(argv[image_arg])!="-"){std::ifstream fs(argv[image_arg],std::ios::binary);fs.read(reinterpret_cast<char*>(image.data()),static_cast<std::streamsize>(image_size));if(!fs||fs.gcount()!=static_cast<std::streamsize>(image_size))throw std::runtime_error("failed to read SnowStamp image");}
      if(std::string(argv[mask_arg])!="-"){std::ifstream fs(argv[mask_arg],std::ios::binary);fs.read(reinterpret_cast<char*>(mask.data()),static_cast<std::streamsize>(mask_size));if(!fs||fs.gcount()!=static_cast<std::streamsize>(mask_size))throw std::runtime_error("failed to read SnowStamp mask");}
      if(count>0){std::ifstream fs(argv[records_arg],std::ios::binary);fs.read(reinterpret_cast<char*>(placements.data()),static_cast<std::streamsize>(placements.size()*sizeof(augmatch::SnowStampPlacement)));if(!fs||fs.gcount()!=static_cast<std::streamsize>(placements.size()*sizeof(augmatch::SnowStampPlacement)))throw std::runtime_error("failed to read SnowStamp placement records");}
      augmatch::SnowStampConfig config{w,h,channels,stamp_width,stamp_height,channels,std::string(argv[image_arg])!="-"?image.data():nullptr,std::string(argv[mask_arg])!="-"?mask.data():nullptr,count>0?placements.data():nullptr,count,alpha,static_cast<std::uint8_t>(fill_value)};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr,*dimage=nullptr,*dmask=nullptr;augmatch::SnowStampPlacement *dplacements=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(config.stamp_image){ck(cudaMalloc(&dimage,image_size));ck(cudaMemcpy(dimage,image.data(),image_size,cudaMemcpyHostToDevice));config.stamp_image=dimage;}if(config.stamp_mask){ck(cudaMalloc(&dmask,mask_size));ck(cudaMemcpy(dmask,mask.data(),mask_size,cudaMemcpyHostToDevice));config.stamp_mask=dmask;}if(count>0){ck(cudaMalloc(&dplacements,placements.size()*sizeof(augmatch::SnowStampPlacement)));ck(cudaMemcpy(dplacements,placements.data(),placements.size()*sizeof(augmatch::SnowStampPlacement),cudaMemcpyHostToDevice));config.placements=dplacements;}
      augmatch::snow_stamp_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dplacements)ck(cudaFree(dplacements));if(dmask)ck(cudaFree(dmask));if(dimage)ck(cudaFree(dimage));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::snow_stamp_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc>=10&&argc<=14)&&std::string(argv[1])=="random_gravel")||
     ((argc>=11&&argc<=15)&&std::string(argv[1])=="weather"&&std::string(argv[2])=="random_gravel")){ 
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,count_arg=nested?8:7;
      const int alpha_arg=count_arg+1,seed_arg=count_arg+2;
      if(argc<=seed_arg) throw std::invalid_argument("RandomGravel expects count, alpha, and seed");
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),channels=std::stoi(argv[c_arg]);
      const int count=std::stoi(argv[count_arg]);
      const float alpha=std::stof(argv[alpha_arg]);
      const std::uint64_t seed=static_cast<std::uint64_t>(std::stoull(argv[seed_arg]));
      const float radius_min=argc>seed_arg+1?std::stof(argv[seed_arg+1]):1.0f;
      const float radius_max=argc>seed_arg+2?std::stof(argv[seed_arg+2]):3.0f;
      const int fill_value=argc>seed_arg+3?std::stoi(argv[seed_arg+3]):128;
      if(fill_value<0||fill_value>255)throw std::invalid_argument("RandomGravel fill must be in [0,255]");
      const bool explicit_particles=argc>seed_arg+4;
      std::vector<augmatch::GravelParticle> particles;
      if(explicit_particles){
        if(count<0) throw std::invalid_argument("RandomGravel particle count must be nonnegative");
        particles.resize(static_cast<std::size_t>(count));
        std::ifstream fs(argv[seed_arg+4],std::ios::binary);
        fs.read(reinterpret_cast<char*>(particles.data()),static_cast<std::streamsize>(particles.size()*sizeof(augmatch::GravelParticle)));
        if(!fs||fs.gcount()!=static_cast<std::streamsize>(particles.size()*sizeof(augmatch::GravelParticle)))throw std::runtime_error("failed to read RandomGravel particle records");
      }
      augmatch::RandomGravelConfig config{w,h,channels,explicit_particles?particles.data():nullptr,count,alpha,radius_min,radius_max,static_cast<std::uint8_t>(fill_value),seed};
      const std::size_t n=static_cast<std::size_t>(w)*h*channels;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::GravelParticle *dp=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(explicit_particles&&count>0){ck(cudaMalloc(&dp,particles.size()*sizeof(augmatch::GravelParticle)));ck(cudaMemcpy(dp,particles.data(),particles.size()*sizeof(augmatch::GravelParticle),cudaMemcpyHostToDevice));config.particles=dp;}
      augmatch::random_gravel_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dp)ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_gravel_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc>=10&&argc<=16)&&(std::string(argv[1])=="random_spatter"||std::string(argv[1])=="spatter"))||
     ((argc>=11&&argc<=17)&&std::string(argv[1])=="weather"&&
      (std::string(argv[2])=="random_spatter"||std::string(argv[2])=="spatter"))){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,count_arg=nested?8:7;
      const int alpha_arg=count_arg+1,seed_arg=count_arg+2;
      if(argc<=seed_arg) throw std::invalid_argument("Spatter expects count, alpha, and seed");
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),channels=std::stoi(argv[c_arg]);
      const int count=std::stoi(argv[count_arg]);
      const float alpha=std::stof(argv[alpha_arg]);
      const std::uint64_t seed=static_cast<std::uint64_t>(std::stoull(argv[seed_arg]));
      const float radius_min=argc>seed_arg+1?std::stof(argv[seed_arg+1]):1.0f;
      const float radius_max=argc>seed_arg+2?std::stof(argv[seed_arg+2]):3.0f;
      const int red=argc>seed_arg+3?std::stoi(argv[seed_arg+3]):255;
      const int green=argc>seed_arg+4?std::stoi(argv[seed_arg+4]):255;
      const int blue=argc>seed_arg+5?std::stoi(argv[seed_arg+5]):255;
      if(red<0||red>255||green<0||green>255||blue<0||blue>255)
        throw std::invalid_argument("Spatter color must be in [0,255]");
      const bool explicit_droplets=argc>seed_arg+6;
      std::vector<augmatch::SpatterDroplet> droplets;
      if(explicit_droplets){
        if(count<0) throw std::invalid_argument("Spatter droplet count must be nonnegative");
        droplets.resize(static_cast<std::size_t>(count));
        std::ifstream fs(argv[seed_arg+6],std::ios::binary);
        fs.read(reinterpret_cast<char*>(droplets.data()),static_cast<std::streamsize>(droplets.size()*sizeof(augmatch::SpatterDroplet)));
        if(!fs||fs.gcount()!=static_cast<std::streamsize>(droplets.size()*sizeof(augmatch::SpatterDroplet)))throw std::runtime_error("failed to read Spatter droplet records");
      }
      augmatch::RandomSpatterConfig config{w,h,channels,explicit_droplets?droplets.data():nullptr,count,alpha,radius_min,radius_max,static_cast<std::uint8_t>(red),static_cast<std::uint8_t>(green),static_cast<std::uint8_t>(blue),seed};
      const std::size_t n=static_cast<std::size_t>(w)*h*channels;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::SpatterDroplet *dd=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(explicit_droplets&&count>0){ck(cudaMalloc(&dd,droplets.size()*sizeof(augmatch::SpatterDroplet)));ck(cudaMemcpy(dd,droplets.data(),droplets.size()*sizeof(augmatch::SpatterDroplet),cudaMemcpyHostToDevice));config.droplets=dd;}
      augmatch::random_spatter_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dd)ck(cudaFree(dd));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_spatter_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc>=10&&argc<=21)&&std::string(argv[1])=="random_sun_flare")||
     ((argc>=11&&argc<=22)&&std::string(argv[1])=="weather"&&std::string(argv[2])=="random_sun_flare")){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6,count_arg=nested?8:7;
      const int opacity_arg=count_arg+1,seed_arg=count_arg+2;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),channels=std::stoi(argv[c_arg]);
      const int count=std::stoi(argv[count_arg]);
      const float opacity=std::stof(argv[opacity_arg]);
      const std::uint64_t seed=static_cast<std::uint64_t>(std::stoull(argv[seed_arg]));
      augmatch::SunFlareSource source{0.5f*static_cast<float>(w),0.5f*static_cast<float>(h),8.0f,1.0f};
      if(argc>seed_arg+1) source.x=std::stof(argv[seed_arg+1]);
      if(argc>seed_arg+2) source.y=std::stof(argv[seed_arg+2]);
      if(argc>seed_arg+3) source.radius=std::stof(argv[seed_arg+3]);
      if(argc>seed_arg+4) source.alpha=std::stof(argv[seed_arg+4]);
      const float length_min=argc>seed_arg+5?std::stof(argv[seed_arg+5]):16.0f;
      const float length_max=argc>seed_arg+6?std::stof(argv[seed_arg+6]):48.0f;
      const float angle_min=argc>seed_arg+7?std::stof(argv[seed_arg+7]):0.0f;
      const float angle_max=argc>seed_arg+8?std::stof(argv[seed_arg+8]):360.0f;
      const float width_min=argc>seed_arg+9?std::stof(argv[seed_arg+9]):1.0f;
      const float width_max=argc>seed_arg+10?std::stof(argv[seed_arg+10]):2.0f;
      const bool explicit_rays=argc>seed_arg+11;
      std::vector<augmatch::SunFlareRay> rays;
      if(explicit_rays){
        if(count<0) throw std::invalid_argument("RandomSunFlare ray count must be nonnegative");
        rays.resize(static_cast<std::size_t>(count));
        std::ifstream fs(argv[seed_arg+11],std::ios::binary);
        fs.read(reinterpret_cast<char*>(rays.data()),static_cast<std::streamsize>(rays.size()*sizeof(augmatch::SunFlareRay)));
        if(!fs||fs.gcount()!=static_cast<std::streamsize>(rays.size()*sizeof(augmatch::SunFlareRay)))throw std::runtime_error("failed to read RandomSunFlare ray records");
      }
      augmatch::RandomSunFlareConfig config{w,h,channels,source,explicit_rays?rays.data():nullptr,count,opacity,length_min,length_max,angle_min,angle_max,width_min,width_max,seed};
      const std::size_t n=static_cast<std::size_t>(w)*h*channels;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::SunFlareRay *dr=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(explicit_rays&&count>0){ck(cudaMalloc(&dr,rays.size()*sizeof(augmatch::SunFlareRay)));ck(cudaMemcpy(dr,rays.data(),rays.size()*sizeof(augmatch::SunFlareRay),cudaMemcpyHostToDevice));config.rays=dr;}
      augmatch::random_sun_flare_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dr)ck(cudaFree(dr));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_sun_flare_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc==10||argc==11)&&(std::string(argv[1])=="random_fog"||std::string(argv[1])=="fog"))||
     ((argc==11||argc==12)&&std::string(argv[1])=="weather"&&(std::string(argv[2])=="random_fog"||std::string(argv[2])=="fog"))){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6;
      const int density_arg=nested?8:7,opacity_arg=nested?9:8,seed_arg=nested?10:9;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),channels=std::stoi(argv[c_arg]);
      const bool explicit_field=argc>(seed_arg+1);
      const std::size_t pixels=static_cast<std::size_t>(w)*h,n=pixels*channels;
      std::vector<float> field;
      if(explicit_field){
        field.resize(pixels);
        std::ifstream ff(argv[seed_arg+1],std::ios::binary);
        ff.read(reinterpret_cast<char*>(field.data()),static_cast<std::streamsize>(field.size()*sizeof(float)));
        if(!ff||ff.gcount()!=static_cast<std::streamsize>(field.size()*sizeof(float)))throw std::runtime_error("failed to read RandomFog field");
      }
      augmatch::RandomFogConfig config{w,h,channels,std::stof(argv[density_arg]),std::stof(argv[opacity_arg]),explicit_field?field.data():nullptr,static_cast<std::uint64_t>(std::stoull(argv[seed_arg]))};
      std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *df=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(explicit_field){ck(cudaMalloc(&df,field.size()*sizeof(float)));ck(cudaMemcpy(df,field.data(),field.size()*sizeof(float),cudaMemcpyHostToDevice));config.field=df;}
      augmatch::random_fog_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(df)ck(cudaFree(df));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_fog_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc==11)&&std::string(argv[1])=="random_shadow")||
     ((argc==12)&&std::string(argv[1])=="weather"&&std::string(argv[2])=="random_shadow")){
    try{
      const bool nested=std::string(argv[1])=="weather";
      const int input_arg=nested?3:2,output_arg=nested?4:3,w_arg=nested?5:4,h_arg=nested?6:5,c_arg=nested?7:6;
      const int opacity_arg=nested?8:7,fill_arg=nested?9:8,polygon_file_arg=nested?10:9,rectangle_file_arg=nested?11:10;
      const int w=std::stoi(argv[w_arg]),h=std::stoi(argv[h_arg]),channels=std::stoi(argv[c_arg]);
      const float opacity=std::stof(argv[opacity_arg]);
      const int fill_value=std::stoi(argv[fill_arg]);
      if(fill_value<0||fill_value>255)throw std::invalid_argument("RandomShadow fill must be in [0,255]");
      std::vector<augmatch::ShadowPoint> points;
      std::vector<augmatch::ShadowPolygon> polygons;
      if(std::string(argv[polygon_file_arg])!="-"){
        std::ifstream fp(argv[polygon_file_arg],std::ios::binary);
        if(!fp)throw std::runtime_error("failed to open RandomShadow polygon file");
        while(fp.peek()!=std::char_traits<char>::eof()){
          std::uint32_t count=0;float alpha=0.0f;
          fp.read(reinterpret_cast<char*>(&count),sizeof(count));fp.read(reinterpret_cast<char*>(&alpha),sizeof(alpha));
          if(!fp||count>1000000u)throw std::runtime_error("invalid RandomShadow polygon record");
          const std::size_t offset=points.size();points.resize(offset+count);
          fp.read(reinterpret_cast<char*>(points.data()+offset),static_cast<std::streamsize>(count*sizeof(augmatch::ShadowPoint)));
          if(!fp||count<3u)throw std::runtime_error("invalid RandomShadow polygon record");
          polygons.push_back({nullptr,static_cast<int>(count),alpha});
        }
        for(std::size_t i=0;i<polygons.size();++i) {
          std::size_t offset=0;for(std::size_t j=0;j<i;++j)offset+=static_cast<std::size_t>(polygons[j].point_count);
          polygons[i].points=points.data()+offset;
        }
      }
      std::vector<augmatch::ShadowRectangle> rectangles;
      if(std::string(argv[rectangle_file_arg])!="-"){
        std::ifstream fr(argv[rectangle_file_arg],std::ios::binary);
        if(!fr)throw std::runtime_error("failed to open RandomShadow rectangle file");
        augmatch::ShadowRectangle rectangle{};
        while(fr.read(reinterpret_cast<char*>(&rectangle),sizeof(rectangle)))rectangles.push_back(rectangle);
        if(!fr.eof())throw std::runtime_error("invalid RandomShadow rectangle file");
      }
      augmatch::RandomShadowConfig config{w,h,channels,polygons.empty()?nullptr:polygons.data(),static_cast<int>(polygons.size()),rectangles.empty()?nullptr:rectangles.data(),static_cast<int>(rectangles.size()),opacity,static_cast<std::uint8_t>(fill_value),0};
      const std::size_t n=static_cast<std::size_t>(w)*h*channels;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[input_arg],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::ShadowPoint *dpoints=nullptr;augmatch::ShadowPolygon *dpolygons=nullptr;augmatch::ShadowRectangle *drectangles=nullptr;
      ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(!points.empty()){
        ck(cudaMalloc(&dpoints,points.size()*sizeof(augmatch::ShadowPoint)));ck(cudaMemcpy(dpoints,points.data(),points.size()*sizeof(augmatch::ShadowPoint),cudaMemcpyHostToDevice));
        for(auto& polygon:polygons)polygon.points=dpoints+(polygon.points-points.data());
        ck(cudaMalloc(&dpolygons,polygons.size()*sizeof(augmatch::ShadowPolygon)));ck(cudaMemcpy(dpolygons,polygons.data(),polygons.size()*sizeof(augmatch::ShadowPolygon),cudaMemcpyHostToDevice));config.polygons=dpolygons;
      }
      if(!rectangles.empty()){ck(cudaMalloc(&drectangles,rectangles.size()*sizeof(augmatch::ShadowRectangle)));ck(cudaMemcpy(drectangles,rectangles.data(),rectangles.size()*sizeof(augmatch::ShadowRectangle),cudaMemcpyHostToDevice));config.rectangles=drectangles;}
      augmatch::random_shadow_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(drectangles)ck(cudaFree(drectangles));if(dpolygons)ck(cudaFree(dpolygons));if(dpoints)ck(cudaFree(dpoints));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_shadow_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[output_arg],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10||argc==11)&&std::string(argv[1])=="random_grid_shuffle"){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),rows=std::stoi(argv[7]),cols=std::stoi(argv[8]);
      if(rows<=0||cols<=0)throw std::invalid_argument("random grid shuffle dimensions must be positive");
      const std::size_t n=static_cast<std::size_t>(w)*h*c,cell_count=static_cast<std::size_t>(rows)*cols;
      std::vector<std::uint8_t> in(n),out(n);std::vector<std::uint32_t> permutation;
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      augmatch::RandomGridShuffleConfig cfg{w,h,c,rows,cols,nullptr,static_cast<std::uint64_t>(std::stoull(argv[9]))};
      if(argc==11){permutation.resize(cell_count);std::ifstream fp(argv[10],std::ios::binary);fp.read(reinterpret_cast<char*>(permutation.data()),cell_count*sizeof(std::uint32_t));if(!fp||fp.gcount()!=static_cast<std::streamsize>(cell_count*sizeof(std::uint32_t)))throw std::runtime_error("failed to read random grid shuffle permutation");cfg.cell_permutation=permutation.data();}
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;std::uint32_t *dp=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(cfg.cell_permutation){ck(cudaMalloc(&dp,cell_count*sizeof(std::uint32_t)));ck(cudaMemcpy(dp,permutation.data(),cell_count*sizeof(std::uint32_t),cudaMemcpyHostToDevice));cfg.cell_permutation=dp;}augmatch::random_grid_shuffle_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dp)ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_grid_shuffle_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==8||argc==9)&&std::string(argv[1])=="geometry"){ 
    try{
      const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);int ow=(op=="transpose"?h:(op=="rot90"&&((std::stoi(argv[8])%2+2)%2)?h:w));int oh=(op=="transpose"?w:(op=="rot90"&&((std::stoi(argv[8])%2+2)%2)?w:h));std::vector<std::uint8_t> in(size_t(w)*h*c),out(size_t(ow)*oh*c);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));if(op=="transpose")augmatch::transpose_u8(di,doo,{w,h,c});else if(op=="rot90")augmatch::rotate90_u8(di,doo,{w,h,c,std::stoi(argv[8])});else throw std::invalid_argument("unknown geometry operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="transpose")augmatch::transpose_u8(in.data(),out.data(),{w,h,c});else if(op=="rot90")augmatch::rotate90_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});else throw std::invalid_argument("unknown geometry operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(std::string(argv[1])=="size" && (argc==13||argc==12||argc==11) &&
     (std::string(argv[2])=="pad_multiples"||std::string(argv[2])=="crop_multiples"||std::string(argv[2])=="center_pad_multiples"||std::string(argv[2])=="center_crop_multiples"||std::string(argv[2])=="pad_powers"||std::string(argv[2])=="crop_powers"||std::string(argv[2])=="center_pad_powers"||std::string(argv[2])=="center_crop_powers"||std::string(argv[2])=="pad_aspect"||std::string(argv[2])=="crop_aspect"||std::string(argv[2])=="center_pad_aspect"||std::string(argv[2])=="center_crop_aspect"||std::string(argv[2])=="keep_size_resize")){ 
    try{
      const std::string op=argv[2]; const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);
      int ow=0,oh=0; bool keep=op=="keep_size_resize";
      augmatch::MultiplesOfConfig mc{w,h,c,0,0,0,augmatch::PadBorder::Constant,augmatch::SizeAnchor::TopLeft};
      augmatch::PowersOfConfig pc{w,h,c,0,0,augmatch::PadBorder::Constant,augmatch::SizeAnchor::TopLeft};
      augmatch::AspectRatioConfig ac{w,h,c,0.0f,augmatch::SizeRounding::Nearest,0,augmatch::PadBorder::Constant,augmatch::SizeAnchor::TopLeft};
      augmatch::KeepSizeByResizeConfig kc{w,h,c,0,0,augmatch::Interpolation::Linear};
      if(op.find("multiples")!=std::string::npos){mc.multiple_width=std::stoi(argv[8]);mc.multiple_height=std::stoi(argv[9]);mc.value=static_cast<std::uint8_t>(std::stoi(argv[10]));mc.border=static_cast<augmatch::PadBorder>(std::stoi(argv[11]));mc.anchor=static_cast<augmatch::SizeAnchor>(std::stoi(argv[12]));const auto d=op.find("pad")!=std::string::npos?augmatch::make_pad_to_multiples_of_dimensions(mc):augmatch::make_crop_to_multiples_of_dimensions(mc);ow=d.width;oh=d.height;}
      else if(op.find("powers")!=std::string::npos){pc.base=std::stoi(argv[8]);pc.value=static_cast<std::uint8_t>(std::stoi(argv[9]));pc.border=static_cast<augmatch::PadBorder>(std::stoi(argv[10]));pc.anchor=static_cast<augmatch::SizeAnchor>(std::stoi(argv[11]));const auto d=op.find("pad")!=std::string::npos?augmatch::make_pad_to_powers_of_dimensions(pc):augmatch::make_crop_to_powers_of_dimensions(pc);ow=d.width;oh=d.height;}
      else if(op.find("aspect")!=std::string::npos){ac.aspect_ratio=std::stof(argv[8]);ac.rounding=static_cast<augmatch::SizeRounding>(std::stoi(argv[9]));ac.value=static_cast<std::uint8_t>(std::stoi(argv[10]));ac.border=static_cast<augmatch::PadBorder>(std::stoi(argv[11]));ac.anchor=static_cast<augmatch::SizeAnchor>(std::stoi(argv[12]));const auto d=op.find("pad")!=std::string::npos?augmatch::make_pad_to_aspect_ratio_dimensions(ac):augmatch::make_crop_to_aspect_ratio_dimensions(ac);ow=d.width;oh=d.height;}
      else if(keep){if(argc!=11)throw std::invalid_argument("keep_size_resize expects width height channels intermediate_width intermediate_height interpolation");kc.intermediate_width=std::stoi(argv[8]);kc.intermediate_height=std::stoi(argv[9]);kc.interpolation=static_cast<augmatch::Interpolation>(std::stoi(argv[10]));ow=w;oh=h;}
      else throw std::invalid_argument("unknown catalog size operation");
      const std::size_t ni=static_cast<std::size_t>(w)*h*c,no=static_cast<std::size_t>(ow)*oh*c;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(ni));if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));
      if(keep)augmatch::keep_size_by_resize_u8(di,doo,kc);else if(op=="pad_multiples")augmatch::pad_to_multiples_of_u8(di,doo,mc);else if(op=="crop_multiples")augmatch::crop_to_multiples_of_u8(di,doo,mc);else if(op=="center_pad_multiples")augmatch::center_pad_to_multiples_of_u8(di,doo,mc);else if(op=="center_crop_multiples")augmatch::center_crop_to_multiples_of_u8(di,doo,mc);else if(op=="pad_powers")augmatch::pad_to_powers_of_u8(di,doo,pc);else if(op=="crop_powers")augmatch::crop_to_powers_of_u8(di,doo,pc);else if(op=="center_pad_powers")augmatch::center_pad_to_powers_of_u8(di,doo,pc);else if(op=="center_crop_powers")augmatch::center_crop_to_powers_of_u8(di,doo,pc);else if(op=="pad_aspect")augmatch::pad_to_aspect_ratio_u8(di,doo,ac);else if(op=="crop_aspect")augmatch::crop_to_aspect_ratio_u8(di,doo,ac);else if(op=="center_pad_aspect")augmatch::center_pad_to_aspect_ratio_u8(di,doo,ac);else if(op=="center_crop_aspect")augmatch::center_crop_to_aspect_ratio_u8(di,doo,ac);else throw std::invalid_argument("unknown catalog size operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(keep)augmatch::keep_size_by_resize_u8(in.data(),out.data(),kc);else if(op=="pad_multiples")augmatch::pad_to_multiples_of_u8(in.data(),out.data(),mc);else if(op=="crop_multiples")augmatch::crop_to_multiples_of_u8(in.data(),out.data(),mc);else if(op=="center_pad_multiples")augmatch::center_pad_to_multiples_of_u8(in.data(),out.data(),mc);else if(op=="center_crop_multiples")augmatch::center_crop_to_multiples_of_u8(in.data(),out.data(),mc);else if(op=="pad_powers")augmatch::pad_to_powers_of_u8(in.data(),out.data(),pc);else if(op=="crop_powers")augmatch::crop_to_powers_of_u8(in.data(),out.data(),pc);else if(op=="center_pad_powers")augmatch::center_pad_to_powers_of_u8(in.data(),out.data(),pc);else if(op=="center_crop_powers")augmatch::center_crop_to_powers_of_u8(in.data(),out.data(),pc);else if(op=="pad_aspect")augmatch::pad_to_aspect_ratio_u8(in.data(),out.data(),ac);else if(op=="crop_aspect")augmatch::crop_to_aspect_ratio_u8(in.data(),out.data(),ac);else if(op=="center_pad_aspect")augmatch::center_pad_to_aspect_ratio_u8(in.data(),out.data(),ac);else if(op=="center_crop_aspect")augmatch::center_crop_to_aspect_ratio_u8(in.data(),out.data(),ac);else throw std::invalid_argument("unknown catalog size operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(no));return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10||argc==12)&&std::string(argv[1])=="size"){ 
    try{
      const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]),ow=std::stoi(argv[8]),oh=std::stoi(argv[9]);std::vector<std::uint8_t> in(size_t(w)*h*c),out(size_t(ow)*oh*c);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));if(op=="center_crop")augmatch::center_crop_u8(di,doo,{w,h,ow,oh,c});else if(op=="pad"||op=="square_pad"){augmatch::PadConfig pc{w,h,ow,oh,c,static_cast<std::uint8_t>(std::stoi(argv[10])),static_cast<augmatch::PadBorder>(std::stoi(argv[11]))};if(op=="square_pad")augmatch::square_symmetric_pad_u8(di,doo,pc);else augmatch::pad_to_size_u8(di,doo,pc);}else throw std::invalid_argument("unknown size operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="center_crop")augmatch::center_crop_u8(in.data(),out.data(),{w,h,ow,oh,c});else if(op=="pad"||op=="square_pad"){augmatch::PadConfig pc{w,h,ow,oh,c,static_cast<std::uint8_t>(std::stoi(argv[10])),static_cast<augmatch::PadBorder>(std::stoi(argv[11]))};if(op=="square_pad")augmatch::square_symmetric_pad_u8(in.data(),out.data(),pc);else augmatch::pad_to_size_u8(in.data(),out.data(),pc);}else throw std::invalid_argument("unknown size operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(std::string(argv[1])=="arithmetic"&&argc>=9&&argc<=13&&
     (std::string(argv[2])=="salt"||std::string(argv[2])=="pepper"||std::string(argv[2])=="salt_and_pepper"||
      std::string(argv[2])=="coarse_salt"||std::string(argv[2])=="coarse_pepper"||std::string(argv[2])=="coarse_salt_and_pepper")){
    try{
      const std::string op=argv[2]; const bool coarse=op.rfind("coarse_",0)==0;
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]); const std::size_t n=static_cast<std::size_t>(w)*h*c, pixels=static_cast<std::size_t>(w)*h;
      std::vector<std::uint8_t> in(n),out(n),mask; std::vector<augmatch::SaltPepperRectangle> rectangles;
      std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      augmatch::SaltPepperConfig cfg; cfg.width=w;cfg.height=h;cfg.channels=c;cfg.probability=0.0f;cfg.salt_probability=0.5f;cfg.seed=0;
      if(argc==9){
        if(coarse) rectangles=parse_salt_rectangles(argv[8],w,h); else {mask.resize(pixels);std::ifstream fm(argv[8],std::ios::binary);fm.read(reinterpret_cast<char*>(mask.data()),static_cast<std::streamsize>(pixels));if(!fm||fm.gcount()!=static_cast<std::streamsize>(pixels))throw std::runtime_error("failed to read salt and pepper mask");}
      } else if(!coarse&&argc==10) {cfg.probability=std::stof(argv[8]);cfg.seed=static_cast<std::uint64_t>(std::stoull(argv[9]));}
      else if(!coarse&&argc==11) {cfg.probability=std::stof(argv[8]);cfg.salt_probability=std::stof(argv[9]);cfg.seed=static_cast<std::uint64_t>(std::stoull(argv[10]));}
      else if(coarse&&argc==11) {rectangles=parse_salt_rectangles(argv[8],w,h);cfg.block_width=std::stoi(argv[9]);cfg.block_height=std::stoi(argv[10]);}
      else if(coarse&&argc==12) {cfg.probability=std::stof(argv[8]);cfg.seed=static_cast<std::uint64_t>(std::stoull(argv[9]));cfg.block_width=std::stoi(argv[10]);cfg.block_height=std::stoi(argv[11]);}
      else if(coarse&&argc==13) {cfg.probability=std::stof(argv[8]);cfg.salt_probability=std::stof(argv[9]);cfg.seed=static_cast<std::uint64_t>(std::stoull(argv[10]));cfg.block_width=std::stoi(argv[11]);cfg.block_height=std::stoi(argv[12]);}
      else throw std::invalid_argument("invalid salt and pepper CLI form");
      cfg.mask=mask.empty()?nullptr:mask.data();cfg.rectangles=rectangles.empty()?nullptr:rectangles.data();cfg.rectangle_count=static_cast<int>(rectangles.size());
      if(op=="salt"||op=="coarse_salt")cfg.salt_probability=1.0f; else if(op=="pepper"||op=="coarse_pepper")cfg.salt_probability=0.0f;
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr,*dm=nullptr;augmatch::SaltPepperRectangle* dr=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(!mask.empty()){ck(cudaMalloc(&dm,pixels));ck(cudaMemcpy(dm,mask.data(),pixels,cudaMemcpyHostToDevice));cfg.mask=dm;} if(!rectangles.empty()){ck(cudaMalloc(&dr,rectangles.size()*sizeof(*dr)));ck(cudaMemcpy(dr,rectangles.data(),rectangles.size()*sizeof(*dr),cudaMemcpyHostToDevice));cfg.rectangles=dr;}
      if(op=="salt")augmatch::salt_u8(di,doo,cfg);else if(op=="pepper")augmatch::pepper_u8(di,doo,cfg);else if(op=="salt_and_pepper")augmatch::salt_and_pepper_u8(di,doo,cfg);else if(op=="coarse_salt")augmatch::coarse_salt_u8(di,doo,cfg);else if(op=="coarse_pepper")augmatch::coarse_pepper_u8(di,doo,cfg);else augmatch::coarse_salt_and_pepper_u8(di,doo,cfg);
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dr)ck(cudaFree(dr));if(dm)ck(cudaFree(dm));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="salt")augmatch::salt_u8(in.data(),out.data(),cfg);else if(op=="pepper")augmatch::pepper_u8(in.data(),out.data(),cfg);else if(op=="salt_and_pepper")augmatch::salt_and_pepper_u8(in.data(),out.data(),cfg);else if(op=="coarse_salt")augmatch::coarse_salt_u8(in.data(),out.data(),cfg);else if(op=="coarse_pepper")augmatch::coarse_pepper_u8(in.data(),out.data(),cfg);else augmatch::coarse_salt_and_pepper_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10||argc==11)&&std::string(argv[1])=="arithmetic"&&(std::string(argv[2])=="replace_elementwise"||std::string(argv[2])=="impulse_noise")){ 
    try{
      const bool is_replace=std::string(argv[2])=="replace_elementwise";const bool explicit_arrays=argc==10;
      const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(w)*h*c;
      std::vector<std::uint8_t> in(n),out(n),mask,values;std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      float probability=0.0f,salt_probability=0.5f;std::uint8_t replacement=0;std::uint64_t seed=0;
      if(explicit_arrays){mask.resize(n);values.resize(n);std::ifstream fm(argv[8],std::ios::binary),fv(argv[9],std::ios::binary);fm.read(reinterpret_cast<char*>(mask.data()),static_cast<std::streamsize>(n));fv.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(n));if(!fm||fm.gcount()!=static_cast<std::streamsize>(n)||!fv||fv.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read replacement mask/value arrays");}
      else if(is_replace){probability=std::stof(argv[8]);const int value=std::stoi(argv[9]);if(value<0||value>255)throw std::invalid_argument("replacement value must be in [0,255]");replacement=static_cast<std::uint8_t>(value);seed=static_cast<std::uint64_t>(std::stoull(argv[10]));}
      else {probability=std::stof(argv[8]);salt_probability=std::stof(argv[9]);seed=static_cast<std::uint64_t>(std::stoull(argv[10]));}
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr,*dm=nullptr,*dv=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(explicit_arrays){ck(cudaMalloc(&dm,n));ck(cudaMalloc(&dv,n));ck(cudaMemcpy(dm,mask.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,values.data(),n,cudaMemcpyHostToDevice));}
      if(is_replace)augmatch::replace_elementwise_u8(di,doo,{w,h,c,dm,dv,replacement,probability,seed});else augmatch::impulse_noise_u8(di,doo,{w,h,c,dm,dv,probability,salt_probability,seed});ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dv)ck(cudaFree(dv));if(dm)ck(cudaFree(dm));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(is_replace)augmatch::replace_elementwise_u8(in.data(),out.data(),{w,h,c,explicit_arrays?mask.data():nullptr,explicit_arrays?values.data():nullptr,replacement,probability,seed});else augmatch::impulse_noise_u8(in.data(),out.data(),{w,h,c,explicit_arrays?mask.data():nullptr,explicit_arrays?values.data():nullptr,probability,salt_probability,seed});
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==9||argc==11||argc==12)&&std::string(argv[1])=="arithmetic"&&(std::string(argv[2])=="add_elementwise"||std::string(argv[2])=="multiply_elementwise")){ 
    try{
      const bool is_add=std::string(argv[2])=="add_elementwise";const int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;const bool explicit_values=argc==9||argc==12;const int values_arg=argc==9?8:11;
      std::vector<float> values;if(explicit_values){values.resize(n);std::ifstream fv(argv[values_arg],std::ios::binary);fv.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fv||fv.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read elementwise values");}
      const float default_value=is_add?0.0f:1.0f;const float min_value=explicit_values?default_value:std::stof(argv[8]);const float max_value=explicit_values?default_value:std::stof(argv[9]);const std::uint64_t seed=explicit_values?(argc==12?static_cast<std::uint64_t>(std::stoull(argv[10])):0):static_cast<std::uint64_t>(std::stoull(argv[10]));
      std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *dv=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(explicit_values){ck(cudaMalloc(&dv,n*sizeof(float)));ck(cudaMemcpy(dv,values.data(),n*sizeof(float),cudaMemcpyHostToDevice));}
      if(is_add)augmatch::add_elementwise_u8(di,doo,{w,h,c,dv,min_value,max_value,seed});else augmatch::multiply_elementwise_u8(di,doo,{w,h,c,dv,min_value,max_value,seed});
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(dv)ck(cudaFree(dv));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(is_add)augmatch::add_elementwise_u8(in.data(),out.data(),{w,h,c,explicit_values?values.data():nullptr,min_value,max_value,seed});else augmatch::multiply_elementwise_u8(in.data(),out.data(),{w,h,c,explicit_values?values.data():nullptr,min_value,max_value,seed});
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc>=8&&argc<=11)&&std::string(argv[1])=="arithmetic"){ 
    try{
      const std::string op=argv[2]; int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);
      std::vector<std::uint8_t> in(size_t(w)*h*c),out(in.size()); std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));
      if(op=="add")augmatch::add_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="multiply")augmatch::multiply_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="posterize")augmatch::posterize_u8(di,doo,{w,h,c,std::stoi(argv[8])});else if(op=="gamma")augmatch::gamma_u8(di,doo,{w,h,c,std::stof(argv[8])});else if(op=="solarize")augmatch::solarize_u8(di,doo,{w,h,c,std::stoi(argv[8])});else if(op=="invert")augmatch::invert_u8(di,doo,w,h,c);else if(op=="gaussian_noise")augmatch::gaussian_noise_u8(di,doo,{w,h,c,std::stof(argv[8]),static_cast<std::uint64_t>(std::stoull(argv[9]))});else if(op=="salt_pepper")augmatch::salt_pepper_u8(di,doo,{w,h,c,std::stof(argv[8]),std::stof(argv[9]),static_cast<std::uint64_t>(std::stoull(argv[10]))});else throw std::invalid_argument("unknown arithmetic operation");
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="add")augmatch::add_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="multiply")augmatch::multiply_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="posterize")augmatch::posterize_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});else if(op=="gamma")augmatch::gamma_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});else if(op=="solarize")augmatch::solarize_u8(in.data(),out.data(),{w,h,c,std::stoi(argv[8])});else if(op=="invert")augmatch::invert_u8(in.data(),out.data(),w,h,c);else if(op=="gaussian_noise")augmatch::gaussian_noise_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),static_cast<std::uint64_t>(std::stoull(argv[9]))});else if(op=="salt_pepper")augmatch::salt_pepper_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),std::stof(argv[9]),static_cast<std::uint64_t>(std::stoull(argv[10]))});else throw std::invalid_argument("unknown arithmetic operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="bilateral_blur"){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),radius=std::stoi(argv[7]);
      const augmatch::BilateralBlurConfig config{w,h,c,radius,std::stof(argv[8]),std::stof(argv[9])};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::bilateral_blur_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::bilateral_blur_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="mean_shift_blur"){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      const augmatch::MeanShiftBlurConfig config{w,h,c,std::stoi(argv[7]),std::stof(argv[8]),std::stoi(argv[9])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::mean_shift_blur_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::mean_shift_blur_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="blur"){ 
    try{
      int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),k=std::stoi(argv[7]); float sigma=std::stof(argv[8]); int mode=std::stoi(argv[9]);
      std::vector<std::uint8_t> in(size_t(w)*h*c),out(in.size()); std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
      augmatch::BlurConfig config{w,h,c,k,sigma,static_cast<augmatch::BlurMode>(mode)};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));augmatch::blur_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::blur_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&(std::string(argv[1])=="non_local_means"||std::string(argv[1])=="non_local_means_denoising")){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      augmatch::NonLocalMeansDenoisingConfig config{w,h,c,std::stoi(argv[7]),std::stoi(argv[8]),std::stof(argv[9])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::non_local_means_denoising_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::non_local_means_denoising_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="superpixels"){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      augmatch::SuperpixelConfig config{w,h,c,std::stoi(argv[7]),std::stoi(argv[8])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::superpixels_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::superpixels_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==8||argc==9)&& (std::string(argv[1])=="voronoi"||std::string(argv[1])=="uniform_voronoi"||std::string(argv[1])=="regular_grid_voronoi"||std::string(argv[1])=="relative_regular_grid_voronoi")){ 
    try{
      const std::string op=argv[1]; const int w=std::stoi(argv[4]),h=std::stoi(argv[5]); const std::size_t n=static_cast<std::size_t>(w)*h; std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[2],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n)); if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)) throw std::runtime_error("failed to read Voronoi mask");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr; ck(cudaMalloc(&di,n)); ck(cudaMalloc(&doo,n)); ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(op=="voronoi"){ augmatch::VoronoiConfig c{w,h,std::stoi(argv[6]),static_cast<std::uint64_t>(std::stoull(argv[7])),nullptr}; augmatch::voronoi_u8(di,doo,c); }
      else if(op=="uniform_voronoi"){ augmatch::UniformVoronoiConfig c{w,h,std::stoi(argv[6]),static_cast<std::uint64_t>(std::stoull(argv[7]))}; augmatch::uniform_voronoi_u8(di,doo,c); }
      else if(op=="regular_grid_voronoi"){ if(argc!=8) throw std::invalid_argument("regular_grid_voronoi IN OUT W H GRID_W GRID_H"); augmatch::RegularGridVoronoiConfig c{w,h,std::stoi(argv[6]),std::stoi(argv[7])}; augmatch::regular_grid_voronoi_u8(di,doo,c); }
      else { if(argc!=8) throw std::invalid_argument("relative_regular_grid_voronoi IN OUT W H REL_W REL_H"); augmatch::RelativeRegularGridVoronoiConfig c{w,h,std::stof(argv[6]),std::stof(argv[7])}; augmatch::relative_regular_grid_voronoi_u8(di,doo,c); }
      ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost)); ck(cudaFree(di)); ck(cudaFree(doo));
#else
      if(op=="voronoi") augmatch::voronoi_u8(in.data(),out.data(),{w,h,std::stoi(argv[6]),static_cast<std::uint64_t>(std::stoull(argv[7])),nullptr});
      else if(op=="uniform_voronoi") augmatch::uniform_voronoi_u8(in.data(),out.data(),{w,h,std::stoi(argv[6]),static_cast<std::uint64_t>(std::stoull(argv[7]))});
      else if(op=="regular_grid_voronoi"){ if(argc!=8) throw std::invalid_argument("regular_grid_voronoi IN OUT W H GRID_W GRID_H"); augmatch::regular_grid_voronoi_u8(in.data(),out.data(),{w,h,std::stoi(argv[6]),std::stoi(argv[7])}); }
      else { if(argc!=8) throw std::invalid_argument("relative_regular_grid_voronoi IN OUT W H REL_W REL_H"); augmatch::relative_regular_grid_voronoi_u8(in.data(),out.data(),{w,h,std::stof(argv[6]),std::stof(argv[7])}); }
#endif
      std::ofstream fo(argv[3],std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n)); if(!fo) throw std::runtime_error("failed to write Voronoi output"); return 0;
    }catch(const std::exception& e){ std::cerr<<e.what()<<'\n'; return 1; }
  }
  if(argc==10&&std::string(argv[1])=="zoom_blur"){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),steps=std::stoi(argv[9]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      augmatch::ZoomBlurConfig config{w,h,c,std::stof(argv[7]),std::stof(argv[8]),steps};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::zoom_blur_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::zoom_blur_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==11||argc==12)&&std::string(argv[1])=="glass_blur"){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),max_delta=std::stoi(argv[8]),iterations=std::stoi(argv[9]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c,count=augmatch::glass_blur_swap_count(w,h,max_delta,iterations);std::vector<std::uint8_t> in(n),out(n);std::vector<augmatch::GlassBlurSwap> swaps;
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      augmatch::GlassBlurConfig config{w,h,c,std::stof(argv[7]),max_delta,iterations,nullptr,static_cast<std::uint64_t>(std::stoull(argv[10]))};
      if(argc==12){swaps.resize(count);std::ifstream fs(argv[11],std::ios::binary);fs.read(reinterpret_cast<char*>(swaps.data()),static_cast<std::streamsize>(count*sizeof(augmatch::GlassBlurSwap)));if(!fs||fs.gcount()!=static_cast<std::streamsize>(count*sizeof(augmatch::GlassBlurSwap)))throw std::runtime_error("failed to read glass blur swap sequence");config.swap_sequence=swaps.data();}
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;augmatch::GlassBlurSwap* ds=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(config.swap_sequence){ck(cudaMalloc(&ds,count*sizeof(augmatch::GlassBlurSwap)));ck(cudaMemcpy(ds,swaps.data(),count*sizeof(augmatch::GlassBlurSwap),cudaMemcpyHostToDevice));config.swap_sequence=ds;}augmatch::glass_blur_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));if(ds)ck(cudaFree(ds));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::glass_blur_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="ringing_overshoot"){
    try{
      int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),k=std::stoi(argv[7]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
      augmatch::RingingOvershootConfig config{w,h,c,k,std::stof(argv[8]),std::stof(argv[9])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::ringing_overshoot_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::ringing_overshoot_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="advanced_blur"){ 
    try{
      int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),k=std::stoi(argv[7]); float sx=std::stof(argv[8]),sy=std::stof(argv[9]),angle=std::stof(argv[10]);
      std::vector<std::uint8_t> in(size_t(w)*h*c),out(in.size()); std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
      augmatch::AdvancedBlurConfig config{w,h,c,k,sx,sy,angle};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));augmatch::advanced_blur_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::advanced_blur_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==8||argc==9||argc==10)&&std::string(argv[1])=="pixel"){ 
    try{
      const std::string op=argv[2]; int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]);
      std::vector<std::uint8_t> in(size_t(w)*h*c),out(in.size());
      std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));
      if(op=="gamma")augmatch::gamma_u8(di,doo,{w,h,c,std::stof(argv[8])});
      else if(op=="brightness_contrast"){if(argc!=10)throw std::invalid_argument("brightness_contrast needs two values");augmatch::brightness_contrast_u8(di,doo,{w,h,c,std::stof(argv[8]),std::stof(argv[9])});}
      else if(op=="grayscale")augmatch::grayscale_u8(di,doo,{w,h,c});else throw std::invalid_argument("unknown pixel operation");
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="gamma")augmatch::gamma_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8])});
      else if(op=="brightness_contrast"){if(argc!=10)throw std::invalid_argument("brightness_contrast needs two values");augmatch::brightness_contrast_u8(in.data(),out.data(),{w,h,c,std::stof(argv[8]),std::stof(argv[9])});}
      else if(op=="grayscale")augmatch::grayscale_u8(in.data(),out.data(),{w,h,c});else throw std::invalid_argument("unknown pixel operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="shift_scale_rotate"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::ShiftScaleRotateConfig cfg{w,h,c,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[11]))};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::shift_scale_rotate_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::shift_scale_rotate_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="diffraction_blur"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::DiffractionBlurConfig cfg{w,h,c,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::diffraction_blur_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::diffraction_blur_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="bokeh_blur"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::BokehBlurConfig cfg{w,h,c,std::stoi(argv[7]),std::stof(argv[8])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::bokeh_blur_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::bokeh_blur_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="cat_eye_bokeh"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::CatEyeBokehConfig cfg{w,h,c,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::cat_eye_bokeh_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::cat_eye_bokeh_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="aperture_shape_blur"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::ApertureShapeBlurConfig cfg{w,h,c,std::stoi(argv[7]),std::stoi(argv[8]),std::stof(argv[9]),std::stof(argv[10])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::aperture_shape_blur_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::aperture_shape_blur_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&std::string(argv[1])=="defocus"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::DefocusConfig cfg{w,h,c,std::stoi(argv[7]),std::stof(argv[8])};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::defocus_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::defocus_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&(std::string(argv[1])=="depth_defocus"||std::string(argv[1])=="depth_dependent_defocus")){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);std::size_t n=static_cast<std::size_t>(w)*h*c,fn=static_cast<std::size_t>(w)*h;std::vector<std::uint8_t> in(n),out(n);std::vector<float> depth(fn);std::ifstream fi(argv[2],std::ios::binary),fd(argv[7],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);fd.read(reinterpret_cast<char*>(depth.data()),fn*sizeof(float));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!fd||fd.gcount()!=static_cast<std::streamsize>(fn*sizeof(float)))throw std::runtime_error("failed to read depth-defocus input");augmatch::DepthDependentDefocusConfig cfg{w,h,c,depth.data(),std::stof(argv[8]),std::stof(argv[9]),std::stoi(argv[10])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *dd=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMalloc(&dd,fn*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dd,depth.data(),fn*sizeof(float),cudaMemcpyHostToDevice));cfg.depth=dd;augmatch::depth_dependent_defocus_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));ck(cudaFree(dd));
#else
      augmatch::depth_dependent_defocus_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&(std::string(argv[1])=="camera_shake"||std::string(argv[1])=="camera_shake_blur"||std::string(argv[1])=="rolling_shutter"||std::string(argv[1])=="rolling_shutter_motion")){
    try{const bool rolling=std::string(argv[1])=="rolling_shutter"||std::string(argv[1])=="rolling_shutter_motion";int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),count=std::stoi(argv[9]);std::size_t n=static_cast<std::size_t>(w)*h*c,fn=rolling?static_cast<std::size_t>(h):static_cast<std::size_t>(count);std::vector<std::uint8_t> in(n),out(n);std::vector<float> ax(fn),ay(fn);std::ifstream fi(argv[2],std::ios::binary),fx(argv[7],std::ios::binary),fy(argv[8],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);fx.read(reinterpret_cast<char*>(ax.data()),fn*sizeof(float));fy.read(reinterpret_cast<char*>(ay.data()),fn*sizeof(float));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!fx||fx.gcount()!=static_cast<std::streamsize>(fn*sizeof(float))||!fy||fy.gcount()!=static_cast<std::streamsize>(fn*sizeof(float)))throw std::runtime_error("failed to read motion field input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *dx=nullptr,*dy=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMalloc(&dx,fn*sizeof(float)));ck(cudaMalloc(&dy,fn*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dx,ax.data(),fn*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dy,ay.data(),fn*sizeof(float),cudaMemcpyHostToDevice));if(rolling){augmatch::RollingShutterMotionBlurConfig cfg{w,h,c,count,dx,dy};augmatch::rolling_shutter_motion_blur_u8(di,doo,cfg);}else{augmatch::CameraShakeBlurConfig cfg{w,h,c,count,dx,dy};augmatch::camera_shake_blur_u8(di,doo,cfg);}ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));ck(cudaFree(dx));ck(cudaFree(dy));
#else
      if(rolling)augmatch::rolling_shutter_motion_blur_u8(in.data(),out.data(),{w,h,c,count,ax.data(),ay.data()});else augmatch::camera_shake_blur_u8(in.data(),out.data(),{w,h,c,count,ax.data(),ay.data()});
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&(std::string(argv[1])=="linear_directional"||std::string(argv[1])=="linear_directional_blur")){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::LinearDirectionalBlurConfig cfg{w,h,c,std::stoi(argv[9]),std::stof(argv[7]),std::stof(argv[8])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::linear_directional_blur_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::linear_directional_blur_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&(std::string(argv[1])=="rotational_motion"||std::string(argv[1])=="rotational_motion_blur")){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");augmatch::RotationalMotionBlurConfig cfg{w,h,c,std::stoi(argv[10]),std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9])};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::rotational_motion_blur_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::rotational_motion_blur_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==9&&(std::string(argv[1])=="rotate"||std::string(argv[1])=="safe_rotate")){  
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::RotateConfig cfg{w,h,c,std::stof(argv[7]),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[8]))};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(std::string(argv[1])=="safe_rotate")augmatch::safe_rotate_u8(di,doo,{w,h,c,std::stof(argv[7]),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[8]))});else augmatch::rotate_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(std::string(argv[1])=="safe_rotate")augmatch::safe_rotate_u8(in.data(),out.data(),{w,h,c,std::stof(argv[7]),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[8]))});else augmatch::rotate_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="elastic_transform"){
    try{
      int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c,fn=static_cast<std::size_t>(w)*h;
      std::vector<std::uint8_t> in(n),out(n);std::vector<float> dx(fn),dy(fn);
      std::ifstream fi(argv[2],std::ios::binary),fx(argv[7],std::ios::binary),fy(argv[8],std::ios::binary);
      fi.read(reinterpret_cast<char*>(in.data()),n);fx.read(reinterpret_cast<char*>(dx.data()),fn*sizeof(float));fy.read(reinterpret_cast<char*>(dy.data()),fn*sizeof(float));
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!fx||fx.gcount()!=static_cast<std::streamsize>(fn*sizeof(float))||!fy||fy.gcount()!=static_cast<std::streamsize>(fn*sizeof(float)))throw std::runtime_error("failed to read elastic transform input");
      augmatch::ElasticTransformConfig cfg{w,h,c,dx.data(),dy.data(),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[9]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *ddx=nullptr,*ddy=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMalloc(&ddx,fn*sizeof(float)));ck(cudaMalloc(&ddy,fn*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(ddx,dx.data(),fn*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(ddy,dy.data(),fn*sizeof(float),cudaMemcpyHostToDevice));cfg.displacement_x=ddx;cfg.displacement_y=ddy;augmatch::elastic_transform_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));ck(cudaFree(ddx));ck(cudaFree(ddy));
#else
      augmatch::elastic_transform_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="grid_distortion"){
    try{
      int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),gw=std::stoi(argv[7]),gh=std::stoi(argv[8]);
      if(gw<2||gh<2)throw std::invalid_argument("grid dimensions must be at least 2");
      const std::size_t n=static_cast<std::size_t>(w)*h*c,gn=static_cast<std::size_t>(gw)*gh;std::vector<std::uint8_t> in(n),out(n);std::vector<float> dx(gn),dy(gn);
      std::ifstream fi(argv[2],std::ios::binary),fx(argv[9],std::ios::binary),fy(argv[10],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);fx.read(reinterpret_cast<char*>(dx.data()),gn*sizeof(float));fy.read(reinterpret_cast<char*>(dy.data()),gn*sizeof(float));
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!fx||fx.gcount()!=static_cast<std::streamsize>(gn*sizeof(float))||!fy||fy.gcount()!=static_cast<std::streamsize>(gn*sizeof(float)))throw std::runtime_error("failed to read grid distortion input");
      augmatch::GridDistortionConfig cfg{w,h,c,gw,gh,dx.data(),dy.data(),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[11]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *ddx=nullptr,*ddy=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMalloc(&ddx,gn*sizeof(float)));ck(cudaMalloc(&ddy,gn*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(ddx,dx.data(),gn*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(ddy,dy.data(),gn*sizeof(float),cudaMemcpyHostToDevice));cfg.displacement_x=ddx;cfg.displacement_y=ddy;augmatch::grid_distortion_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));ck(cudaFree(ddx));ck(cudaFree(ddy));
#else
      augmatch::grid_distortion_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="piecewise_affine"){
    try{
      int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),gw=std::stoi(argv[7]),gh=std::stoi(argv[8]);
      if(gw<2||gh<2)throw std::invalid_argument("piecewise affine grid dimensions must be at least 2");
      const std::size_t n=static_cast<std::size_t>(w)*h*c,gn=static_cast<std::size_t>(gw)*gh;std::vector<std::uint8_t> in(n),out(n);std::vector<float> dx(gn),dy(gn);
      std::ifstream fi(argv[2],std::ios::binary),fx(argv[9],std::ios::binary),fy(argv[10],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);fx.read(reinterpret_cast<char*>(dx.data()),gn*sizeof(float));fy.read(reinterpret_cast<char*>(dy.data()),gn*sizeof(float));
      if(!fi||fi.gcount()!=static_cast<std::streamsize>(n)||!fx||fx.gcount()!=static_cast<std::streamsize>(gn*sizeof(float))||!fy||fy.gcount()!=static_cast<std::streamsize>(gn*sizeof(float)))throw std::runtime_error("failed to read piecewise affine input");
      augmatch::PiecewiseAffineConfig cfg{w,h,c,gw,gh,dx.data(),dy.data(),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[11]))};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;float *ddx=nullptr,*ddy=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMalloc(&ddx,gn*sizeof(float)));ck(cudaMalloc(&ddy,gn*sizeof(float)));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(ddx,dx.data(),gn*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(ddy,dy.data(),gn*sizeof(float),cudaMemcpyHostToDevice));cfg.displacement_x=ddx;cfg.displacement_y=ddy;augmatch::piecewise_affine_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));ck(cudaFree(ddx));ck(cudaFree(ddy));
#else
      augmatch::piecewise_affine_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==12||argc==14)&&std::string(argv[1])=="focus_breathing"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::FocusBreathingConfig cfg;cfg.width=w;cfg.height=h;cfg.channels=c;cfg.focus_position=std::stof(argv[7]);cfg.breathing_strength=std::stof(argv[8]);cfg.radial_strength=std::stof(argv[9]);cfg.center_x=std::stof(argv[10]);cfg.center_y=std::stof(argv[11]);if(argc==14){cfg.interpolation=static_cast<augmatch::Interpolation>(std::stoi(argv[12]));cfg.fill=static_cast<std::uint8_t>(std::stoi(argv[13]));}std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::focus_breathing_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::focus_breathing_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="optical_distortion"){ 
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::OpticalDistortionConfig cfg{w,h,c,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),augmatch::Interpolation::Linear,static_cast<std::uint8_t>(std::stoi(argv[11]))};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::optical_distortion_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::optical_distortion_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==16&&(std::string(argv[1])=="perspective"||std::string(argv[1])=="perspective_transform")){ 
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::PerspectiveConfig cfg{w,h,c,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11]),std::stof(argv[12]),std::stof(argv[13]),std::stof(argv[14]),std::stof(argv[15]),augmatch::Interpolation::Linear,0};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));if(std::string(argv[1])=="perspective_transform")augmatch::perspective_transform_u8(di,doo,cfg);else augmatch::perspective_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(std::string(argv[1])=="perspective_transform")augmatch::perspective_transform_u8(in.data(),out.data(),cfg);else augmatch::perspective_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==15&&std::string(argv[1])=="affine"){ 
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),ow=std::stoi(argv[7]),oh=std::stoi(argv[8]);augmatch::AffineConfig cfg{w,h,ow,oh,c,std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11]),std::stof(argv[12]),std::stof(argv[13]),std::stof(argv[14]),augmatch::Interpolation::Linear,0};std::size_t ni=static_cast<std::size_t>(w)*h*c,no=static_cast<std::size_t>(ow)*oh*c;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::affine_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::affine_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="maxsize"){ 
    try{const std::string op=argv[2];int w=std::stoi(argv[5]),h=std::stoi(argv[6]),c=std::stoi(argv[7]),m=std::stoi(argv[8]),ow=std::stoi(argv[9]),oh=std::stoi(argv[10]);std::size_t ni=static_cast<std::size_t>(w)*h*c,no=static_cast<std::size_t>(ow)*oh*c;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[3],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");augmatch::MaxSizeConfig cfg{w,h,ow,oh,m,c,augmatch::Interpolation::Linear};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));if(op=="longest")augmatch::longest_max_size_u8(di,doo,cfg);else if(op=="smallest")augmatch::smallest_max_size_u8(di,doo,cfg);else throw std::invalid_argument("unknown maxsize operation");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(op=="longest")augmatch::longest_max_size_u8(in.data(),out.data(),cfg);else if(op=="smallest")augmatch::smallest_max_size_u8(in.data(),out.data(),cfg);else throw std::invalid_argument("unknown maxsize operation");
#endif
      std::ofstream fo(argv[4],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==12||argc==16)&&(std::string(argv[1])=="crop_from_borders"||std::string(argv[1])=="random_crop_from_borders")){
    try{
      augmatch::RandomCropFromBordersConfig cfg;cfg.input_width=std::stoi(argv[4]);cfg.input_height=std::stoi(argv[5]);cfg.channels=std::stoi(argv[6]);cfg.crop_left=std::stof(argv[7]);cfg.crop_right=std::stof(argv[8]);cfg.crop_top=std::stof(argv[9]);cfg.crop_bottom=std::stof(argv[10]);cfg.seed=std::stoull(argv[11]);
      if(argc==16){cfg.left_offset=std::stoi(argv[12]);cfg.right_offset=std::stoi(argv[13]);cfg.top_offset=std::stoi(argv[14]);cfg.bottom_offset=std::stoi(argv[15]);}
      const auto offsets=augmatch::make_random_crop_from_borders_offsets(cfg);const int ow=cfg.input_width-offsets.left-offsets.right,oh=cfg.input_height-offsets.top-offsets.bottom;const std::size_t ni=static_cast<std::size_t>(cfg.input_width)*cfg.input_height*cfg.channels,no=static_cast<std::size_t>(ow)*oh*cfg.channels;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::random_crop_from_borders_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_crop_from_borders_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="crop"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),ow=std::stoi(argv[7]),oh=std::stoi(argv[8]),left=std::stoi(argv[9]),top=std::stoi(argv[10]);std::size_t ni=static_cast<std::size_t>(w)*h*c,no=static_cast<std::size_t>(ow)*oh*c;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");augmatch::RandomCropConfig cfg{w,h,ow,oh,c,left,top};
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::random_crop_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_crop_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==15||argc==16)&&std::string(argv[1])=="crop_non_empty_mask_if_exists"){
    try{
      augmatch::CropNonEmptyMaskIfExistsConfig cfg; cfg.input_width=std::stoi(argv[6]); cfg.input_height=std::stoi(argv[7]); cfg.channels=std::stoi(argv[8]); cfg.mask_channels=std::stoi(argv[9]); cfg.crop_width=std::stoi(argv[10]); cfg.crop_height=std::stoi(argv[11]); cfg.seed=std::stoull(argv[12]); cfg.fallback_x=std::stoi(argv[13]); cfg.fallback_y=std::stoi(argv[14]); if(argc==16) cfg.mask_threshold=static_cast<std::uint8_t>(std::stoi(argv[15]));
      const std::size_t ni=static_cast<std::size_t>(cfg.input_width)*cfg.input_height*cfg.channels, nm=static_cast<std::size_t>(cfg.input_width)*cfg.input_height*cfg.mask_channels, no=static_cast<std::size_t>(cfg.crop_width)*cfg.crop_height*cfg.channels, nom=static_cast<std::size_t>(cfg.crop_width)*cfg.crop_height*cfg.mask_channels;
      std::vector<std::uint8_t> in(ni),mask(nm),out(no),out_mask(nom); std::ifstream fi(argv[2],std::ios::binary),fm(argv[3],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(ni)); fm.read(reinterpret_cast<char*>(mask.data()),static_cast<std::streamsize>(nm)); if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni)||!fm||fm.gcount()!=static_cast<std::streamsize>(nm)) throw std::runtime_error("failed to read CropNonEmptyMaskIfExists input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*dm=nullptr,*doo=nullptr,*dom=nullptr; ck(cudaMalloc(&di,ni)); ck(cudaMalloc(&dm,nm)); ck(cudaMalloc(&doo,no)); ck(cudaMalloc(&dom,nom)); ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice)); ck(cudaMemcpy(dm,mask.data(),nm,cudaMemcpyHostToDevice)); augmatch::crop_non_empty_mask_if_exists_u8(di,dm,doo,dom,cfg); ck(cudaDeviceSynchronize()); ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost)); ck(cudaMemcpy(out_mask.data(),dom,nom,cudaMemcpyDeviceToHost)); ck(cudaFree(di)); ck(cudaFree(dm)); ck(cudaFree(doo)); ck(cudaFree(dom));
#else
      augmatch::crop_non_empty_mask_if_exists_u8(in.data(),mask.data(),out.data(),out_mask.data(),cfg);
#endif
      std::ofstream fo(argv[4],std::ios::binary),fmo(argv[5],std::ios::binary); fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(no)); fmo.write(reinterpret_cast<const char*>(out_mask.data()),static_cast<std::streamsize>(nom)); if(!fo||!fmo) throw std::runtime_error("failed to write CropNonEmptyMaskIfExists output"); return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==14&&std::string(argv[1])=="bbox_safe_random_crop"){
    try{augmatch::BBoxSafeRandomCropConfig cfg;cfg.input_width=std::stoi(argv[4]);cfg.input_height=std::stoi(argv[5]);cfg.channels=std::stoi(argv[6]);auto boxes=read_target_boxes(argv[7]);cfg.boxes=boxes.data();cfg.box_count=boxes.size();cfg.min_crop_width=std::stoi(argv[8]);cfg.min_crop_height=std::stoi(argv[9]);cfg.max_crop_width=std::stoi(argv[10]);cfg.max_crop_height=std::stoi(argv[11]);cfg.erosion_rate=std::stof(argv[12]);cfg.seed=std::stoull(argv[13]);const auto r=augmatch::make_bbox_safe_random_crop_rectangle(cfg);const std::size_t ni=static_cast<std::size_t>(cfg.input_width)*cfg.input_height*cfg.channels,no=static_cast<std::size_t>(r.width)*r.height*cfg.channels;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read target crop input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::bbox_safe_random_crop_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::bbox_safe_random_crop_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(no));std::cout<<r.x<<' '<<r.y<<' '<<r.width<<' '<<r.height<<' '<<r.used_fallback<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==13&&std::string(argv[1])=="random_sized_bbox_safe_crop"){
    try{augmatch::RandomSizedBBoxSafeCropConfig cfg;cfg.input_width=std::stoi(argv[4]);cfg.input_height=std::stoi(argv[5]);cfg.channels=std::stoi(argv[6]);auto boxes=read_target_boxes(argv[7]);cfg.boxes=boxes.data();cfg.box_count=boxes.size();cfg.output_width=std::stoi(argv[8]);cfg.output_height=std::stoi(argv[9]);cfg.erosion_rate=std::stof(argv[10]);cfg.interpolation=static_cast<augmatch::Interpolation>(std::stoi(argv[11]));cfg.seed=std::stoull(argv[12]);const auto r=augmatch::make_random_sized_bbox_safe_crop_rectangle(cfg);const std::size_t ni=static_cast<std::size_t>(cfg.input_width)*cfg.input_height*cfg.channels,no=static_cast<std::size_t>(cfg.output_width)*cfg.output_height*cfg.channels;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read target crop input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::random_sized_bbox_safe_crop_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_sized_bbox_safe_crop_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(no));std::cout<<r.x<<' '<<r.y<<' '<<r.width<<' '<<r.height<<' '<<r.scale_x<<' '<<r.scale_y<<' '<<r.used_fallback<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==14&&std::string(argv[1])=="at_least_one_bbox_random_crop"){
    try{augmatch::AtLeastOneBBoxRandomCropConfig cfg;cfg.input_width=std::stoi(argv[4]);cfg.input_height=std::stoi(argv[5]);cfg.channels=std::stoi(argv[6]);auto boxes=read_target_boxes(argv[7]);cfg.boxes=boxes.data();cfg.box_count=boxes.size();cfg.crop_width=std::stoi(argv[8]);cfg.crop_height=std::stoi(argv[9]);cfg.erosion_factor=std::stof(argv[10]);cfg.fallback_x=std::stoi(argv[11]);cfg.fallback_y=std::stoi(argv[12]);cfg.seed=std::stoull(argv[13]);const auto r=augmatch::make_at_least_one_bbox_random_crop_rectangle(cfg);const std::size_t ni=static_cast<std::size_t>(cfg.input_width)*cfg.input_height*cfg.channels,no=static_cast<std::size_t>(r.width)*r.height*cfg.channels;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read target crop input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::at_least_one_bbox_random_crop_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::at_least_one_bbox_random_crop_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(no));std::cout<<r.x<<' '<<r.y<<' '<<r.width<<' '<<r.height<<' '<<r.selected_box<<' '<<r.used_fallback<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==14||argc==18)&&(std::string(argv[1])=="random_crop_near_bbox"||std::string(argv[1])=="crop_near_bbox")){
    try{
      augmatch::RandomCropNearBBoxConfig cfg;cfg.input_width=std::stoi(argv[4]);cfg.input_height=std::stoi(argv[5]);cfg.channels=std::stoi(argv[6]);cfg.bbox={std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10])};cfg.max_part_shift_x=std::stof(argv[11]);cfg.max_part_shift_y=std::stof(argv[12]);cfg.seed=std::stoull(argv[13]);
      if(argc==18){cfg.left_offset=std::stoi(argv[14]);cfg.right_offset=std::stoi(argv[15]);cfg.top_offset=std::stoi(argv[16]);cfg.bottom_offset=std::stoi(argv[17]);}
      const auto r=augmatch::make_random_crop_near_bbox_rectangle(cfg);const std::size_t ni=static_cast<std::size_t>(cfg.input_width)*cfg.input_height*cfg.channels,no=static_cast<std::size_t>(r.width)*r.height*cfg.channels;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(ni));if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::random_crop_near_bbox_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::random_crop_near_bbox_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(no));return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==14||argc==15)&&(std::string(argv[1])=="random_resized_crop"||std::string(argv[1])=="random_sized_crop")){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),ow=std::stoi(argv[7]),oh=std::stoi(argv[8]);
      augmatch::RandomResizedCropConfig cfg{w,h,ow,oh,c,std::stoi(argv[9]),std::stoi(argv[10]),std::stoi(argv[11]),std::stoi(argv[12]),static_cast<augmatch::Interpolation>(std::stoi(argv[13])),argc==15?std::stoull(argv[14]):0};
      const auto r=augmatch::make_random_resized_crop_rectangle(cfg);const std::size_t ni=static_cast<std::size_t>(w)*h*c,no=static_cast<std::size_t>(ow)*oh*c;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(ni));if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));if(std::string(argv[1])=="random_resized_crop")augmatch::random_resized_crop_u8(di,doo,cfg);else augmatch::random_sized_crop_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(std::string(argv[1])=="random_resized_crop")augmatch::random_resized_crop_u8(in.data(),out.data(),cfg);else augmatch::random_sized_crop_u8(in.data(),out.data(),cfg);
#endif
      (void)r;std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(no));return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="crop_pad"){  
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::CropPadConfig cfg{w,h,c,std::stoi(argv[7]),std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10]),static_cast<std::uint8_t>(std::stoi(argv[11]))};int ow=w+cfg.left+cfg.right,oh=h+cfg.top+cfg.bottom;if(ow<=0||oh<=0)throw std::invalid_argument("invalid output dimensions");std::size_t ni=static_cast<std::size_t>(w)*h*c,no=static_cast<std::size_t>(ow)*oh*c;std::vector<std::uint8_t> in(ni),out(no);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),ni);if(!fi||fi.gcount()!=static_cast<std::streamsize>(ni))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,ni));ck(cudaMalloc(&doo,no));ck(cudaMemcpy(di,in.data(),ni,cudaMemcpyHostToDevice));augmatch::crop_and_pad_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,no,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::crop_and_pad_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),no);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="motion_blur"){
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::MotionBlurConfig cfg{w,h,c,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),true};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::motion_blur_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::motion_blur_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="sharpen"){ 
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::SharpenConfig cfg{w,h,c,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10])};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::sharpen_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::sharpen_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="unsharp"){ 
    try{int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);augmatch::UnsharpConfig cfg{w,h,c,std::stoi(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stoi(argv[10])};std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::unsharp_mask_u8(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::unsharp_mask_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==10&&std::string(argv[1])=="resize"){   
    try{
      int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),ow=std::stoi(argv[7]),oh=std::stoi(argv[8]);
      augmatch::ResizeConfig config{w,h,ow,oh,c,static_cast<augmatch::Interpolation>(std::stoi(argv[9]))};
      std::vector<std::uint8_t> in(size_t(w)*h*c),out(size_t(ow)*oh*c);
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));augmatch::resize_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::resize_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==11&&std::string(argv[1])=="pool"){ 
    try{
      int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]),kw=std::stoi(argv[7]),kh=std::stoi(argv[8]);
      augmatch::PoolMode mode=static_cast<augmatch::PoolMode>(std::stoi(argv[9]));bool keep=std::stoi(argv[10])!=0;
      augmatch::Pooling p{w,h,c,kw,kh,mode,keep};int ow=keep?w:w/kw,oh=keep?h:h/kh;
      std::vector<std::uint8_t> in(size_t(w)*h*c),out(size_t(ow)*oh*c);
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));augmatch::pool_u8(di,doo,p);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::pool_u8(in.data(),out.data(),p);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(((argc==10||argc==11)&&(std::string(argv[1])=="adc_dnl"||std::string(argv[1])=="adc_differential_non_linearity"||std::string(argv[1])=="adc_differential_nonlinearity"||std::string(argv[1])=="adc_inl"||std::string(argv[1])=="adc_integral_non_linearity"||std::string(argv[1])=="adc_integral_nonlinearity"))||((argc==9||argc==10)&&(std::string(argv[1])=="well_capacity"||std::string(argv[1])=="sensor_well_capacity_variation"))){
    try{
      const std::string op=argv[1]; const bool dnl=op=="adc_dnl"||op=="adc_differential_non_linearity"||op=="adc_differential_nonlinearity"; const bool inl=op=="adc_inl"||op=="adc_integral_non_linearity"||op=="adc_integral_nonlinearity";
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]); const std::size_t n=static_cast<std::size_t>(w)*h*c;
      std::vector<float> in(n),out(n),map; std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");
      const bool explicit_map=dnl||inl?argc==11:argc==10; const std::size_t map_count=dnl||inl?static_cast<std::size_t>(c)*std::stoi(argv[7]):n;
      if(explicit_map){const char* path=argv[dnl||inl?10:9];map.resize(map_count);std::ifstream fm(path,std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read non-linearity/capacity LUT");}
      const int levels=dnl||inl?std::stoi(argv[7]):0; const float stddev=std::stof(argv[dnl||inl?8:7]); const std::uint64_t seed=static_cast<std::uint64_t>(std::stoull(argv[dnl||inl?9:8]));
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(explicit_map){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));}
      if(dnl){augmatch::AdcDifferentialNonLinearityConfig cfg{w,h,c,levels,dm,stddev,seed};augmatch::adc_differential_non_linearity_f32(di,doo,cfg);}else if(inl){augmatch::AdcIntegralNonLinearityConfig cfg{w,h,c,levels,dm,stddev,seed};augmatch::adc_integral_non_linearity_f32(di,doo,cfg);}else{augmatch::SensorWellCapacityVariationConfig cfg{w,h,c,dm,stddev,seed};augmatch::sensor_well_capacity_variation_f32(di,doo,cfg);}ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)ck(cudaFree(dm));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(dnl)augmatch::adc_differential_non_linearity_f32(in.data(),out.data(),{w,h,c,levels,explicit_map?map.data():nullptr,stddev,seed});else if(inl)augmatch::adc_integral_non_linearity_f32(in.data(),out.data(),{w,h,c,levels,explicit_map?map.data():nullptr,stddev,seed});else augmatch::sensor_well_capacity_variation_f32(in.data(),out.data(),{w,h,c,explicit_map?map.data():nullptr,stddev,seed});
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write float output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==9||argc==10)&&(std::string(argv[1])=="prnu"||std::string(argv[1])=="pixel_response_non_uniformity"||std::string(argv[1])=="fpn"||std::string(argv[1])=="fixed_pattern_offset_noise")){ 
    try{
      const std::string op=argv[1]; const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      const std::size_t n=static_cast<std::size_t>(w)*h*c; std::vector<float> in(n),out(n),map;
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");
      const bool is_prnu=op=="prnu"||op=="pixel_response_non_uniformity";
      if(argc==10){map.resize(n);std::ifstream fm(argv[9],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read noise map");}
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));
      if(!map.empty()){ck(cudaMalloc(&dm,n*sizeof(float)));ck(cudaMemcpy(dm,map.data(),n*sizeof(float),cudaMemcpyHostToDevice));}
      if(is_prnu)augmatch::pixel_response_non_uniformity_f32(di,doo,{w,h,c,std::stof(argv[7]),dm,static_cast<std::uint64_t>(std::stoull(argv[8]))});
      else augmatch::fixed_pattern_offset_noise_f32(di,doo,{w,h,c,std::stof(argv[7]),dm,static_cast<std::uint64_t>(std::stoull(argv[8]))});
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));if(dm)ck(cudaFree(dm));
#else
      if(is_prnu)augmatch::pixel_response_non_uniformity_f32(in.data(),out.data(),{w,h,c,std::stof(argv[7]),map.empty()?nullptr:map.data(),static_cast<std::uint64_t>(std::stoull(argv[8]))});
      else augmatch::fixed_pattern_offset_noise_f32(in.data(),out.data(),{w,h,c,std::stof(argv[7]),map.empty()?nullptr:map.data(),static_cast<std::uint64_t>(std::stoull(argv[8]))});
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write noise output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10||argc==12)&&(std::string(argv[1])=="row_column_correlated_noise"||std::string(argv[1])=="row_column_noise")){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]); const std::size_t n=static_cast<std::size_t>(w)*h*c;
      std::vector<float> in(n),out(n),row(static_cast<std::size_t>(h)*c),column(static_cast<std::size_t>(w)*c); std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");
      const bool explicit_maps=argc==12; if(explicit_maps){std::ifstream fr(argv[10],std::ios::binary),fc(argv[11],std::ios::binary);fr.read(reinterpret_cast<char*>(row.data()),static_cast<std::streamsize>(row.size()*sizeof(float)));fc.read(reinterpret_cast<char*>(column.data()),static_cast<std::streamsize>(column.size()*sizeof(float)));if(!fr||fr.gcount()!=static_cast<std::streamsize>(row.size()*sizeof(float))||!fc||fc.gcount()!=static_cast<std::streamsize>(column.size()*sizeof(float)))throw std::runtime_error("failed to read row/column noise map");}
      augmatch::RowColumnCorrelatedNoiseConfig config{w,h,c,std::stof(argv[7]),std::stof(argv[8]),explicit_maps?row.data():nullptr,explicit_maps?column.data():nullptr,0.0f,1.0f,static_cast<std::uint64_t>(std::stoull(argv[9]))};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dr=nullptr,*dc=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(explicit_maps){ck(cudaMalloc(&dr,row.size()*sizeof(float)));ck(cudaMalloc(&dc,column.size()*sizeof(float)));ck(cudaMemcpy(dr,row.data(),row.size()*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dc,column.data(),column.size()*sizeof(float),cudaMemcpyHostToDevice));config.row_map=dr;config.column_map=dc;}augmatch::row_column_correlated_noise_f32(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dc)ck(cudaFree(dc));if(dr)ck(cudaFree(dr));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::row_column_correlated_noise_f32(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write noise output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==12||argc==13)&&(std::string(argv[1])=="clustered_defective_pixels"||std::string(argv[1])=="clustered_defects")){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]); const std::size_t n=static_cast<std::size_t>(w)*h*c; std::vector<float> in(n),out(n); std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");
      std::vector<augmatch::ClusteredDefect> records; const bool explicit_records=argc==13; if(explicit_records){std::ifstream fd(argv[12],std::ios::binary);fd.seekg(0,std::ios::end);const std::streamoff bytes=fd.tellg();fd.seekg(0);if(bytes<0||bytes%static_cast<std::streamoff>(sizeof(augmatch::ClusteredDefect))!=0)throw std::runtime_error("clustered defect file size is not a record multiple");records.resize(static_cast<std::size_t>(bytes)/sizeof(augmatch::ClusteredDefect));fd.read(reinterpret_cast<char*>(records.data()),bytes);if(!fd)throw std::runtime_error("failed to read clustered defect records");}
      augmatch::ClusteredDefectivePixelsConfig config{w,h,c,explicit_records?records.data():nullptr,records.size(),std::stoi(argv[7]),std::stoi(argv[8]),std::stof(argv[9]),std::stof(argv[10]),static_cast<std::uint64_t>(std::stoull(argv[11]))};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;augmatch::ClusteredDefect *dr=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(explicit_records){ck(cudaMalloc(&dr,records.size()*sizeof(augmatch::ClusteredDefect)));ck(cudaMemcpy(dr,records.data(),records.size()*sizeof(augmatch::ClusteredDefect),cudaMemcpyHostToDevice));config.defects=dr;}augmatch::clustered_defective_pixels_f32(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dr)ck(cudaFree(dr));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::clustered_defective_pixels_f32(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write noise output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc>=11&&argc<=13)&&(std::string(argv[1])=="blooming_vertical_smear"||std::string(argv[1])=="blooming_and_vertical_smear"||std::string(argv[1])=="blooming_smear")){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]); const std::size_t n=static_cast<std::size_t>(w)*h*c;
      std::vector<float> in(n),out(n),column(static_cast<std::size_t>(w)*c); std::vector<augmatch::BrightPixel> pixels;
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");
      if(argc>=12){std::ifstream fp(argv[11],std::ios::binary);fp.seekg(0,std::ios::end);const std::streamoff bytes=fp.tellg();fp.seekg(0);if(bytes<0||bytes%static_cast<std::streamoff>(sizeof(augmatch::BrightPixel))!=0)throw std::runtime_error("bright-pixel file size is not a record multiple");pixels.resize(static_cast<std::size_t>(bytes)/sizeof(augmatch::BrightPixel));fp.read(reinterpret_cast<char*>(pixels.data()),bytes);if(!fp)throw std::runtime_error("failed to read bright-pixel records");}
      if(argc==13){column.resize(static_cast<std::size_t>(w)*c);std::ifstream fm(argv[12],std::ios::binary);fm.read(reinterpret_cast<char*>(column.data()),static_cast<std::streamsize>(column.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(column.size()*sizeof(float)))throw std::runtime_error("failed to read column smear map");}
      augmatch::BloomingVerticalSmearConfig config{w,h,c,pixels.empty()?nullptr:pixels.data(),pixels.size(),argc==13?column.data():nullptr,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),0.0f,1.0f,static_cast<std::uint64_t>(std::stoull(argv[10]))};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;augmatch::BrightPixel *dp=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(!pixels.empty()){ck(cudaMalloc(&dp,pixels.size()*sizeof(augmatch::BrightPixel)));ck(cudaMemcpy(dp,pixels.data(),pixels.size()*sizeof(augmatch::BrightPixel),cudaMemcpyHostToDevice));config.bright_pixels=dp;}if(argc==13){ck(cudaMalloc(&dm,column.size()*sizeof(float)));ck(cudaMemcpy(dm,column.data(),column.size()*sizeof(float),cudaMemcpyHostToDevice));config.column_smear_map=dm;}augmatch::blooming_vertical_smear_f32(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)ck(cudaFree(dm));if(dp)ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::blooming_vertical_smear_f32(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write blooming output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10)&&(std::string(argv[1])=="sensor_dust_opaque_mask"||std::string(argv[1])=="sensor_dust_and_opaque_pixel_mask"||std::string(argv[1])=="sensor_dust")){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n);std::vector<std::uint8_t> mask(static_cast<std::size_t>(w)*h);std::ifstream fi(argv[2],std::ios::binary),fm(argv[9],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));fm.read(reinterpret_cast<char*>(mask.data()),static_cast<std::streamsize>(mask.size()));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float))||!fm||fm.gcount()!=static_cast<std::streamsize>(mask.size()))throw std::runtime_error("failed to read sensor dust input or mask");
      augmatch::SensorDustOpaqueMaskConfig config{w,h,c,mask.data(),std::stoi(argv[7]),std::stof(argv[8])};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;std::uint8_t *dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMalloc(&dm,mask.size()));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dm,mask.data(),mask.size(),cudaMemcpyHostToDevice));config.mask=dm;augmatch::sensor_dust_opaque_mask_f32(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(dm));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::sensor_dust_opaque_mask_f32(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write sensor dust output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10||argc==11)&&std::string(argv[1])=="flare"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),map;std::vector<augmatch::FlareSource> sources;std::vector<augmatch::FlareHalo> halos;std::ifstream fi(argv[2],std::ios::binary),fs(argv[8],std::ios::binary),fh(argv[9],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));auto read_records=[](std::ifstream& f,auto& v){f.seekg(0,std::ios::end);std::streamoff bytes=f.tellg();f.seekg(0);if(bytes<0||bytes%static_cast<std::streamoff>(sizeof(v[0]))!=0)throw std::runtime_error("flare record file size is invalid");v.resize(static_cast<std::size_t>(bytes)/sizeof(v[0]));f.read(reinterpret_cast<char*>(v.data()),bytes);if(!f&&!v.empty())throw std::runtime_error("failed to read flare records");};read_records(fs,sources);read_records(fh,halos);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read flare input");if(argc==11){map.resize(static_cast<std::size_t>(w)*h);std::ifstream fm(argv[10],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read flare map");}augmatch::FlareConfig cfg{w,h,c,sources.empty()?nullptr:sources.data(),sources.size(),halos.empty()?nullptr:halos.data(),halos.size(),map.empty()?nullptr:map.data(),1,std::stof(argv[7]),0,1};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;augmatch::FlareSource *ds=nullptr;augmatch::FlareHalo *dh=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(!sources.empty()){ck(cudaMalloc(&ds,sources.size()*sizeof(augmatch::FlareSource)));ck(cudaMemcpy(ds,sources.data(),sources.size()*sizeof(augmatch::FlareSource),cudaMemcpyHostToDevice));cfg.sources=ds;}if(!halos.empty()){ck(cudaMalloc(&dh,halos.size()*sizeof(augmatch::FlareHalo)));ck(cudaMemcpy(dh,halos.data(),halos.size()*sizeof(augmatch::FlareHalo),cudaMemcpyHostToDevice));cfg.halos=dh;}if(!map.empty()){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.map=dm;}augmatch::flare_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);if(dh)cudaFree(dh);if(ds)cudaFree(ds);cudaFree(di);cudaFree(doo);
#else
      augmatch::flare_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==9||argc==10)&&std::string(argv[1])=="ghosting"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),map;std::vector<augmatch::Ghost> ghosts;std::ifstream fi(argv[2],std::ios::binary),fg(argv[8],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));fg.seekg(0,std::ios::end);std::streamoff bytes=fg.tellg();fg.seekg(0);if(bytes<0||bytes%static_cast<std::streamoff>(sizeof(augmatch::Ghost))!=0)throw std::runtime_error("ghost file size is invalid");ghosts.resize(static_cast<std::size_t>(bytes)/sizeof(augmatch::Ghost));fg.read(reinterpret_cast<char*>(ghosts.data()),bytes);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float))||(!fg&& !ghosts.empty()))throw std::runtime_error("failed to read ghost input");if(argc==10){map.resize(static_cast<std::size_t>(w)*h);std::ifstream fm(argv[9],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read ghost map");}augmatch::GhostingConfig cfg{w,h,c,ghosts.empty()?nullptr:ghosts.data(),ghosts.size(),map.empty()?nullptr:map.data(),1,std::stof(argv[7]),0,1};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;augmatch::Ghost *dg=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(!ghosts.empty()){ck(cudaMalloc(&dg,ghosts.size()*sizeof(augmatch::Ghost)));ck(cudaMemcpy(dg,ghosts.data(),ghosts.size()*sizeof(augmatch::Ghost),cudaMemcpyHostToDevice));cfg.ghosts=dg;}if(!map.empty()){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.map=dm;}augmatch::ghosting_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);if(dg)cudaFree(dg);cudaFree(di);cudaFree(doo);
#else
      augmatch::ghosting_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==9||argc==10)&&(std::string(argv[1])=="dirty_lens_blur"||std::string(argv[1])=="dirty-lens-blur")){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),map;std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read dirty-lens input");if(argc==10){map.resize(static_cast<std::size_t>(w)*h);std::ifstream fm(argv[9],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read dirty-lens map");}augmatch::DirtyLensBlurConfig cfg;cfg.width=w;cfg.height=h;cfg.channels=c;cfg.radius=std::stoi(argv[7]);cfg.strength=std::stof(argv[8]);cfg.map=map.empty()?nullptr:map.data();
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(!map.empty()){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.map=dm;}augmatch::dirty_lens_blur_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);cudaFree(di);cudaFree(doo);
#else
      augmatch::dirty_lens_blur_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==11||argc==12)&&(std::string(argv[1])=="electromagnetic_interference"||std::string(argv[1])=="emi")){ 
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),map;std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read EMI input");if(argc==12){map.resize(static_cast<std::size_t>(w)*h);std::ifstream fm(argv[11],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read EMI map");}augmatch::ElectromagneticInterferenceConfig cfg;cfg.width=w;cfg.height=h;cfg.channels=c;cfg.amplitude=std::stof(argv[7]);cfg.frequency_x=std::stof(argv[8]);cfg.frequency_y=std::stof(argv[9]);cfg.phase=std::stof(argv[10]);cfg.map=map.empty()?nullptr:map.data();
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(!map.empty()){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.map=dm;}augmatch::electromagnetic_interference_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);cudaFree(di);cudaFree(doo);
#else
      augmatch::electromagnetic_interference_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==15&&(std::string(argv[1])=="sensor_temperature_drift"||std::string(argv[1])=="temperature_drift")){
    try{const int t=std::stoi(argv[4]),h=std::stoi(argv[5]),w=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(t)*h*w*c;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read temperature-drift input");augmatch::SensorTemperatureDriftConfig cfg;cfg.frames=t;cfg.height=h;cfg.width=w;cfg.channels=c;cfg.start_temperature_celsius=std::stof(argv[8]);cfg.end_temperature_celsius=std::stof(argv[9]);cfg.reference_temperature_celsius=std::stof(argv[10]);cfg.dark_current_electrons_per_second=std::stof(argv[11]);cfg.electrons_per_unit=std::stof(argv[12]);cfg.exposure_seconds=std::stof(argv[13]);cfg.temperature_coefficient_per_celsius=std::stof(argv[14]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));augmatch::sensor_temperature_drift_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      augmatch::sensor_temperature_drift_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==13&&(std::string(argv[1])=="power_supply_banding"||std::string(argv[1])=="power-supply-banding"||std::string(argv[1])=="power_banding")){ 
    try{const int t=std::stoi(argv[4]),h=std::stoi(argv[5]),w=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(t)*h*w*c;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read power-banding input");augmatch::PowerSupplyBandingConfig cfg;cfg.frames=t;cfg.height=h;cfg.width=w;cfg.channels=c;cfg.amplitude=std::stof(argv[8]);cfg.temporal_frequency_hz=std::stof(argv[9]);cfg.frame_rate_hz=std::stof(argv[10]);cfg.row_frequency=std::stof(argv[11]);cfg.phase=std::stof(argv[12]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));augmatch::power_supply_banding_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      augmatch::power_supply_banding_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&(std::string(argv[1])=="fluorescent_light_flicker"||std::string(argv[1])=="fluorescent-flicker"||std::string(argv[1])=="fluorescent_flicker")){
    try{const int t=std::stoi(argv[4]),h=std::stoi(argv[5]),w=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(t)*h*w*c;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read fluorescent-flicker input");augmatch::FluorescentLightFlickerConfig cfg;cfg.frames=t;cfg.height=h;cfg.width=w;cfg.channels=c;cfg.amplitude=std::stof(argv[8]);cfg.flicker_frequency_hz=std::stof(argv[9]);cfg.frame_rate_hz=std::stof(argv[10]);cfg.phase=std::stof(argv[11]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));augmatch::fluorescent_light_flicker_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      augmatch::fluorescent_light_flicker_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==13&&(std::string(argv[1])=="led_rolling_band_artifacts"||std::string(argv[1])=="led-rolling-band"||std::string(argv[1])=="led_rolling_band")){
    try{const int t=std::stoi(argv[4]),h=std::stoi(argv[5]),w=std::stoi(argv[6]),c=std::stoi(argv[7]);const std::size_t n=static_cast<std::size_t>(t)*h*w*c;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read LED-band input");augmatch::LedRollingBandArtifactsConfig cfg;cfg.frames=t;cfg.height=h;cfg.width=w;cfg.channels=c;cfg.amplitude=std::stof(argv[8]);cfg.modulation_frequency_hz=std::stof(argv[9]);cfg.frame_rate_hz=std::stof(argv[10]);cfg.row_cycles=std::stof(argv[11]);cfg.phase=std::stof(argv[12]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));augmatch::led_rolling_band_artifacts_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      augmatch::led_rolling_band_artifacts_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==8||argc==9)&&(std::string(argv[1])=="atmospheric_haze"||std::string(argv[1])=="fog_veil"||std::string(argv[1])=="smoke_veil"||std::string(argv[1])=="window_glare"||std::string(argv[1])=="backlight_washout")){
    try{const std::string op=argv[1];const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),map;std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read environmental input");if(argc==9){map.resize(static_cast<std::size_t>(w)*h);std::ifstream fm(argv[8],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read environmental map");}augmatch::EnvironmentalVeilConfig cfg{w,h,c,map.empty()?nullptr:map.data(),1,std::stof(argv[7]),op=="smoke_veil"?0.35f:1.0f,0,1};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(!map.empty()){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.map=dm;}if(op=="atmospheric_haze"||op=="fog_veil")augmatch::atmospheric_haze_f32(di,doo,cfg);else if(op=="smoke_veil")augmatch::smoke_veil_f32(di,doo,cfg);else if(op=="window_glare")augmatch::window_glare_f32(di,doo,cfg);else augmatch::backlight_washout_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);cudaFree(di);cudaFree(doo);
#else
      if(op=="atmospheric_haze"||op=="fog_veil")augmatch::atmospheric_haze_f32(in.data(),out.data(),cfg);else if(op=="smoke_veil")augmatch::smoke_veil_f32(in.data(),out.data(),cfg);else if(op=="window_glare")augmatch::window_glare_f32(in.data(),out.data(),cfg);else augmatch::backlight_washout_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==8&&(std::string(argv[1])=="low_light_amplification"||std::string(argv[1])=="underexposure"||std::string(argv[1])=="overexposure")){
    try{const std::string op=argv[1];const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read gain input");augmatch::AcquisitionGainConfig cfg{w,h,c,std::stof(argv[7]),0,1};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(op=="low_light_amplification")augmatch::low_light_amplification_f32(di,doo,cfg);else if(op=="underexposure")augmatch::underexposure_f32(di,doo,cfg);else augmatch::overexposure_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      if(op=="low_light_amplification")augmatch::low_light_amplification_f32(in.data(),out.data(),cfg);else if(op=="underexposure")augmatch::underexposure_f32(in.data(),out.data(),cfg);else augmatch::overexposure_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==8||argc==9)&&(std::string(argv[1])=="veiling_glare"||std::string(argv[1])=="veiling-glare")){ 
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),map;std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read glare input");if(argc==9){map.resize(static_cast<std::size_t>(w)*h);std::ifstream fm(argv[8],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read glare map");}augmatch::VeilingGlareConfig cfg{w,h,c,map.empty()?nullptr:map.data(),1,std::stof(argv[7]),0,1};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(!map.empty()){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.map=dm;}augmatch::veiling_glare_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);cudaFree(di);cudaFree(doo);
#else
      augmatch::veiling_glare_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10||argc==11)&&std::string(argv[1])=="bloom"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),map;std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read bloom input");if(argc==11){map.resize(static_cast<std::size_t>(w)*h);std::ifstream fm(argv[10],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read bloom map");}augmatch::BloomConfig cfg{w,h,c,std::stoi(argv[9]),std::stof(argv[7]),std::stof(argv[8]),map.empty()?nullptr:map.data(),1,0,1};
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(!map.empty()){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));cfg.map=dm;}augmatch::bloom_f32(di,doo,cfg);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);cudaFree(di);cudaFree(doo);
#else
      augmatch::bloom_f32(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc>=8&&(std::string(argv[1])=="lens_vignetting"||std::string(argv[1])=="color_dependent_vignetting"||std::string(argv[1])=="color_vignetting"||std::string(argv[1])=="optical_falloff"||std::string(argv[1])=="lens_shading"||std::string(argv[1])=="uneven_illumination"||std::string(argv[1])=="sensor_lens_dust_shadows"||std::string(argv[1])=="dust_shadows")){ 
    try{
      const std::string op=argv[1];const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),map;std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read optical input");
      const bool radial=(op=="lens_vignetting"||op=="color_dependent_vignetting"||op=="color_vignetting"||op=="optical_falloff"),shading=op=="lens_shading",illumination=op=="uneven_illumination";const bool has_map=(radial&&argc==11)||(shading&&argc==9)||(illumination&&argc==10)||(!radial&&!shading&&!illumination&&argc==9);if(has_map){const std::size_t mn=(!radial&&!shading&&!illumination)?static_cast<std::size_t>(w)*h:static_cast<std::size_t>(w)*h;map.resize(mn);std::ifstream fm(argv[argc-1],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read optical map");}
      const float a=radial?std::stof(argv[7]):std::stof(argv[7]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(has_map){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));}
      if(radial){augmatch::LensVignettingConfig cfg;cfg.width=w;cfg.height=h;cfg.channels=c;cfg.c0=1.0f;cfg.c1=a;cfg.c2=std::stof(argv[8]);cfg.c3=std::stof(argv[9]);cfg.map=dm;if(op=="color_dependent_vignetting"||op=="color_vignetting"){augmatch::ColorDependentVignettingConfig cc;static_cast<augmatch::LensVignettingConfig&>(cc)=cfg;augmatch::color_dependent_vignetting_f32(di,doo,cc);}else if(op=="optical_falloff")augmatch::optical_falloff_f32(di,doo,cfg);else augmatch::lens_vignetting_f32(di,doo,cfg);}else if(shading){augmatch::LensShadingConfig cfg{w,h,c,dm,1,a,0.0f,1.0f};augmatch::lens_shading_f32(di,doo,cfg);}else if(illumination){augmatch::UnevenIlluminationConfig cfg{w,h,c,dm,1,a,std::stof(argv[8]),0.0f,1.0f};augmatch::uneven_illumination_f32(di,doo,cfg);}else{augmatch::SensorLensDustShadowsConfig cfg{w,h,c,dm,a,0.0f,1.0f};augmatch::sensor_lens_dust_shadows_f32(di,doo,cfg);}ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)ck(cudaFree(dm));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(radial){augmatch::LensVignettingConfig cfg;cfg.width=w;cfg.height=h;cfg.channels=c;cfg.c0=1.0f;cfg.c1=a;cfg.c2=std::stof(argv[8]);cfg.c3=std::stof(argv[9]);cfg.map=has_map?map.data():nullptr;if(op=="color_dependent_vignetting"||op=="color_vignetting"){augmatch::ColorDependentVignettingConfig cc;static_cast<augmatch::LensVignettingConfig&>(cc)=cfg;augmatch::color_dependent_vignetting_f32(in.data(),out.data(),cc);}else if(op=="optical_falloff")augmatch::optical_falloff_f32(in.data(),out.data(),cfg);else augmatch::lens_vignetting_f32(in.data(),out.data(),cfg);}else if(shading){augmatch::LensShadingConfig cfg{w,h,c,has_map?map.data():nullptr,1,a,0.0f,1.0f};augmatch::lens_shading_f32(in.data(),out.data(),cfg);}else if(illumination){augmatch::UnevenIlluminationConfig cfg{w,h,c,has_map?map.data():nullptr,1,a,std::stof(argv[8]),0.0f,1.0f};augmatch::uneven_illumination_f32(in.data(),out.data(),cfg);}else{augmatch::SensorLensDustShadowsConfig cfg{w,h,c,has_map?map.data():nullptr,a,0.0f,1.0f};augmatch::sensor_lens_dust_shadows_f32(in.data(),out.data(),cfg);}
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write optical output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==10)&&(std::string(argv[1])=="shot_noise"||std::string(argv[1])=="shotnoise")){ 
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      augmatch::ShotNoiseConfig config{w,h,c,std::stof(argv[7]),std::stof(argv[8]),static_cast<std::uint64_t>(std::stoull(argv[9]))};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::shot_noise_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::shot_noise_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==14)&&(std::string(argv[1])=="iso_noise"||std::string(argv[1])=="isonoise")){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      augmatch::ISONoiseConfig config{w,h,c,std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9]),std::stof(argv[10]),std::stof(argv[11]),std::stof(argv[12]),static_cast<std::uint64_t>(std::stoull(argv[13]))};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::iso_noise_u8(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::iso_noise_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Exposure-time dark current: iso_dark_current IN OUT W H C ISO KNOTS_F32 COUNT RATE_EPS EXPOSURE_SECONDS ELECTRONS_PER_UNIT SEED.
  if((argc==14)&&(std::string(argv[1])=="iso_dark_current"||std::string(argv[1])=="exposure_time_dark_current"||std::string(argv[1])=="exposure_dark_current")){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;const std::size_t count=static_cast<std::size_t>(std::stoull(argv[9]));
      if(count==0||count>100000)throw std::invalid_argument("ISO profile knot count must be in [1,100000]");std::vector<augmatch::IsoNoiseProfilePoint> points(count);std::ifstream fp(argv[8],std::ios::binary);fp.read(reinterpret_cast<char*>(points.data()),static_cast<std::streamsize>(count*sizeof(points[0])));if(!fp||fp.gcount()!=static_cast<std::streamsize>(count*sizeof(points[0])))throw std::runtime_error("failed to read ISO profile knots");
      std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");
      augmatch::ExposureTimeDarkCurrentConfig config{};config.width=w;config.height=h;config.channels=c;config.iso=std::stof(argv[7]);config.profile={points.data(),count};config.dark_current_electrons_per_second=std::stof(argv[10]);config.exposure_seconds=std::stof(argv[11]);config.electrons_per_unit=std::stof(argv[12]);config.seed=static_cast<std::uint64_t>(std::stoull(argv[13]));
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;augmatch::IsoNoiseProfilePoint *dp=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMalloc(&dp,count*sizeof(points[0])));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dp,points.data(),count*sizeof(points[0]),cudaMemcpyHostToDevice));config.profile.points=dp;augmatch::exposure_time_dark_current_f32(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::exposure_time_dark_current_f32(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write dark-current output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Temperature noise scaling: iso_temperature_noise IN OUT W H C ISO KNOTS_F32 COUNT NOISE_E TEMP_C REF_C COEFF_PER_C ELECTRONS_PER_UNIT SEED.
  if((argc==16)&&(std::string(argv[1])=="iso_temperature_noise"||std::string(argv[1])=="temperature_noise_scaling"||std::string(argv[1])=="temperature_dependent_noise")){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;const std::size_t count=static_cast<std::size_t>(std::stoull(argv[9]));
      if(count==0||count>100000)throw std::invalid_argument("ISO profile knot count must be in [1,100000]");std::vector<augmatch::IsoNoiseProfilePoint> points(count);std::ifstream fp(argv[8],std::ios::binary);fp.read(reinterpret_cast<char*>(points.data()),static_cast<std::streamsize>(count*sizeof(points[0])));if(!fp||fp.gcount()!=static_cast<std::streamsize>(count*sizeof(points[0])))throw std::runtime_error("failed to read ISO profile knots");
      std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");
      augmatch::TemperatureNoiseScalingConfig config{};config.width=w;config.height=h;config.channels=c;config.iso=std::stof(argv[7]);config.profile={points.data(),count};config.noise_stddev_electrons=std::stof(argv[10]);config.temperature_celsius=std::stof(argv[11]);config.reference_temperature_celsius=std::stof(argv[12]);config.temperature_noise_coefficient_per_celsius=std::stof(argv[13]);config.electrons_per_unit=std::stof(argv[14]);config.seed=static_cast<std::uint64_t>(std::stoull(argv[15]));
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;augmatch::IsoNoiseProfilePoint *dp=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMalloc(&dp,count*sizeof(points[0])));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dp,points.data(),count*sizeof(points[0]),cudaMemcpyHostToDevice));config.profile.points=dp;augmatch::temperature_noise_scaling_f32(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::temperature_noise_scaling_f32(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fo)throw std::runtime_error("failed to write temperature-noise output");return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Dual conversion gain: dual_conversion_gain IN OUT W H C ISO KNOTS_F32 COUNT LOW_T HIGH_T LOW_GAIN HIGH_GAIN LOW_READ_E HIGH_READ_E ELECTRONS_PER_UNIT SEED.
  if(argc==18&&std::string(argv[1])=="dual_conversion_gain"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c,count=static_cast<std::size_t>(std::stoull(argv[9]));if(count==0||count>100000)throw std::invalid_argument("ISO profile knot count must be in [1,100000]");std::vector<augmatch::IsoNoiseProfilePoint> p(count);std::ifstream fp(argv[8],std::ios::binary);fp.read(reinterpret_cast<char*>(p.data()),static_cast<std::streamsize>(count*sizeof(p[0])));if(!fp||fp.gcount()!=static_cast<std::streamsize>(count*sizeof(p[0])))throw std::runtime_error("failed to read ISO profile knots");std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");augmatch::DualConversionGainConfig z{};z.width=w;z.height=h;z.channels=c;z.iso=std::stof(argv[7]);z.profile={p.data(),count};z.low_gain_threshold=std::stof(argv[10]);z.high_gain_threshold=std::stof(argv[11]);z.low_conversion_gain=std::stof(argv[12]);z.high_conversion_gain=std::stof(argv[13]);z.low_read_noise_electrons=std::stof(argv[14]);z.high_read_noise_electrons=std::stof(argv[15]);z.electrons_per_unit=std::stof(argv[16]);z.seed=std::stoull(argv[17]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;augmatch::IsoNoiseProfilePoint *dp=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMalloc(&dp,count*sizeof(p[0])));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dp,p.data(),count*sizeof(p[0]),cudaMemcpyHostToDevice));z.profile.points=dp;augmatch::dual_conversion_gain_f32(di,doo,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::dual_conversion_gain_f32(in.data(),out.data(),z);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Gain-switch transition: gain_switch_transition IN OUT W H C ISO KNOTS_F32 COUNT LOW_T HIGH_T LOW_SIGNAL HIGH_SIGNAL HYSTERESIS TRANSITION_WIDTH STRENGTH INITIAL_HIGH SEED.
  if(argc==19&&std::string(argv[1])=="gain_switch_transition"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c,count=static_cast<std::size_t>(std::stoull(argv[9]));if(count==0||count>100000)throw std::invalid_argument("ISO profile knot count must be in [1,100000]");std::vector<augmatch::IsoNoiseProfilePoint> p(count);std::ifstream fp(argv[8],std::ios::binary);fp.read(reinterpret_cast<char*>(p.data()),static_cast<std::streamsize>(count*sizeof(p[0])));if(!fp||fp.gcount()!=static_cast<std::streamsize>(count*sizeof(p[0])))throw std::runtime_error("failed to read ISO profile knots");std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");augmatch::GainSwitchTransitionConfig z{};z.width=w;z.height=h;z.channels=c;z.iso=std::stof(argv[7]);z.profile={p.data(),count};z.low_gain_threshold=std::stof(argv[10]);z.high_gain_threshold=std::stof(argv[11]);z.low_signal_gain=std::stof(argv[12]);z.high_signal_gain=std::stof(argv[13]);z.hysteresis=std::stof(argv[14]);z.transition_width=std::stof(argv[15]);z.transition_strength=std::stof(argv[16]);z.initial_high_gain=std::stoi(argv[17])!=0;z.seed=std::stoull(argv[18]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;augmatch::IsoNoiseProfilePoint *dp=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMalloc(&dp,count*sizeof(p[0])));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dp,p.data(),count*sizeof(p[0]),cudaMemcpyHostToDevice));z.profile.points=dp;augmatch::gain_switch_transition_f32(di,doo,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::gain_switch_transition_f32(in.data(),out.data(),z);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Camera profile LUT: camera_profile_lookup IN OUT W H C ISO LUT_F32 COUNT SEED.
  // LUT entries are ten little-endian float32 fields in CameraProfileLookupPoint order.
  if(argc==11&&(std::string(argv[1])=="camera_profile_lookup"||std::string(argv[1])=="camera_profile_lut")){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c,count=static_cast<std::size_t>(std::stoull(argv[9]));if(count==0||count>100000)throw std::invalid_argument("camera profile LUT count must be in [1,100000]");std::vector<augmatch::CameraProfileLookupPoint> p(count);std::ifstream fp(argv[8],std::ios::binary);fp.read(reinterpret_cast<char*>(p.data()),static_cast<std::streamsize>(count*sizeof(p[0])));if(!fp||fp.gcount()!=static_cast<std::streamsize>(count*sizeof(p[0])))throw std::runtime_error("failed to read camera profile LUT");std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");augmatch::CameraProfileLookupConfig z{};z.width=w;z.height=h;z.channels=c;z.iso=std::stof(argv[7]);z.profile={p.data(),count};z.seed=std::stoull(argv[10]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;augmatch::CameraProfileLookupPoint *dp=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMalloc(&dp,count*sizeof(p[0])));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dp,p.data(),count*sizeof(p[0]),cudaMemcpyHostToDevice));z.profile.points=dp;augmatch::camera_profile_lookup_f32(di,doo,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::camera_profile_lookup_f32(in.data(),out.data(),z);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Calibration-derived noise: calibration_noise IN OUT W H C READ_STD FPN_STD SHOT_SCALE SEED [MAP_F32].
  // An optional map fixture stores three consecutive HWC float32 maps: read sigma, FPN offset, gain.
  if((argc==11||argc==12)&&(std::string(argv[1])=="calibration_noise"||std::string(argv[1])=="calibration_frame_noise")){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<float> in(n),out(n),maps;std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");if(argc==12){maps.resize(3*n);std::ifstream fm(argv[11],std::ios::binary);fm.read(reinterpret_cast<char*>(maps.data()),static_cast<std::streamsize>(maps.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(maps.size()*sizeof(float)))throw std::runtime_error("failed to read calibration maps");}augmatch::CalibrationFrameNoiseConfig z{};z.width=w;z.height=h;z.channels=c;z.calibration.read_noise_stddev=std::stof(argv[7]);z.calibration.fpn_stddev=std::stof(argv[8]);z.calibration.shot_noise_scale=std::stof(argv[9]);z.calibration.map_count=argc==12?n:0;z.seed=std::stoull(argv[10]);if(argc==12){z.calibration.read_noise_map=maps.data();z.calibration.fpn_map=maps.data()+n;z.calibration.gain_map=maps.data()+2*n;}
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));if(argc==12){ck(cudaMalloc(&dm,maps.size()*sizeof(float)));ck(cudaMemcpy(dm,maps.data(),maps.size()*sizeof(float),cudaMemcpyHostToDevice));z.calibration.read_noise_map=dm;z.calibration.fpn_map=dm+n;z.calibration.gain_map=dm+2*n;}augmatch::calibration_frame_noise_f32(di,doo,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)ck(cudaFree(dm));ck(cudaFree(di));ck(cudaFree(doo));
#else
      augmatch::calibration_frame_noise_f32(in.data(),out.data(),z);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // ISO profile form: iso_profile IN OUT W H C ISO KNOTS_F32 COUNT SHOT_SCALE READ_STD FPN_STD SEED OP.
  // Each knot is seven little-endian float32 values in IsoNoiseProfilePoint order.
  if(argc==15&&std::string(argv[1])=="iso_profile"){
    try{
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::size_t n=static_cast<std::size_t>(w)*h*c;const std::size_t count=static_cast<std::size_t>(std::stoull(argv[9]));
      if(count==0||count>100000)throw std::invalid_argument("ISO profile knot count must be in [1,100000]");std::vector<augmatch::IsoNoiseProfilePoint> points(count);std::ifstream fp(argv[8],std::ios::binary);fp.read(reinterpret_cast<char*>(points.data()),static_cast<std::streamsize>(count*sizeof(points[0])));if(!fp||fp.gcount()!=static_cast<std::streamsize>(count*sizeof(points[0])))throw std::runtime_error("failed to read ISO profile knots");
      augmatch::IsoNoiseApplicationConfig config{};config.width=w;config.height=h;config.channels=c;config.iso=std::stof(argv[7]);config.profile={points.data(),count};config.shot_scale=std::stof(argv[10]);config.read_noise_stddev=std::stof(argv[11]);config.fpn_stddev=std::stof(argv[12]);config.seed=static_cast<std::uint64_t>(std::stoull(argv[13]));const std::string op=argv[14];
      if(op=="read"||op=="fpn"){
        std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read float input");
#if AUGMATCH_HAS_CUDA
        float *di=nullptr,*doo=nullptr;augmatch::IsoNoiseProfilePoint *dp=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMalloc(&dp,count*sizeof(points[0])));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));ck(cudaMemcpy(dp,points.data(),count*sizeof(points[0]),cudaMemcpyHostToDevice));config.profile.points=dp;if(op=="read")augmatch::iso_read_noise_f32(di,doo,config);else augmatch::iso_fpn_f32(di,doo,config);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
        if(op=="read")augmatch::iso_read_noise_f32(in.data(),out.data(),config);else augmatch::iso_fpn_f32(in.data(),out.data(),config);
#endif
        std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));
      }else{
        std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
        std::uint8_t *di=nullptr,*doo=nullptr;augmatch::IsoNoiseProfilePoint *dp=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMalloc(&dp,count*sizeof(points[0])));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));ck(cudaMemcpy(dp,points.data(),count*sizeof(points[0]),cudaMemcpyHostToDevice));config.profile.points=dp;if(op=="shot")augmatch::iso_shot_noise_u8(di,doo,config);else if(op=="black")augmatch::iso_black_level_u8(di,doo,config);else if(op=="saturation")augmatch::iso_saturation_level_u8(di,doo,config);else if(op=="levels")augmatch::iso_signal_levels_u8(di,doo,config);else throw std::invalid_argument("ISO profile OP must be shot, read, fpn, black, saturation, or levels");ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(dp));ck(cudaFree(di));ck(cudaFree(doo));
#else
        if(op=="shot")augmatch::iso_shot_noise_u8(in.data(),out.data(),config);else if(op=="black")augmatch::iso_black_level_u8(in.data(),out.data(),config);else if(op=="saturation")augmatch::iso_saturation_level_u8(in.data(),out.data(),config);else if(op=="levels")augmatch::iso_signal_levels_u8(in.data(),out.data(),config);else throw std::invalid_argument("ISO profile OP must be shot, read, fpn, black, saturation, or levels");
#endif
        std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n));
      }
      return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if ((argc==9 && (std::string(argv[1])=="jpeg_compression" || std::string(argv[1])=="image_compression" ||
                   std::string(argv[1])=="jpeg_quality_variation" || std::string(argv[1])=="jpeg_chroma_subsampling")) ||
      (argc==11 && std::string(argv[1])=="jpeg_quantization_table_variation") ||
      (argc==10 && (std::string(argv[1])=="jpeg_ringing" || std::string(argv[1])=="jpeg_blocking" ||
                    std::string(argv[1])=="jpeg_mosquito_noise")) ||
      (argc==11 && (std::string(argv[1])=="jpeg_restart_marker_damage" ||
                    std::string(argv[1])=="jpeg_progressive_decoding" ||
                    std::string(argv[1])=="jpeg_progressive_decoding_artifacts"))) {
    try {
#if AUGMATCH_HAS_JPEG
      const std::string op=argv[1];
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      if (op=="jpeg_quantization_table_variation") {
        const augmatch::JpegQuantizationTableVariationConfig config{
          w,h,c,std::stoi(argv[7]),parse_jpeg_subsampling(argv[10]),std::stof(argv[8]),std::stof(argv[9])};
        run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_quantization_table_variation_u8);
      } else if (op=="jpeg_ringing" || op=="jpeg_blocking" || op=="jpeg_mosquito_noise") {
        const augmatch::JpegArtifactConfig config{w,h,c,std::stoi(argv[7]),parse_jpeg_subsampling(argv[8]),std::stof(argv[9])};
        if (op=="jpeg_ringing") run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_ringing_u8);
        else if (op=="jpeg_blocking") run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_blocking_u8);
        else run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_mosquito_noise_u8);
      } else if (op=="jpeg_restart_marker_damage") {
        const augmatch::JpegRestartMarkerDamageConfig config{w,h,c,std::stoi(argv[7]),parse_jpeg_subsampling(argv[8]),std::stof(argv[9]),std::stoi(argv[10])};
        run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_restart_marker_damage_u8);
      } else if (op=="jpeg_progressive_decoding" || op=="jpeg_progressive_decoding_artifacts") {
        const augmatch::JpegProgressiveDecodingConfig config{w,h,c,std::stoi(argv[7]),parse_jpeg_subsampling(argv[8]),std::stof(argv[9]),std::stoi(argv[10])};
        run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_progressive_decoding_u8);
      } else {
        const augmatch::JpegCompressionConfig config{w,h,c,std::stoi(argv[7]),parse_jpeg_subsampling(argv[8])};
        if (op=="jpeg_quality_variation") run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_quality_variation_u8);
        else if (op=="jpeg_chroma_subsampling") run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_chroma_subsampling_u8);
        else run_jpeg_cli(op.c_str(),argv[2],argv[3],config,augmatch::jpeg_compression_u8);
      }
      return 0;
#else
      throw std::runtime_error("JPEG codec unavailable: configure with libjpeg-turbo/libjpeg");
#endif
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
  }
  if(argc==9&&std::string(argv[1])=="video_codec"){
    try{
#if AUGMATCH_HAS_VIDEO_CODEC
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);const std::string name=argv[7];const auto codec=name=="h264"?augmatch::VideoCodec::H264:name=="hevc"?augmatch::VideoCodec::HEVC:name=="av1"?augmatch::VideoCodec::AV1:throw std::invalid_argument("codec must be h264, hevc, or av1");
      augmatch::VideoCodecConfig cfg{w,h,c,std::stoi(argv[8]),codec};const std::size_t n=std::size_t(w)*h*c;std::vector<std::uint8_t> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));augmatch::video_codec_roundtrip_u8(di,doo,cfg);ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      augmatch::video_codec_roundtrip_u8(in.data(),out.data(),cfg);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);return fo?0:1;
#else
      throw std::runtime_error("FFmpeg video codec unavailable: configure with libavcodec, libavutil, and libswscale development files");
#endif
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if((argc==9&&(std::string(argv[1])=="webp_compression"||std::string(argv[1])=="webp")) ||
     (argc==8&&(std::string(argv[1])=="webp_lossy_compression"||std::string(argv[1])=="webp_lossless_compression"||
                std::string(argv[1])=="webp_lossy"||std::string(argv[1])=="webp_lossless"))){ 
    try{
#if AUGMATCH_HAS_WEBP
      const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),c=std::stoi(argv[6]);
      const bool explicit_lossy = std::string(argv[1])=="webp_lossy_compression" || std::string(argv[1])=="webp_lossy";
      const bool explicit_lossless = std::string(argv[1])=="webp_lossless_compression" || std::string(argv[1])=="webp_lossless";
      bool lossless = explicit_lossless;
      if(!explicit_lossy && !explicit_lossless) {
        const std::string lossless_arg=argv[8];
        if(lossless_arg!="0"&&lossless_arg!="1"&&lossless_arg!="false"&&lossless_arg!="true")
          throw std::invalid_argument("WebP lossless must be 0 or 1");
        lossless=lossless_arg=="1"||lossless_arg=="true";
      }
      const augmatch::WebPCompressionConfig config{w,h,c,std::stoi(argv[7]),lossless};
      const std::size_t n=static_cast<std::size_t>(w)*h*c;std::vector<std::uint8_t> in(n),out(n);
      std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n);if(!fi||fi.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
      std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n));ck(cudaMalloc(&doo,n));ck(cudaMemcpy(di,in.data(),n,cudaMemcpyHostToDevice));
      if(explicit_lossy) augmatch::webp_lossy_compression_u8(di,doo,config); else if(explicit_lossless) augmatch::webp_lossless_compression_u8(di,doo,config); else augmatch::webp_compression_u8(di,doo,config);
      ck(cudaMemcpy(out.data(),doo,n,cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
      if(explicit_lossy) augmatch::webp_lossy_compression_u8(in.data(),out.data(),config); else if(explicit_lossless) augmatch::webp_lossless_compression_u8(in.data(),out.data(),config); else augmatch::webp_compression_u8(in.data(),out.data(),config);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n);if(!fo)throw std::runtime_error("failed to write output");return 0;
#else
      throw std::runtime_error("WebP codec unavailable: configure with libwebp headers and library");
#endif
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Remaining video/lens catalog surrogates. Video input is float32 T-H-W-C;
  // lens and telegraph input is float32 H-W-C. Codec names are deliberately
  // absent: these operations do not parse or emit AV1/H.264/H.265.
  if((argc==11&&std::string(argv[1])=="inter_frame_compression_noise")||(argc==12&&std::string(argv[1])=="gop_keyframe_artifacts")||(argc==12&&std::string(argv[1])=="block_motion_estimation_artifacts")){
    try{const std::string op=argv[1];const int t=std::stoi(argv[4]),h=std::stoi(argv[5]),w=std::stoi(argv[6]),ch=std::stoi(argv[7]);if(t<=0||h<=0||w<=0||ch<=0)throw std::invalid_argument("invalid video dimensions");const std::size_t n=(std::size_t)t*h*w*ch;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n*sizeof(float));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read video input");
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));
      if(op=="inter_frame_compression_noise"){augmatch::InterFrameCompressionNoiseConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.quantization_step=std::stof(argv[8]);z.strength=std::stof(argv[9]);z.seed=std::stoull(argv[10]);augmatch::inter_frame_compression_noise_f32(di,doo,z);}else if(op=="gop_keyframe_artifacts"){augmatch::GopKeyframeArtifactsConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.gop_size=std::stoi(argv[8]);z.keyframe_strength=std::stof(argv[9]);z.interframe_strength=std::stof(argv[10]);z.seed=std::stoull(argv[11]);augmatch::gop_keyframe_artifacts_f32(di,doo,z);}else{augmatch::BlockMotionEstimationArtifactsConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.block_width=std::stoi(argv[8]);z.block_height=std::stoi(argv[9]);z.strength=std::stof(argv[10]);z.seed=std::stoull(argv[11]);augmatch::block_motion_estimation_artifacts_f32(di,doo,z);}ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      if(op=="inter_frame_compression_noise"){augmatch::InterFrameCompressionNoiseConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.quantization_step=std::stof(argv[8]);z.strength=std::stof(argv[9]);z.seed=std::stoull(argv[10]);augmatch::inter_frame_compression_noise_f32(in.data(),out.data(),z);}else if(op=="gop_keyframe_artifacts"){augmatch::GopKeyframeArtifactsConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.gop_size=std::stoi(argv[8]);z.keyframe_strength=std::stof(argv[9]);z.interframe_strength=std::stof(argv[10]);z.seed=std::stoull(argv[11]);augmatch::gop_keyframe_artifacts_f32(in.data(),out.data(),z);}else{augmatch::BlockMotionEstimationArtifactsConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.block_width=std::stoi(argv[8]);z.block_height=std::stoi(argv[9]);z.strength=std::stof(argv[10]);z.seed=std::stoull(argv[11]);augmatch::block_motion_estimation_artifacts_f32(in.data(),out.data(),z);}
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n*sizeof(float));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="random_telegraph_signal_noise"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),ch=std::stoi(argv[6]);const std::size_t n=(std::size_t)w*h*ch;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n*sizeof(float));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read lens input");augmatch::RandomTelegraphSignalNoiseConfig z;z.width=w;z.height=h;z.channels=ch;z.low_offset=std::stof(argv[7]);z.high_offset=std::stof(argv[8]);z.transition_probability=std::stof(argv[9]);z.initial_state=static_cast<std::uint8_t>(std::stoi(argv[10]));z.seed=std::stoull(argv[11]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));augmatch::random_telegraph_signal_noise_f32(di,doo,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      augmatch::random_telegraph_signal_noise_f32(in.data(),out.data(),z);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n*sizeof(float));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==13&&std::string(argv[1])=="water_droplets_on_lens"){
    try{const int w=std::stoi(argv[4]),h=std::stoi(argv[5]),ch=std::stoi(argv[6]);const std::size_t n=(std::size_t)w*h*ch;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n*sizeof(float));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read lens input");augmatch::WaterDropletsOnLensConfig z;z.width=w;z.height=h;z.channels=ch;z.droplet_count=std::stoi(argv[7]);z.opacity=std::stof(argv[8]);z.radius_min=std::stof(argv[9]);z.radius_max=std::stof(argv[10]);z.blur_radius=std::stoi(argv[11]);z.seed=std::stoull(argv[12]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));augmatch::water_droplets_on_lens_f32(di,doo,z);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      augmatch::water_droplets_on_lens_f32(in.data(),out.data(),z);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n*sizeof(float));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Video temporal artifacts use explicit T-H-W-C batch metadata. Optional
  // fixture files contain the borrowed mask, index, or transform arrays named
  // by the corresponding public config.
  if(argc>=9&&(std::string(argv[1])=="dead_pixel_persistence"||std::string(argv[1])=="hot_pixel_persistence"||std::string(argv[1])=="frame_drops"||std::string(argv[1])=="frame_drop"||std::string(argv[1])=="duplicate_frames"||std::string(argv[1])=="duplicate_frame"||std::string(argv[1])=="frame_blending"||std::string(argv[1])=="temporal_ghosting"||std::string(argv[1])=="motion_compensation_errors"||std::string(argv[1])=="motion_compensation_error"||std::string(argv[1])=="video_sensor_rolling_shutter"||std::string(argv[1])=="rolling_shutter_video")){
    try{
      const std::string op=argv[1]; const int t=std::stoi(argv[4]),h=std::stoi(argv[5]),w=std::stoi(argv[6]),ch=std::stoi(argv[7]); if(t<=0||h<=0||w<=0||ch<=0) throw std::invalid_argument("invalid video temporal dimensions"); const std::size_t n=static_cast<std::size_t>(t)*h*w*ch,plane=static_cast<std::size_t>(h)*w*ch; std::vector<float> in(n),out(n),weights; std::vector<std::uint8_t> mask,dropmask; std::vector<int> indices; std::vector<augmatch::TemporalGhostTransform> ghosts; std::vector<augmatch::MotionCompensationTransform> motions; std::vector<augmatch::RollingShutterTransform> rows; std::ifstream fi(argv[2],std::ios::binary); fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float))); if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float))) throw std::runtime_error("failed to read video temporal input");
      auto read_bytes=[&](const char* path,std::size_t count){std::vector<std::uint8_t> a(count);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),static_cast<std::streamsize>(count));if(!f||f.gcount()!=static_cast<std::streamsize>(count))throw std::runtime_error("failed to read video temporal mask");return a;};
      const bool dead=op=="dead_pixel_persistence",hot=op=="hot_pixel_persistence",drop=op=="frame_drops"||op=="frame_drop",dup=op=="duplicate_frames"||op=="duplicate_frame",blend=op=="frame_blending",ghost=op=="temporal_ghosting",motion=op=="motion_compensation_errors"||op=="motion_compensation_error";
      if(dead||hot){if(argc!=11&&argc!=12)throw std::invalid_argument("persistence CLI: IN OUT T H W C VALUE PROB SEED [MASK.raw]");if(argc==12)mask=read_bytes(argv[11],plane);}
      else if(drop){if(argc!=11&&argc!=12)throw std::invalid_argument("frame_drops CLI: IN OUT T H W C FILL PROB SEED [MASK.raw]");if(argc==12)dropmask=read_bytes(argv[11],t);}
      else if(dup){if(argc!=10&&argc!=11)throw std::invalid_argument("duplicate_frames CLI: IN OUT T H W C PROB SEED [INDICES.i32]");if(argc==11){indices.resize(t);std::ifstream f(argv[10],std::ios::binary);f.read(reinterpret_cast<char*>(indices.data()),static_cast<std::streamsize>(t*sizeof(int)));if(!f||f.gcount()!=static_cast<std::streamsize>(t*sizeof(int)))throw std::runtime_error("failed to read duplicate indices");}}
      else if(blend){if(argc!=9&&argc!=10)throw std::invalid_argument("frame_blending CLI: IN OUT T H W C WEIGHT [WEIGHTS.f32]");if(argc==10){weights.resize(t);std::ifstream f(argv[9],std::ios::binary);f.read(reinterpret_cast<char*>(weights.data()),static_cast<std::streamsize>(t*sizeof(float)));if(!f||f.gcount()!=static_cast<std::streamsize>(t*sizeof(float)))throw std::runtime_error("failed to read blend weights");}}
      else if(ghost||motion){if(argc!=12&&argc!=13)throw std::invalid_argument("transform CLI: IN OUT T H W C OFFSET DX DY WEIGHT [TRANSFORMS.bin]");if(argc==13){if(ghost){ghosts.resize(t);std::ifstream f(argv[12],std::ios::binary);f.read(reinterpret_cast<char*>(ghosts.data()),static_cast<std::streamsize>(t*sizeof(ghosts[0])));if(!f||f.gcount()!=static_cast<std::streamsize>(t*sizeof(ghosts[0])))throw std::runtime_error("failed to read ghost transforms");}else{motions.resize(t);std::ifstream f(argv[12],std::ios::binary);f.read(reinterpret_cast<char*>(motions.data()),static_cast<std::streamsize>(t*sizeof(motions[0])));if(!f||f.gcount()!=static_cast<std::streamsize>(t*sizeof(motions[0])))throw std::runtime_error("failed to read compensation transforms");}}}
      else {if(op!="video_sensor_rolling_shutter"&&op!="rolling_shutter_video")throw std::invalid_argument("unknown video temporal operation");if(argc!=11&&argc!=12)throw std::invalid_argument("video_sensor_rolling_shutter CLI: IN OUT T H W C DX DY READOUT [ROWS.bin]");if(argc==12){rows.resize(static_cast<std::size_t>(t)*h);std::ifstream f(argv[11],std::ios::binary);f.read(reinterpret_cast<char*>(rows.data()),static_cast<std::streamsize>(rows.size()*sizeof(rows[0])));if(!f||f.gcount()!=static_cast<std::streamsize>(rows.size()*sizeof(rows[0])))throw std::runtime_error("failed to read rolling-shutter rows");}}
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr,*dw=nullptr;int *dix=nullptr;augmatch::TemporalGhostTransform *dgt=nullptr;augmatch::MotionCompensationTransform *dmt=nullptr;augmatch::RollingShutterTransform *drt=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));
      if(!mask.empty()){ck(cudaMalloc(&dm,mask.size()));ck(cudaMemcpy(dm,mask.data(),mask.size(),cudaMemcpyHostToDevice));}if(!dropmask.empty()){ck(cudaMalloc(&dm,dropmask.size()));ck(cudaMemcpy(dm,dropmask.data(),dropmask.size(),cudaMemcpyHostToDevice));}if(!weights.empty()){ck(cudaMalloc(&dw,weights.size()*sizeof(float)));ck(cudaMemcpy(dw,weights.data(),weights.size()*sizeof(float),cudaMemcpyHostToDevice));}if(!indices.empty()){ck(cudaMalloc(&dix,indices.size()*sizeof(int)));ck(cudaMemcpy(dix,indices.data(),indices.size()*sizeof(int),cudaMemcpyHostToDevice));}if(!ghosts.empty()){ck(cudaMalloc(&dgt,ghosts.size()*sizeof(ghosts[0])));ck(cudaMemcpy(dgt,ghosts.data(),ghosts.size()*sizeof(ghosts[0]),cudaMemcpyHostToDevice));}if(!motions.empty()){ck(cudaMalloc(&dmt,motions.size()*sizeof(motions[0])));ck(cudaMemcpy(dmt,motions.data(),motions.size()*sizeof(motions[0]),cudaMemcpyHostToDevice));}if(!rows.empty()){ck(cudaMalloc(&drt,rows.size()*sizeof(rows[0])));ck(cudaMemcpy(drt,rows.data(),rows.size()*sizeof(rows[0]),cudaMemcpyHostToDevice));}
      if(dead){augmatch::DeadPixelPersistenceConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.dead_value=std::stof(argv[8]);z.probability=std::stof(argv[9]);z.seed=std::stoull(argv[10]);z.mask=reinterpret_cast<const std::uint8_t*>(dm);augmatch::dead_pixel_persistence_f32(di,doo,z);}else if(hot){augmatch::HotPixelPersistenceConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.hot_value=std::stof(argv[8]);z.probability=std::stof(argv[9]);z.seed=std::stoull(argv[10]);z.mask=reinterpret_cast<const std::uint8_t*>(dm);augmatch::hot_pixel_persistence_f32(di,doo,z);}else if(drop){augmatch::FrameDropConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.fill_value=std::stof(argv[8]);z.probability=std::stof(argv[9]);z.seed=std::stoull(argv[10]);z.drop_mask=reinterpret_cast<const std::uint8_t*>(dm);augmatch::frame_drops_f32(di,doo,z);}else if(dup){augmatch::DuplicateFrameConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.probability=std::stof(argv[8]);z.seed=std::stoull(argv[9]);z.source_indices=dix;augmatch::duplicate_frames_f32(di,doo,z);}else if(blend){augmatch::FrameBlendingConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.weight=std::stof(argv[8]);z.weights=dw;augmatch::frame_blending_f32(di,doo,z);}else if(ghost){augmatch::TemporalGhostingConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.source_frame_offset=std::stoi(argv[8]);z.dx=std::stoi(argv[9]);z.dy=std::stoi(argv[10]);z.alpha=std::stof(argv[11]);z.transforms=dgt;augmatch::temporal_ghosting_f32(di,doo,z);}else if(motion){augmatch::MotionCompensationErrorConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.source_frame_offset=std::stoi(argv[8]);z.dx=std::stoi(argv[9]);z.dy=std::stoi(argv[10]);z.weight=std::stof(argv[11]);z.transforms=dmt;augmatch::motion_compensation_errors_f32(di,doo,z);}else{augmatch::VideoRollingShutterConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.motion_dx=std::stof(argv[8]);z.motion_dy=std::stof(argv[9]);z.readout_fraction=std::stof(argv[10]);z.transforms=drt;augmatch::video_sensor_rolling_shutter_f32(di,doo,z);}ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);if(dw)cudaFree(dw);if(dix)cudaFree(dix);if(dgt)cudaFree(dgt);if(dmt)cudaFree(dmt);if(drt)cudaFree(drt);cudaFree(di);cudaFree(doo);
#else
      if(dead){augmatch::DeadPixelPersistenceConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.dead_value=std::stof(argv[8]);z.probability=std::stof(argv[9]);z.seed=std::stoull(argv[10]);z.mask=mask.empty()?nullptr:mask.data();augmatch::dead_pixel_persistence_f32(in.data(),out.data(),z);}else if(hot){augmatch::HotPixelPersistenceConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.hot_value=std::stof(argv[8]);z.probability=std::stof(argv[9]);z.seed=std::stoull(argv[10]);z.mask=mask.empty()?nullptr:mask.data();augmatch::hot_pixel_persistence_f32(in.data(),out.data(),z);}else if(drop){augmatch::FrameDropConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.fill_value=std::stof(argv[8]);z.probability=std::stof(argv[9]);z.seed=std::stoull(argv[10]);z.drop_mask=dropmask.empty()?nullptr:dropmask.data();augmatch::frame_drops_f32(in.data(),out.data(),z);}else if(dup){augmatch::DuplicateFrameConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.probability=std::stof(argv[8]);z.seed=std::stoull(argv[9]);z.source_indices=indices.empty()?nullptr:indices.data();augmatch::duplicate_frames_f32(in.data(),out.data(),z);}else if(blend){augmatch::FrameBlendingConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.weight=std::stof(argv[8]);z.weights=weights.empty()?nullptr:weights.data();augmatch::frame_blending_f32(in.data(),out.data(),z);}else if(ghost){augmatch::TemporalGhostingConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.source_frame_offset=std::stoi(argv[8]);z.dx=std::stoi(argv[9]);z.dy=std::stoi(argv[10]);z.alpha=std::stof(argv[11]);z.transforms=ghosts.empty()?nullptr:ghosts.data();augmatch::temporal_ghosting_f32(in.data(),out.data(),z);}else if(motion){augmatch::MotionCompensationErrorConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.source_frame_offset=std::stoi(argv[8]);z.dx=std::stoi(argv[9]);z.dy=std::stoi(argv[10]);z.weight=std::stof(argv[11]);z.transforms=motions.empty()?nullptr:motions.data();augmatch::motion_compensation_errors_f32(in.data(),out.data(),z);}else{augmatch::VideoRollingShutterConfig z;z.frames=t;z.height=h;z.width=w;z.channels=ch;z.motion_dx=std::stof(argv[8]);z.motion_dy=std::stof(argv[9]);z.readout_fraction=std::stof(argv[10]);z.transforms=rows.empty()?nullptr:rows.data();augmatch::video_sensor_rolling_shutter_f32(in.data(),out.data(),z);}
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc==12&&std::string(argv[1])=="row_noise_phase_changes"){
    try{const int t=std::stoi(argv[4]),h=std::stoi(argv[5]),w=std::stoi(argv[6]),ch=std::stoi(argv[7]);const std::size_t n=(std::size_t)t*h*w*ch;std::vector<float> in(n),out(n);std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),n*sizeof(float));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read temporal input");augmatch::RowNoisePhaseChangesConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.row_stddev=std::stof(argv[8]);c.phase_change_probability=std::stof(argv[9]);c.temporal_correlation=std::stof(argv[10]);c.seed=std::stoull(argv[11]);
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));augmatch::row_noise_phase_changes_f32(di,doo,c);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(di);cudaFree(doo);
#else
      augmatch::row_noise_phase_changes_f32(in.data(),out.data(),c);
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),n*sizeof(float));return fo?0:1;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  // Temporal operations use contiguous float32 T-H-W-C batches.  The CLI
  // mirrors the public ownership contract; CUDA builds stage host fixtures to
  // device memory before invoking the asynchronous API.
  if(argc>=11&&(std::string(argv[1])=="temporal_gaussian"||std::string(argv[1])=="temporal_gaussian_noise"||std::string(argv[1])=="temporal_shot"||std::string(argv[1])=="temporally_correlated_shot_noise"||std::string(argv[1])=="temporal_read_noise"||std::string(argv[1])=="temporal_read_noise_correlation"||std::string(argv[1])=="flicker"||std::string(argv[1])=="exposure_flicker"||std::string(argv[1])=="gain_flicker"||std::string(argv[1])=="white_balance_flicker"||std::string(argv[1])=="fixed_pattern_noise_drift"||std::string(argv[1])=="temporal_fpn_drift")){
    try{
      const std::string op=argv[1];const int t=std::stoi(argv[4]),h=std::stoi(argv[5]),w=std::stoi(argv[6]),ch=std::stoi(argv[7]);if(t<=0||h<=0||w<=0||ch<=0)throw std::invalid_argument("invalid temporal dimensions");const std::size_t n=static_cast<std::size_t>(t)*h*w*ch;std::vector<float> in(n),out(n),map;std::ifstream fi(argv[2],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),static_cast<std::streamsize>(n*sizeof(float)));if(!fi||fi.gcount()!=static_cast<std::streamsize>(n*sizeof(float)))throw std::runtime_error("failed to read temporal input");
      const bool wb=op=="white_balance_flicker",drift=op=="fixed_pattern_noise_drift"||op=="temporal_fpn_drift";if(drift&&argc==13){map.resize(static_cast<std::size_t>(h)*w*ch);std::ifstream fm(argv[12],std::ios::binary);fm.read(reinterpret_cast<char*>(map.data()),static_cast<std::streamsize>(map.size()*sizeof(float)));if(!fm||fm.gcount()!=static_cast<std::streamsize>(map.size()*sizeof(float)))throw std::runtime_error("failed to read FPN map");}
      if(wb&&argc!=13)throw std::invalid_argument("white_balance_flicker requires red green blue rho seed");if(!wb&&!drift&&argc!=11)throw std::invalid_argument("invalid temporal CLI arguments");if(drift&&(argc!=12&&argc!=13))throw std::invalid_argument("invalid FPN drift CLI arguments");
#if AUGMATCH_HAS_CUDA
      float *di=nullptr,*doo=nullptr,*dm=nullptr;ck(cudaMalloc(&di,n*sizeof(float)));ck(cudaMalloc(&doo,n*sizeof(float)));ck(cudaMemcpy(di,in.data(),n*sizeof(float),cudaMemcpyHostToDevice));
      if(!map.empty()){ck(cudaMalloc(&dm,map.size()*sizeof(float)));ck(cudaMemcpy(dm,map.data(),map.size()*sizeof(float),cudaMemcpyHostToDevice));}
      if(wb){augmatch::WhiteBalanceFlickerConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.red_stddev=std::stof(argv[8]);c.green_stddev=std::stof(argv[9]);c.blue_stddev=std::stof(argv[10]);c.temporal_correlation=std::stof(argv[11]);c.seed=std::stoull(argv[12]);augmatch::white_balance_flicker_f32(di,doo,c);}
      else if(drift){augmatch::FixedPatternNoiseDriftConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.base_stddev=std::stof(argv[8]);c.drift_stddev=std::stof(argv[9]);c.temporal_correlation=std::stof(argv[10]);c.seed=std::stoull(argv[11]);c.base_map=dm;augmatch::fixed_pattern_noise_drift_f32(di,doo,c);}
      else {const float a=std::stof(argv[8]),rho=std::stof(argv[9]);const std::uint64_t seed=std::stoull(argv[10]);if(op=="temporal_gaussian"||op=="temporal_gaussian_noise"){augmatch::TemporalGaussianNoiseConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.stddev=a;c.temporal_correlation=rho;c.seed=seed;augmatch::temporal_gaussian_noise_f32(di,doo,c);}else if(op=="temporal_shot"||op=="temporally_correlated_shot_noise"){augmatch::TemporallyCorrelatedShotNoiseConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.photons_per_unit=a;c.temporal_correlation=rho;c.seed=seed;augmatch::temporally_correlated_shot_noise_f32(di,doo,c);}else if(op=="temporal_read_noise"||op=="temporal_read_noise_correlation"){augmatch::TemporalReadNoiseCorrelationConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.stddev=a;c.temporal_correlation=rho;c.seed=seed;augmatch::temporal_read_noise_correlation_f32(di,doo,c);}else if(op=="flicker"||op=="exposure_flicker"||op=="gain_flicker"){augmatch::FlickerConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.stddev=a;c.temporal_correlation=rho;c.seed=seed;if(op=="exposure_flicker"){augmatch::ExposureFlickerConfig z;static_cast<augmatch::TemporalBatchConfig&>(z)=static_cast<augmatch::TemporalBatchConfig&>(c);z.stddev=a;augmatch::exposure_flicker_f32(di,doo,z);}else if(op=="gain_flicker"){augmatch::GainFlickerConfig z;static_cast<augmatch::TemporalBatchConfig&>(z)=static_cast<augmatch::TemporalBatchConfig&>(c);z.stddev=a;augmatch::gain_flicker_f32(di,doo,z);}else augmatch::flicker_f32(di,doo,c);}}
      ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,n*sizeof(float),cudaMemcpyDeviceToHost));if(dm)cudaFree(dm);cudaFree(di);cudaFree(doo);
#else
      if(wb){augmatch::WhiteBalanceFlickerConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.red_stddev=std::stof(argv[8]);c.green_stddev=std::stof(argv[9]);c.blue_stddev=std::stof(argv[10]);c.temporal_correlation=std::stof(argv[11]);c.seed=std::stoull(argv[12]);augmatch::white_balance_flicker_f32(in.data(),out.data(),c);}else if(drift){augmatch::FixedPatternNoiseDriftConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.base_stddev=std::stof(argv[8]);c.drift_stddev=std::stof(argv[9]);c.temporal_correlation=std::stof(argv[10]);c.seed=std::stoull(argv[11]);c.base_map=map.empty()?nullptr:map.data();augmatch::fixed_pattern_noise_drift_f32(in.data(),out.data(),c);}else {const float a=std::stof(argv[8]),rho=std::stof(argv[9]);const std::uint64_t seed=std::stoull(argv[10]);if(op=="temporal_gaussian"||op=="temporal_gaussian_noise"){augmatch::TemporalGaussianNoiseConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.stddev=a;c.temporal_correlation=rho;c.seed=seed;augmatch::temporal_gaussian_noise_f32(in.data(),out.data(),c);}else if(op=="temporal_shot"||op=="temporally_correlated_shot_noise"){augmatch::TemporallyCorrelatedShotNoiseConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.photons_per_unit=a;c.temporal_correlation=rho;c.seed=seed;augmatch::temporally_correlated_shot_noise_f32(in.data(),out.data(),c);}else if(op=="temporal_read_noise"||op=="temporal_read_noise_correlation"){augmatch::TemporalReadNoiseCorrelationConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.stddev=a;c.temporal_correlation=rho;c.seed=seed;augmatch::temporal_read_noise_correlation_f32(in.data(),out.data(),c);}else {augmatch::FlickerConfig c;c.frames=t;c.height=h;c.width=w;c.channels=ch;c.stddev=a;c.temporal_correlation=rho;c.seed=seed;if(op=="exposure_flicker"){augmatch::ExposureFlickerConfig z;static_cast<augmatch::TemporalBatchConfig&>(z)=static_cast<augmatch::TemporalBatchConfig&>(c);z.stddev=a;augmatch::exposure_flicker_f32(in.data(),out.data(),z);}else if(op=="gain_flicker"){augmatch::GainFlickerConfig z;static_cast<augmatch::TemporalBatchConfig&>(z)=static_cast<augmatch::TemporalBatchConfig&>(c);z.stddev=a;augmatch::gain_flicker_f32(in.data(),out.data(),z);}else augmatch::flicker_f32(in.data(),out.data(),c);}}
#endif
      std::ofstream fo(argv[3],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),static_cast<std::streamsize>(n*sizeof(float)));return fo?0:1;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
  }
  if(argc!=12&&argc!=18){std::cerr<<"usage: cli IN.raw OUT.raw W H C X Y CW CH FLIP_H FLIP_V [BOX_X1 BOX_Y1 BOX_X2 BOX_Y2 POINT_X POINT_Y]\\n";return 2;}
  try{
    int w=std::stoi(argv[3]),h=std::stoi(argv[4]),c=std::stoi(argv[5]);
    augmatch::Geometry g{w,h,c,std::stoi(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),std::stoi(argv[9]),std::stoi(argv[10])!=0,std::stoi(argv[11])!=0};
    std::vector<std::uint8_t> in(size_t(w)*h*c),out(size_t(g.output_width)*g.output_height*c);
    std::ifstream fi(argv[1],std::ios::binary);fi.read(reinterpret_cast<char*>(in.data()),in.size());if(!fi||fi.gcount()!=static_cast<std::streamsize>(in.size()))throw std::runtime_error("failed to read raw input");
#if AUGMATCH_HAS_CUDA
    std::uint8_t *di=nullptr,*doo=nullptr;ck(cudaMalloc(&di,in.size()));ck(cudaMalloc(&doo,out.size()));ck(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));augmatch::transform_u8(di,doo,g);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,out.size(),cudaMemcpyDeviceToHost));ck(cudaFree(di));ck(cudaFree(doo));
#else
    augmatch::transform_u8(in.data(),out.data(),g);
#endif
    std::ofstream fo(argv[2],std::ios::binary);fo.write(reinterpret_cast<const char*>(out.data()),out.size());if(!fo)throw std::runtime_error("failed to write output");
    if(argc==18){augmatch::BoxXYXY b{std::stof(argv[12]),std::stof(argv[13]),std::stof(argv[14]),std::stof(argv[15])};augmatch::PointXY p{std::stof(argv[16]),std::stof(argv[17])};b=augmatch::transform_box(b,g);p=augmatch::transform_point(p,g);std::cout.precision(9);std::cout<<b.x1<<' '<<b.y1<<' '<<b.x2<<' '<<b.y2<<' '<<p.x<<' '<<p.y<<'\n';}
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
