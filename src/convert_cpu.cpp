#include "augmatch/core/convert.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace augmatch { namespace {void check(int w,int h,int c){if(w<=0||h<=0||c<=0)throw std::invalid_argument("invalid conversion dimensions");} std::uint8_t sat(float x){return static_cast<std::uint8_t>(std::lround(std::max(0.f,std::min(255.f,x))));}}
void to_float_u8(const std::uint8_t* in,float* out,const ConvertConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null to-float buffer");for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=in[i]/255.f;}
void from_float_u8(const float* in,std::uint8_t* out,const ConvertConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out)throw std::invalid_argument("null from-float buffer");for(int i=0,n=c.width*c.height*c.channels;i<n;++i)out[i]=sat(in[i]*255.f);}
void normalize_u8_f32(const std::uint8_t* in,float* out,const NormalizeConfig& c,cudaStream_t){check(c.width,c.height,c.channels);if(!in||!out||c.std0==0||c.std1==0||c.std2==0)throw std::invalid_argument("invalid normalize configuration");for(int i=0,n=c.width*c.height*c.channels;i<n;++i){int ch=i%c.channels;float mean=ch==0?c.mean0:ch==1?c.mean1:c.mean2;float st=ch==0?c.std0:ch==1?c.std1:c.std2;out[i]=(in[i]/255.f-mean)/st;}}
namespace {
void check_tensor_config(const ToTensorV2Config& c) {
  check(c.width, c.height, c.channels);
  if (c.normalize && (c.std0 == 0.0f || c.std1 == 0.0f || c.std2 == 0.0f ||
                      !std::isfinite(c.mean0) || !std::isfinite(c.mean1) || !std::isfinite(c.mean2) ||
                      !std::isfinite(c.std0) || !std::isfinite(c.std1) || !std::isfinite(c.std2)))
    throw std::invalid_argument("invalid ToTensorV2 normalization configuration");
}
float tensor_value(std::uint8_t value, int channel, const ToTensorV2Config& c) {
  const float normalized = value / 255.0f;
  if (!c.normalize) return normalized;
  const float mean = channel == 0 ? c.mean0 : channel == 1 ? c.mean1 : c.mean2;
  const float st = channel == 0 ? c.std0 : channel == 1 ? c.std1 : c.std2;
  return (normalized - mean) / st;
}
void check_tensor_views(const ImageView& input, const MutableTensorViewF32& output) {
  if (!input.valid() || input.type != DataType::UInt8 || input.layout != Layout::HWC || !input.contiguous())
    throw std::invalid_argument("ToTensorV2 requires contiguous HWC uint8 input");
  if (!output.valid() || !output.contiguous())
    throw std::invalid_argument("ToTensorV2 requires contiguous CHW float32 output");
  if (input.width != output.width || input.height != output.height || input.channels != output.channels)
    throw std::invalid_argument("ToTensorV2 input and output shapes differ");
  if (input.memory != MemorySpace::Host || output.memory != MemorySpace::Host)
    throw std::invalid_argument("CPU ToTensorV2 requires host views");
}
}
void to_tensor_v2_u8_f32(const std::uint8_t* in, float* out, const ToTensorV2Config& c, cudaStream_t){
  check_tensor_config(c);
  if (!in || !out) throw std::invalid_argument("null ToTensorV2 buffer");
  for (int channel = 0; channel < c.channels; ++channel)
    for (int y = 0; y < c.height; ++y)
      for (int x = 0; x < c.width; ++x) {
        const std::size_t input_index = (static_cast<std::size_t>(y) * c.width + x) * c.channels + channel;
        const std::size_t output_index = (static_cast<std::size_t>(channel) * c.height + y) * c.width + x;
        out[output_index] = tensor_value(in[input_index], channel, c);
      }
}
void to_tensor_v2(const std::uint8_t* in, float* out, const ToTensorV2Config& c, cudaStream_t stream){
  to_tensor_v2_u8_f32(in, out, c, stream);
}
void to_tensor_chw_f32(const std::uint8_t* in, float* out, const ToTensorV2Config& c, cudaStream_t stream){
  to_tensor_v2_u8_f32(in, out, c, stream);
}
void to_tensor_v2(const ImageView& input, const MutableTensorViewF32& output, const TensorNormalizeConfig& n, cudaStream_t stream){
  check_tensor_views(input, output);
  const ToTensorV2Config c{input.width, input.height, input.channels, n.enabled, n.mean0, n.mean1, n.mean2, n.std0, n.std1, n.std2};
  to_tensor_v2_u8_f32(static_cast<const std::uint8_t*>(input.data), output.data, c, stream);
}
namespace {
void check_tensor_3d_config(const ToTensor3DConfig& c) {
  check(c.width, c.height, c.channels);
  if (c.depth <= 0) throw std::invalid_argument("invalid ToTensor3D dimensions");
  if (c.normalize && (c.std0 == 0.0f || c.std1 == 0.0f || c.std2 == 0.0f ||
                      !std::isfinite(c.mean0) || !std::isfinite(c.mean1) || !std::isfinite(c.mean2) ||
                      !std::isfinite(c.std0) || !std::isfinite(c.std1) || !std::isfinite(c.std2)))
    throw std::invalid_argument("invalid ToTensor3D normalization configuration");
}
float tensor_3d_value(std::uint8_t value, int channel, const ToTensor3DConfig& c) {
  const float normalized = value / 255.0f;
  if (!c.normalize) return normalized;
  const float mean = channel == 0 ? c.mean0 : channel == 1 ? c.mean1 : c.mean2;
  const float st = channel == 0 ? c.std0 : channel == 1 ? c.std1 : c.std2;
  return (normalized - mean) / st;
}
}
void to_tensor_3d_u8_f32(const std::uint8_t* in, float* out, const ToTensor3DConfig& c, cudaStream_t){
  check_tensor_3d_config(c);
  if (!in || !out) throw std::invalid_argument("null ToTensor3D buffer");
  for (int channel = 0; channel < c.channels; ++channel)
    for (int depth = 0; depth < c.depth; ++depth)
      for (int y = 0; y < c.height; ++y)
        for (int x = 0; x < c.width; ++x) {
          const std::size_t input_index =
              ((static_cast<std::size_t>(depth) * c.height + y) * c.width + x) * c.channels + channel;
          const std::size_t output_index =
              ((static_cast<std::size_t>(channel) * c.depth + depth) * c.height + y) * c.width + x;
          out[output_index] = tensor_3d_value(in[input_index], channel, c);
        }
}
void to_tensor_3d(const std::uint8_t* in, float* out, const ToTensor3DConfig& c, cudaStream_t stream){
  to_tensor_3d_u8_f32(in, out, c, stream);
}
void to_tensor_cdhw_f32(const std::uint8_t* in, float* out, const ToTensor3DConfig& c, cudaStream_t stream){
  to_tensor_3d_u8_f32(in, out, c, stream);
}
}
