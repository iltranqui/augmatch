#include "video_codec_impl.hpp"
namespace augmatch {
std::vector<std::uint8_t> video_codec_encode_u8(const std::uint8_t* p,const VideoCodecConfig& c,cudaStream_t){return video_detail::encode(p,c);}
void video_codec_decode_u8(const std::uint8_t* p,std::size_t n,std::uint8_t* o,const VideoDecodeConfig& c,cudaStream_t){video_detail::decode(p,n,o,c);}
void video_codec_roundtrip_u8(const std::uint8_t* p,std::uint8_t* o,const VideoCodecConfig& c,cudaStream_t s){auto b=video_codec_encode_u8(p,c,s);video_codec_decode_u8(b.data(),b.size(),o,{c.width,c.height,c.channels,c.codec},s);}
void av1_intra_frame_artifacts_u8(const std::uint8_t* p,std::uint8_t* o,const VideoCodecConfig& c,cudaStream_t s){if(c.codec!=VideoCodec::AV1)throw std::invalid_argument("AV1 artifact API requires AV1 codec");video_codec_roundtrip_u8(p,o,c,s);}
}
