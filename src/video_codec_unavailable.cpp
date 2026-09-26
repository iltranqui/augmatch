#include "augmatch/video/video_codec.hpp"
#include <stdexcept>
namespace augmatch { namespace { [[noreturn]] void unavailable(){throw std::runtime_error("FFmpeg video codec unavailable: configure with libavcodec, libavutil, and libswscale development files");} }
std::vector<std::uint8_t> video_codec_encode_u8(const std::uint8_t*,const VideoCodecConfig&,cudaStream_t){unavailable();}
void video_codec_decode_u8(const std::uint8_t*,std::size_t,std::uint8_t*,const VideoDecodeConfig&,cudaStream_t){unavailable();}
void video_codec_roundtrip_u8(const std::uint8_t*,std::uint8_t*,const VideoCodecConfig&,cudaStream_t){unavailable();}
void av1_intra_frame_artifacts_u8(const std::uint8_t*,std::uint8_t*,const VideoCodecConfig&,cudaStream_t){unavailable();}
}
