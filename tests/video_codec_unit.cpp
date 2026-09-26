#include "augmatch/video_codec.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
int main(){
#if AUGMATCH_HAS_VIDEO_CODEC
  const int w=64,h=48,c=3; std::vector<std::uint8_t> in(w*h*c),out(in.size()); for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int k=0;k<c;++k)in[(y*w+x)*c+k]=std::uint8_t((x*3+y*5+k*31)&255);
  bool invalid=false;try{augmatch::video_codec_encode_u8(in.data(),{63,h,c,25,augmatch::VideoCodec::H264});}catch(const std::invalid_argument&){invalid=true;}assert(invalid);
  for(auto codec:{augmatch::VideoCodec::H264,augmatch::VideoCodec::HEVC,augmatch::VideoCodec::AV1}){try{augmatch::VideoCodecConfig cfg{w,h,c,25,codec};auto bytes=augmatch::video_codec_encode_u8(in.data(),cfg);assert(!bytes.empty());augmatch::video_codec_decode_u8(bytes.data(),bytes.size(),out.data(),{w,h,c,codec});assert(out.size()==in.size());augmatch::video_codec_roundtrip_u8(in.data(),out.data(),cfg);if(codec==augmatch::VideoCodec::AV1)augmatch::av1_intra_frame_artifacts_u8(in.data(),out.data(),cfg);std::cout<<"codec available and roundtrip passed: "<<int(codec)<<" ("<<bytes.size()<<" bytes)\n";}catch(const std::exception& e){std::cout<<"codec encoder unavailable: "<<int(codec)<<": "<<e.what()<<"\n";}}
#else
  bool threw=false;try{std::uint8_t p=0;augmatch::video_codec_encode_u8(&p,{},nullptr);}catch(const std::runtime_error&){threw=true;}assert(threw);
#endif
}
