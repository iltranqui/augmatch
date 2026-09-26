#include "augmatch/transport.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
int main() {
  using namespace augmatch;
  std::vector<std::uint8_t> bytes{0,1,2,3,4,5}; std::vector<std::uint8_t> out(bytes.size());
  truncated_frame_u8(bytes.data(),out.data(),{bytes.size(),3,255});
  assert((out==std::vector<std::uint8_t>{0,1,2,255,255,255}));
  bit_flips_u8(bytes.data(),out.data(),{bytes.size(),0,0,0,0}); assert(out==bytes);
  std::vector<std::uint8_t> frame(4*4,0); for(std::size_t i=0;i<frame.size();++i) frame[i]=static_cast<std::uint8_t>(i*16);
  block_quantization_u8(frame.data(),out.data(),{2,2,1,2,32}); // output buffer is intentionally too short? use a frame-sized output below.
  std::vector<std::uint8_t> f2(frame.size()); block_quantization_u8(frame.data(),f2.data(),{4,4,1,2,32}); for(auto v:f2) assert(v==255||v%32==0);
  std::vector<std::uint8_t> mask(4,0); mask[1]=1; packet_loss_macroblocks_u8(frame.data(),f2.data(),{4,4,1,2,1,1,0,99,mask.data(),0});
  assert(f2[2]==99 && f2[3]==99 && f2[0]==0);
  std::vector<std::uint8_t> plane{1,2,3,4,5,6}; std::vector<std::uint8_t> shifted(plane.size()); chroma_plane_misalignment_u8(plane.data(),shifted.data(),{3,2,3,1,0,0}); assert((shifted==std::vector<std::uint8_t>{1,1,2,4,4,5}));
  std::vector<std::uint16_t> raw{1,2,3,4}; std::vector<std::uint16_t> rawout(4); std::vector<std::uint8_t> rawmask{0,1,0,0}; RawBayerTransportCorruptionConfig rc{2,2,2,12,0,77,0,rawmask.data(),0}; raw_bayer_transport_corruption_u16(raw.data(),rawout.data(),rc); assert(rawout[1]==77 && rawout[0]==1);
  std::cout << "transport unit passed\n"; return 0;
}
