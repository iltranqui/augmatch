#include "augmatch/filter/superpixels.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace augmatch {
void superpixels_u8(const std::uint8_t* in,std::uint8_t* out,const SuperpixelConfig& c,cudaStream_t){
  if(!in||!out||c.width<=0||c.height<=0||c.channels<=0||c.cell_width<=0||c.cell_height<=0)
    throw std::invalid_argument("invalid superpixel configuration");
  for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x) {
    const int x0=(x/c.cell_width)*c.cell_width;
    const int y0=(y/c.cell_height)*c.cell_height;
    const int x1=(c.cell_width>c.width-x0)?c.width:x0+c.cell_width;
    const int y1=(c.cell_height>c.height-y0)?c.height:y0+c.cell_height;
    const std::uint64_t count=static_cast<std::uint64_t>(x1-x0)*static_cast<std::uint64_t>(y1-y0);
    for(int ch=0;ch<c.channels;++ch) {
      std::uint64_t sum=0;
      for(int py=y0;py<y1;++py) for(int px=x0;px<x1;++px)
        sum+=in[(static_cast<std::size_t>(py)*c.width+px)*c.channels+ch];
      const std::uint8_t value=static_cast<std::uint8_t>((sum+count/2u)/count);
      out[(static_cast<std::size_t>(y)*c.width+x)*c.channels+ch]=value;
    }
  }
}
}
