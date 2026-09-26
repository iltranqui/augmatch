#include "augmatch/canny.hpp"
#include <cstdint>
#include <stdexcept>
#include <vector>
int main(){
  const int w=32,h=24,ch=3; std::vector<std::uint8_t> image(static_cast<std::size_t>(w)*h*ch,0),a(image.size()),b(image.size());
  for(int y=6;y<18;++y)for(int x=8;x<24;++x)for(int c=0;c<ch;++c)image[(static_cast<std::size_t>(y)*w+x)*ch+c]=static_cast<std::uint8_t>(c==0?220:100);
  augmatch::CannyConfig cfg{w,h,ch,20.0f,80.0f,3,augmatch::BorderPolicy::Clamp,0};
  augmatch::canny_u8(image.data(),a.data(),cfg); augmatch::canny_u8(image.data(),b.data(),cfg);
  if(a!=b)return 1;
  bool edge=false;for(std::size_t i=0;i<a.size();i+=ch){if(a[i]!=0&&a[i]!=255)return 2;for(int c=1;c<ch;++c)if(a[i+c]!=a[i])return 3;if(a[i])edge=true;}if(!edge)return 4;
  std::vector<std::uint8_t> flat(image.size(),17),flat_out(image.size());augmatch::canny_u8(flat.data(),flat_out.data(),cfg);for(auto v:flat_out)if(v!=0)return 5;
  bool rejected=false;try{auto bad=cfg;bad.high_threshold=10;augmatch::canny_u8(image.data(),a.data(),bad);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 6;
  return 0;
}
