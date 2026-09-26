#include "augmatch/noise.hpp"
#include <cmath>
#include <vector>
int main(){
  const int w=3,h=2,c=3; std::vector<float> in(w*h*c,0.5f),out(in.size());
  augmatch::LensVignettingConfig v; v.width=w;v.height=h;v.channels=c;v.c0=1;v.c1=0;augmatch::lens_vignetting_f32(in.data(),out.data(),v);if(out!=in)return 1;
  std::vector<float> map{1,0.5f,1,1,1,1};v.map=map.data();augmatch::lens_vignetting_f32(in.data(),out.data(),v);if(std::abs(out[3]-0.25f)>1e-6f)return 2;
  augmatch::ColorDependentVignettingConfig cv;static_cast<augmatch::LensVignettingConfig&>(cv)=v;std::vector<float> coeff{1,0,0,0, 0.5f,0,0,0, 1,0,0,0};cv.coefficients=coeff.data();cv.map=nullptr;augmatch::color_dependent_vignetting_f32(in.data(),out.data(),cv);if(std::abs(out[1]-0.25f)>1e-6f)return 3;
  augmatch::LensShadingConfig ls{w,h,c,map.data(),1,1,0,1};augmatch::lens_shading_f32(in.data(),out.data(),ls);if(std::abs(out[3]-0.25f)>1e-6f)return 4;
  augmatch::UnevenIlluminationConfig ui{w,h,c,map.data(),1,1,0,0,1};augmatch::uneven_illumination_f32(in.data(),out.data(),ui);if(std::abs(out[3]-0.25f)>1e-6f)return 5;
  augmatch::SensorLensDustShadowsConfig ds{w,h,c,map.data(),1,0,1};augmatch::sensor_lens_dust_shadows_f32(in.data(),out.data(),ds);if(std::abs(out[3]-0.25f)>1e-6f)return 6;
  return 0;
}
