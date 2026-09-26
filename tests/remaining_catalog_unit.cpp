#include "augmatch/catalog/remaining_catalog.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
int main(){
  const int w=4,h=3,c=3; std::vector<std::uint8_t> in(w*h*c),out(in.size()),again(in.size());
  for(std::size_t i=0;i<in.size();++i) in[i]=static_cast<std::uint8_t>(i*17);
  augmatch::AdditiveLaplaceNoiseConfig ln{w,h,c,0,9}; augmatch::additive_laplace_noise_u8(in.data(),out.data(),ln); assert(out==in);
  augmatch::AdditivePoissonNoiseConfig pn{w,h,c,1e9f,3}; augmatch::additive_poisson_noise_u8(in.data(),out.data(),pn); assert(out.size()==in.size());
  augmatch::CartoonConfig cartoon{w,h,c,1,1000,0}; augmatch::cartoon_u8(in.data(),out.data(),cartoon); assert(out.size()==in.size());
  augmatch::RandAugmentConfig ra{w,h,c,2,0,4}; augmatch::rand_augment_u8(in.data(),out.data(),ra); augmatch::rand_augment_u8(in.data(),again.data(),ra); assert(out==again);
  augmatch::ChangeColorspaceConfig cs{w,h,c,augmatch::CatalogColorSpace::RGB,augmatch::CatalogColorSpace::HSV}; augmatch::change_colorspace_u8(in.data(),out.data(),cs); assert(out[2]<=255);
  augmatch::KMeansColorQuantizationConfig km{w,h,c,2,3,5}; augmatch::kmeans_color_quantization_u8(in.data(),out.data(),km); assert(out.size()==in.size());
  const float kernel[1]={1}; augmatch::convolve_u8(in.data(),out.data(),{w,h,c,1,1,kernel,1,0}); assert(out==in);
  augmatch::WithPolarWarpingConfig pw{w,h,c}; augmatch::with_polar_warping_u8(in.data(),out.data(),pw); assert(out.size()==in.size());
  augmatch::JigsawConfig j{w,h,c,2,2,nullptr,7}; augmatch::jigsaw_u8(in.data(),out.data(),j); augmatch::jigsaw_u8(in.data(),again.data(),j); assert(out==again);
  std::vector<std::uint8_t> ref(in.size(),200),fda(in.size()); augmatch::fda_u8(in.data(),fda.data(),{w,h,c,ref.data(),w,h,.5f}); assert(fda.size()==in.size());
  augmatch::SaveDebugImageEveryNBatchesConfig dbg{w,h,c,1,2,"remaining_catalog_test.ppm"}; augmatch::save_debug_image_every_n_batches_u8(in.data(),out.data(),dbg); assert(out==in); std::remove("remaining_catalog_test.ppm");
  return 0;
}
