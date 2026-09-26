#include "augmatch/isp/isp_artifacts.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
int main(){const int w=16,h=12,c=3;std::vector<std::uint8_t> in(w*h*c),a(in.size()),b(in.size());for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int k=0;k<c;++k)in[(y*w+x)*c+k]=(x<8?20:235)+k;
 augmatch::GradientPlusLaplacianNoiseConfig gn{w,h,c,2,1,1,77};augmatch::gradient_plus_laplacian_noise_u8(in.data(),a.data(),gn);augmatch::gradient_plus_laplacian_noise_u8(in.data(),b.data(),gn);assert(a==b);
 augmatch::DeblockingHalosConfig db{w,h,c,8,1,.5f};augmatch::deblocking_halos_u8(in.data(),a.data(),db);
 augmatch::DemosaicingEdgeArtifactsConfig dm{w,h,c,.5f,1};augmatch::demosaicing_edge_artifacts_u8(in.data(),a.data(),dm);
 augmatch::EdgeDependentQuantizationConfig eq{w,h,c,8,1};augmatch::edge_dependent_quantization_u8(in.data(),a.data(),eq);
 augmatch::EdgeDependentCompressionErrorConfig ce{w,h,c,8,1,1,7};augmatch::edge_dependent_compression_error_u8(in.data(),a.data(),ce);
 augmatch::GradientReversalConfig gr{w,h,c,.5f,1};augmatch::gradient_reversal_u8(in.data(),a.data(),gr);
 augmatch::ClippedEdgeRingingConfig cr{w,h,c,.5f,1,16};augmatch::clipped_edge_ringing_u8(in.data(),a.data(),cr);for(auto v:a)assert(v<=255);return 0;}
