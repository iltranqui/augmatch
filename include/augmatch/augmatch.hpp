#pragma once
// Augmatch public umbrella header. Include this file to access the complete API.
// Each group below maps to one folder under include/augmatch/ and one page in docs/api/.

// core/ — image views, pixel types, Status, ExecutionContext, dtype conversion.
#include "augmatch/core/image.hpp"
#include "augmatch/core/pixel.hpp"
#include "augmatch/core/status.hpp"
#include "augmatch/core/execution.hpp"
#include "augmatch/core/signal.hpp"
#include "augmatch/core/convert.hpp"
#include "augmatch/core/easy.hpp"  // view-based convenience wrappers (augmatch::easy)

// annotations/ — bounding boxes, keypoints, polygons, and their geometric transforms.
#include "augmatch/annotations/target_metadata.hpp"
#include "augmatch/annotations/transforms.hpp"

// pipeline/ — config-driven Pipeline, stage registry, compose/one_of/some_of, meta ops.
#include "augmatch/pipeline/pipeline.hpp"
#include "augmatch/pipeline/pipeline_registry.hpp"
#include "augmatch/pipeline/composition.hpp"
#include "augmatch/pipeline/meta.hpp"

// geometry/ — flips, crops, resize, affine, focus breathing, dropout, multi-image mixing.
#include "augmatch/geometry/geometric.hpp"
#include "augmatch/geometry/size.hpp"
#include "augmatch/geometry/dropout.hpp"
#include "augmatch/geometry/mixing.hpp"

// color/ — photometric, HSV, colorspace, tone curves, arithmetic, dithering, PIL-like ops.
#include "augmatch/color/color.hpp"
#include "augmatch/color/colorspace.hpp"
#include "augmatch/color/hsv.hpp"
#include "augmatch/color/tone.hpp"
#include "augmatch/color/arithmetic.hpp"
#include "augmatch/color/dithering.hpp"
#include "augmatch/color/pillike.hpp"

// filter/ — convolution/blur, optical motion, aperture PSF, edges, superpixels, blending.
#include "augmatch/filter/filter.hpp"
#include "augmatch/filter/canny.hpp"
#include "augmatch/filter/derivative.hpp"
#include "augmatch/filter/superpixels.hpp"
#include "augmatch/filter/voronoi.hpp"
#include "augmatch/filter/blend.hpp"

// sensor/ — physical sensor noise, noise views, ISO profiles, Bayer/CFA, native sensor API.
#include "augmatch/sensor/noise.hpp"
#include "augmatch/sensor/noise_view.hpp"
#include "augmatch/sensor/iso_profile.hpp"
#include "augmatch/sensor/bayer.hpp"
#include "augmatch/sensor/native_api.hpp"
#include "augmatch/sensor/api.hpp"

// optics/ and isp/ — lens border helpers, demosaicing/quantization/clipping ISP artifacts.
#include "augmatch/optics/api.hpp"
#include "augmatch/isp/isp_artifacts.hpp"
#include "augmatch/isp/api.hpp"

// compression/ and video/ — JPEG, WebP, codec-independent transport corruption, video codecs.
#include "augmatch/compression/jpeg.hpp"
#include "augmatch/compression/webp.hpp"
#include "augmatch/compression/transport.hpp"
#include "augmatch/compression/api.hpp"
#include "augmatch/video/video_codec.hpp"
#include "augmatch/video/api.hpp"

// weather/ — rain, snow, fog, sun flare, shadows, environmental/acquisition effects.
#include "augmatch/weather/weather.hpp"
#include "augmatch/weather/environmental.hpp"

// catalog/ — imgcorruptlike compatibility and the remaining imgaug catalog ops.
#include "augmatch/catalog/imgcorruptlike.hpp"
#include "augmatch/catalog/remaining_catalog.hpp"
