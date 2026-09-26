# Weather and environmental effects

Headers: `include/augmatch/weather/`. Back to the [docs index](../README.md).

## Contents

- [Environmental and acquisition contract](#environmental-and-acquisition-contract)
- [PlasmaShadow contract](#plasmashadow-contract)
- [RandomRain contract](#randomrain-contract)
- [RandomSnow contract](#randomsnow-contract)
- [SnowStamp contract](#snowstamp-contract)
- [RandomGravel contract](#randomgravel-contract)
- [Spatter contract](#spatter-contract)
- [RandomSunFlare contract](#randomsunflare-contract)
- [RandomShadow contract](#randomshadow-contract)
- [RandomFog contract](#randomfog-contract)
- [Catalog weather contracts](#catalog-weather-contracts)

## Environmental and acquisition contract

`augmatch/weather/environmental.hpp` adds deterministic float32 HWC veil, gain, dirty-lens, and electromagnetic-interference APIs, plus T-H-W-C sensor-temperature drift, power-supply banding, fluorescent flicker, and LED rolling-band APIs. HWC maps are borrowed HxW or HxWxC arrays in the execution memory space; temporal batches are contiguous T-H-W-C. CPU and CUDA maps follow the existing host/device ownership contract, and all outputs clip to the configured range. Dirty-lens is an opacity-weighted square-box blur; EMI is a sinusoidal additive field; temperature drift is exponential dark-current scaling across an explicit temperature ramp; the lighting artifacts are sinusoidal acquisition approximations rather than physical lens, RF, ballast, PWM, or exposure simulations. CLI forms and parameters are documented in `NOISE_CATALOG.md`. Fog, rain, snow, and dust use the explicit seeded weather records in `weather.hpp`; headlight flare and reflections reuse the explicit optical source/ghost records.

## PlasmaShadow contract

`plasma_shadow_u8` applies deterministic spatial shadows to contiguous HWC `uint8`
data. For a field value below `threshold`, the normalized shadow amount is
`(threshold - field) / threshold`; values at or above the threshold are unchanged.
Each channel is multiplied by `1 - strength * shadow`, rounded with
`floor(value + 0.5)`, and clipped to `[0,255]`. `threshold` and `strength` are
finite values in `[0,1]`; a zero threshold disables shadows. The explicit field
contains exactly `width*height` finite floats in `[0,1]` in row-major order, while
a null field uses the shared four-scale hash generated from `seed` on CPU and
CUDA. The CLI form is `color plasma_shadow IN.raw OUT.raw W H C THRESHOLD
STRENGTH SEED [FIELD.f32]`.

## RandomRain contract

`random_rain_u8` accepts contiguous HWC `uint8` data and overlays white rain on
pixel-centre samples. A `RainStreak` is an explicit `(x0,y0,x1,y1,width,alpha)`
segment in image pixels. Supplying `RandomRainConfig::streaks` makes placement
fully explicit; a null list generates the requested count from
`SplitMix64(seed + 6*index + offset)` and the configured length, angle, and width
ranges. `make_random_rain_streaks` materializes that generated list for logging
or exact CPU/CUDA replay, so stochastic-looking placement is never hidden.
For each covered pixel and each channel, segments are composited in list order as
`round((1-a)*current + a*255)`, where `a=clamp(config.alpha*streak.alpha,0,1)`;
overlaps are sequential and alpha zero is identity. CPU streak lists are host
memory and CUDA streak lists are device memory. The CLI form is
`weather random_rain IN.raw OUT.raw W H C COUNT ALPHA SEED [LENGTH_MIN LENGTH_MAX ANGLE_MIN ANGLE_MAX WIDTH [STREAKS.bin]]`;
the direct `random_rain` spelling is also accepted. A binary streak list contains
`COUNT` little-endian records of six float32 values.

## RandomSnow contract

`random_snow_u8` accepts contiguous HWC `uint8` data and overlays white circular
snowflakes on pixel-centre samples. A `Snowflake` is an explicit
`(x,y,radius,alpha)` record in image pixels. Supplying
`RandomSnowConfig::snowflakes` makes placement fully explicit; a null list
generates the requested count from `SplitMix64(seed + 4*index + offset)` with
uniform position and radius in the configured range. `make_random_snowflakes`
materializes generated records for logging or exact CPU/CUDA replay. For each
covered pixel and channel, records are composited in list order as
`round((1-a)*current + a*255)`, where `a=clamp(config.alpha*snowflake.alpha,0,1)`;
overlaps are sequential, alpha zero is identity, and all channels receive the
same blend. CPU records are host memory and CUDA records are device memory. The
CLI form is `weather random_snow IN.raw OUT.raw W H C COUNT ALPHA SEED [RADIUS_MIN RADIUS_MAX [SNOWFLAKES.bin]]`; the direct `random_snow` spelling is also accepted. A binary snowflake list contains `COUNT` little-endian records of four float32 values.

## SnowStamp contract

`snow_stamp_u8` overlays one explicit stamp image and mask at an ordered list of
integer top-left placements. The stamp image is row-major HWC `uint8` with one
channel (broadcast) or exactly the output channel count; a null image uses the
scalar `fill`. The optional row-major one-channel mask contains opacity bytes
(0..255), with a null mask treated as fully opaque. Each placement is a binary
12-byte little-endian record `(int32 x, int32 y, float32 alpha)`. For each
covered stamp pixel and output channel, in placement order, the result is
`round((1-a)*current+a*fill)`, where
`a=clamp(global_alpha*placement.alpha*mask/255,0,1)`; image values supply
`fill` when present. Stamps are clipped at image boundaries, overlaps are
sequential, and there is no random state or interpolation. CPU arrays are host
memory and CUDA arrays are device memory. The CLI form is
`weather snow_stamp IN.raw OUT.raw W H C STAMP_W STAMP_H COUNT ALPHA FILL IMAGE.raw MASK.raw PLACEMENTS.bin`;
use `-` for IMAGE or MASK to select the null behavior. The direct `snow_stamp`
spelling is also accepted.

## RandomGravel contract

`random_gravel_u8` accepts contiguous HWC `uint8` data and overlays filled gravel
particles at pixel-centre samples. A `GravelParticle` is an explicit
`(x,y,radius,alpha,fill)` record; `fill` is one scalar uint8 value applied to every
channel. Supplying `RandomGravelConfig::particles` makes placement, radius, alpha,
and fill fully explicit. A null list generates the requested count from
`SplitMix64(seed + 4*index + offset)` with uniform position and configured radius
range, using `config.fill` and local alpha one. `make_random_gravel_particles`
materializes generated records for logging or exact CPU/CUDA replay. For each
covered pixel and channel, records are composited in list order as
`round((1-a)*current + a*fill)`, where `a=clamp(config.alpha*particle.alpha,0,1)`;
overlaps are sequential, values are clipped to `[0,255]`, and zero alpha or an
empty list is identity. CPU particle lists are host memory and CUDA lists are
device memory. The CLI form is
`weather random_gravel IN.raw OUT.raw W H C COUNT ALPHA SEED [RADIUS_MIN RADIUS_MAX FILL [PARTICLES.bin]]`;
the direct spelling is also accepted. A binary particle list contains `COUNT`
records of four little-endian float32 values followed by uint8 fill and three
padding bytes.

## Spatter contract

`random_spatter_u8` (also exposed as `spatter_u8`) accepts contiguous HWC `uint8`
data and overlays colored filled discs at pixel-centre samples. A
`SpatterDroplet` is an explicit `(x,y,radius,red,green,blue,alpha)` record;
its RGB color is blended into the first three channels and channels beyond RGB
are copied unchanged. Supplying `RandomSpatterConfig::droplets` makes
placement, radius, color, and local alpha fully explicit. A null list generates
the requested count from `SplitMix64(seed + 4*index + offset)` with uniform
position and configured radius range, using the configured RGB color and local
alpha one. `make_random_spatter_droplets` materializes generated records for
logging or exact CPU/CUDA replay. For covered RGB channels, records are
composited in list order as `round((1-a)*current+a*color)`, where
`a=clamp(config.alpha*droplet.alpha,0,1)`; results are clipped to `[0,255]` and
alpha zero or an empty list is identity. CPU records are host memory and CUDA
records are device memory. The CLI form is
`weather spatter IN.raw OUT.raw W H C COUNT ALPHA SEED [RADIUS_MIN RADIUS_MAX RED GREEN BLUE [DROPLETS.bin]]`;
`random_spatter` and the direct `spatter` spelling are also accepted. A binary
droplet list contains `COUNT` little-endian records of three float32 values,
three color bytes, one padding byte, and one float32 alpha (20 bytes).

## RandomSunFlare contract

`random_sun_flare_u8` accepts contiguous HWC `uint8` data and overlays a white
source disc followed by radial ray capsules. `RandomSunFlareConfig::source` is
an explicit `(x,y,radius,alpha)` source in image pixels. A non-null `rays` list
contains explicit `(angle,length,width,alpha)` records, with angle in degrees
from +x; a null list generates `ray_count` records from
`SplitMix64(seed + 4*index + offset)` and the configured ranges. The generated
records can be materialized with `make_random_sun_flare_rays`. Pixel centres
use hard disc/capsule tests. The source is blended first, then rays in list
order, using `round((1-a)*current + a*255)` with
`a=clamp(opacity*local_alpha,0,1)` for every channel. CPU ray lists are host
memory and CUDA ray lists are device memory. The CLI form is
`weather random_sun_flare IN.raw OUT.raw W H C COUNT OPACITY SEED [SOURCE_X SOURCE_Y SOURCE_RADIUS SOURCE_ALPHA [LENGTH_MIN LENGTH_MAX ANGLE_MIN ANGLE_MAX WIDTH_MIN WIDTH_MAX [RAYS.bin]]]`; the direct spelling is also accepted. A binary ray list contains `COUNT` little-endian records of four float32 values.

## RandomShadow contract

`random_shadow_u8` applies deterministic explicit shadow masks to contiguous HWC
`uint8` data. A `ShadowPolygon` contains a borrowed point array and local alpha;
a `ShadowRectangle` contains two image-pixel corners and local alpha. Pixel
centres `(x+0.5,y+0.5)` are tested with an even-odd polygon rule (boundary
included) and inclusive normalized rectangle bounds. Polygon records are
composited first, followed by rectangles in list order. For every covered
channel, the result is `round((1-a)*current + a*fill)`, where
`a=clamp(opacity*record.alpha,0,1)` and the scalar `fill` is applied to every
channel. Values are clipped to `[0,255]`; empty lists and zero opacity are
identity. The seed is retained for API symmetry but does not affect explicit
masks. CPU mask and point arrays are host memory; CUDA arrays and point arrays
are device memory. The CLI form is
`weather random_shadow IN.raw OUT.raw W H C OPACITY FILL POLYGONS.bin RECTANGLES.bin`;
the direct spelling is also accepted. Use `-` for either empty file. A polygon
file contains repeated little-endian records `<uint32 point_count, float alpha>`
followed by `point_count` `<float x,float y>` pairs. A rectangle file contains
repeated little-endian `<float x0,y0,x1,y1,alpha>` records.

## RandomFog contract

`random_fog_u8` accepts contiguous HWC `uint8` data and applies one white,
depth-independent veil per pixel. A non-null `RandomFogConfig::field` contains
exactly `width*height` row-major float32 values in `[0,1]`; each value is the
local fog density. A null field uses `SplitMix64(seed + pixel)` on both CPU and
CUDA, and `make_random_fog_field` materializes that deterministic field. For
every channel, the result is
`round((1-a)*input + a*255)`, with
`a=clamp(config.opacity*config.density*field[pixel],0,1)`, followed by clipping
to `[0,255]`. Density and opacity are finite values in `[0,1]`; zero density,
zero opacity, and an all-zero field are identity. The same veil is applied to
all channels and no depth input is accepted. CPU fields are host memory and
CUDA fields are device memory. The CLI form is
`weather random_fog IN.raw OUT.raw W H C DENSITY OPACITY SEED [FIELD.f32]`; the
direct `random_fog` spelling is also accepted. A field file contains
`width*height` little-endian float32 values.

## Catalog weather contracts

The eight `augmenters.weather` entries are available through
`augmatch/weather/weather.hpp` and the umbrella header. `FastSnowyLandscape` measures
mean RGB brightness and blends bright pixels toward white above `snow_point`;
this is a deterministic brightness-threshold approximation, not a semantic
landscape model. `Clouds` uses a deterministic four-counter SplitMix64 field
(or a caller-supplied HxW field) and white-composites it. `Fog` is the existing
seeded depth-independent white veil under the catalog spelling.

`Snowflakes` and `Rain` expose the explicit `Snowflake` and `RainStreak`
records and their seeded materializers under catalog aliases. Records are
processed in list order, overlaps are sequential, and zero alpha or an empty
list is identity. `CloudLayer`, `SnowflakesLayer`, and `RainLayer` each accept
a borrowed row-major HxW float32 map in `[0,1]`; the map is never retained or
resized and CUDA maps must be device allocations. Their target is a borrowed
contiguous HWC uint8 image, and every channel receives
`round((1-a)*input+a*255)` with `a=clamp(opacity*map,0,1)`.

CLI examples are `weather fast_snowy_landscape IN.raw OUT.raw W H C
SNOW_POINT ALPHA`, `weather clouds IN.raw OUT.raw W H C ALPHA DENSITY SEED`,
`weather fog IN.raw OUT.raw W H C DENSITY OPACITY SEED`, and
`weather {cloud_layer|snowflakes_layer|rain_layer} IN.raw OUT.raw W H C
OPACITY LAYER.f32`. `weather snowflakes` and `weather rain` use the same
record formats and seeded arguments as `random_snow` and `random_rain`.
These approximations intentionally operate only on uint8 HWC images.
