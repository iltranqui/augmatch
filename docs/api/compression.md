# Compression and transport

Headers: `include/augmatch/compression/`. Back to the [docs index](../README.md).

## Contents

- [JPEG ImageCompression/JpegCompression contract](#jpeg-imagecompressionjpegcompression-contract)
- [Codec blockers and WebP](#codec-blockers-and-webp)
- [Codec-independent transport corruption](#codec-independent-transport-corruption)
- [WebP WebPCompression contract](#webp-webpcompression-contract)

## JPEG ImageCompression/JpegCompression contract

`jpeg_compression_u8` and its `image_compression_u8` alias perform a real JPEG encode/decode round trip through libjpeg-turbo (or a compatible libjpeg implementation discovered by CMake). The input is contiguous HWC `uint8` with one grayscale or three RGB channels. `JpegCompressionConfig` makes quality `[1,100]` and chroma sampling explicit: `Y444`, `Y422`, or `Y420`; the output has the same dimensions and channel count. `jpeg_encode_u8` exposes the host-owned encoded byte stream, and `jpeg_decode_u8` decodes it into a caller-owned buffer. JPEG is lossy, so exact identity is not expected.

The CPU implementation is native libjpeg-turbo. The CUDA `.cu` implementation does **not** claim GPU-native JPEG: it synchronously copies device pixels to host memory, invokes the host codec, and copies decoded pixels back to the device. This fallback is intentional because the project has no CUDA JPEG codec dependency. If libjpeg-turbo/libjpeg is not found, CMake disables these APIs and reports the dependency warning. The CLI form is `jpeg_compression IN.raw OUT.raw W H C QUALITY SUBSAMPLING`, where subsampling is `444`, `422`, or `420`; `image_compression` is an alias.

`jpeg_quality_variation_u8` exposes the same codec with an explicit quality value, and `jpeg_chroma_subsampling_u8` exposes the three libjpeg sampling modes directly. `jpeg_quantization_table_variation_u8` accepts independent positive luminance and chrominance table multipliers (one is neutral); its CLI is `jpeg_quantization_table_variation IN.raw OUT.raw W H C QUALITY LUMA_SCALE CHROMA_SCALE SUBSAMPLING`. JPEG ringing, blocking, and mosquito-noise entries first perform the same libjpeg round trip and then apply deterministic, clipped post-decode artifact models. Their shared CLI form is `jpeg_{ringing|blocking|mosquito_noise} IN.raw OUT.raw W H C QUALITY SUBSAMPLING STRENGTH`.

`jpeg_restart_marker_damage_u8` emits a configured restart interval and deterministically corrupts entropy bytes at selected restart boundaries before decoding. `jpeg_progressive_decoding_u8` uses a progressive JPEG stream and exposes a deterministic early-scan preview model. Their CLI forms are `jpeg_restart_marker_damage IN.raw OUT.raw W H C QUALITY SUBSAMPLING STRENGTH RESTART_INTERVAL` and `jpeg_progressive_decoding IN.raw OUT.raw W H C QUALITY SUBSAMPLING STRENGTH SCAN`. All CPU pointers are host-owned. CUDA pointers are device-owned, but every JPEG entry uses the documented synchronous host-codec fallback and honors the supplied stream for transfers.

## Codec blockers and WebP

`webp_lossy_compression_u8` and `webp_lossless_compression_u8` are explicit libwebp-backed catalog entry points; their aliases are `webp_lossy_u8` and `webp_lossless_u8`. The CLI forms are `webp_lossy_compression IN.raw OUT.raw W H C QUALITY` and `webp_lossless_compression IN.raw OUT.raw W H C QUALITY`. CMake conditionally enables libwebp when both public headers and the native library are found. Without it, the public symbols remain available but fail clearly with a dependency error rather than silently substituting another codec. CUDA uses the documented synchronous host-codec fallback.

`video_codec.hpp` provides conditional native FFmpeg single-frame encode/decode for AV1, H.264, and HEVC, plus an AV1 intra encode/decode artifact entry point. Its CLI round-trip is `augmatch_cli video_codec IN.raw OUT.raw W H C h264|hevc|av1 QUALITY`; input/output are interleaved uint8 grayscale or RGB, dimensions must be positive and even, and quality is in `[1,100]`. CMake enables the API only when libavcodec, libavutil, and libswscale development files are found; otherwise calls fail clearly as unavailable. This is not an arbitrary bitstream parser or mutator and does not implement temporal prediction, reference-frame semantics, container support, or external-codec parity, so all four catalog entries remain partial. CUDA buffers use a documented synchronous host-codec fallback rather than GPU-native encoding. `video_codec_unit` attempts actual encoder/decoder round trips for each codec and reports unavailable individual encoders without turning transport surrogates into codec behavior.

## Codec-independent transport corruption

`transport.hpp` adds raw transport operations without an external video codec. `truncated_frame_u8` and `bit_flips_u8` operate on explicit byte counts; `block_quantization_u8`, `deblocking_filter_mismatch_u8`, and `packet_loss_macroblocks_u8` operate on interleaved HWC frames; `chroma_plane_misalignment_u8` operates on one row-major plane (invoke it separately for U and V); and `raw_bayer_transport_corruption_u16` operates on unpacked Bayer samples with explicit bit depth. CPU pointers are host-owned and CUDA pointers are device-owned; optional masks are borrowed in the execution memory space and stream-safe. Every operation has a CUDA kernel where meaningful. Example CLI forms are `truncated_frame IN OUT BYTES PRESERVE FILL`, `bit_flips IN OUT BYTES PROBABILITY MASK SEED`, `block_quantization IN OUT W H C BLOCK QUANT_STEP`, and `chroma_plane_misalignment IN OUT W H STRIDE DX DY FILL`. AV1, H.264, and H.265 bitstream parsing, reference-picture reconstruction, and codec-specific damage remain unsupported blockers.

## WebP WebPCompression contract

`webp_compression_u8` performs a real native libwebp encode/decode round trip for contiguous HWC `uint8` images with one grayscale or three RGB channels. `WebPCompressionConfig` makes quality `[0,100]` and the `lossless` mode explicit. Lossy mode uses libwebp's quality scale; lossless mode selects libwebp lossless encoding. The output preserves dimensions and channel count, while lossy output is not expected to be identical. `webp_encode_u8` exposes the host-owned WebP byte stream and `webp_decode_u8` decodes it into a caller-owned buffer.

The CPU implementation is native libwebp. The CUDA `.cu` implementation does **not** claim GPU-native WebP: it synchronously copies device pixels to host memory, invokes libwebp, and copies decoded pixels back to the device. CMake enables these APIs and their unit/parity tests only when both libwebp headers and the native library are found; otherwise it reports a warning and omits the codec source. The CLI form is `webp_compression IN.raw OUT.raw W H C QUALITY LOSSLESS`, where `LOSSLESS` is `0` or `1`; `webp` is an alias. This entry covers WebP image compression only and makes no AV1 or video-codec claim.
