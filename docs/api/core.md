# Core: views, conversion, tensors

Headers: `include/augmatch/core/`. Back to the [docs index](../README.md).

## Contents

- [ToTensorV2 native tensor-buffer contract](#totensorv2-native-tensor-buffer-contract)
- [ToTensor3D native tensor-buffer contract](#totensor3d-native-tensor-buffer-contract)

## ToTensorV2 native tensor-buffer contract

`to_tensor_v2_u8_f32` and `to_tensor_v2` convert contiguous HWC `uint8` pixels to a caller-owned contiguous CHW `float32` buffer. The output index is `output[c * height * width + y * width + x]`; without normalization it is `input[(y * width + x) * channels + c] / 255`. With `normalize=true`, the value is `(input / 255 - mean[min(c,2)]) / std[min(c,2)]`; channels after the third reuse the third parameter. `TensorViewF32` and `MutableTensorViewF32` explicitly describe CHW layout, float32 dtype, element strides, and host/device memory. The conversion does not allocate or copy ownership and is a native tensor-buffer operation, not a dependency on PyTorch.

CPU and CUDA use the same API and binary layout. The CUDA view API requires device views and the CPU view API requires host views; callers synchronize a supplied CUDA stream when completion is needed. The CLI path writes little-endian float32 CHW data:

```sh
augmatch_cli to_tensor_v2 IN.raw OUT.f32 WIDTH HEIGHT CHANNELS [NORMALIZE]
augmatch_cli to_tensor_v2 IN.raw OUT.f32 WIDTH HEIGHT CHANNELS 1 MEAN0 MEAN1 MEAN2 STD0 STD1 STD2
```

## ToTensor3D native tensor-buffer contract

`to_tensor_3d_u8_f32` converts contiguous DHWC `uint8` voxels to contiguous
CDHW `float32` data. The source index is `(((d * H + y) * W + x) * C + c)`;
the output index is `(((c * D + d) * H + y) * W + x)`. Values are divided by
255, then optionally normalized with the same three channel mean/std pairs as
ToTensorV2 (channels after the third reuse the third pair). CPU pointers refer
to host memory and CUDA pointers refer to device memory; the stream argument is
asynchronous on CUDA. The CLI writes little-endian float32 CDHW data:

```sh
augmatch_cli to_tensor_3d IN.raw OUT.f32 WIDTH HEIGHT DEPTH CHANNELS [NORMALIZE]
augmatch_cli to_tensor_3d IN.raw OUT.f32 WIDTH HEIGHT DEPTH CHANNELS 1 MEAN0 MEAN1 MEAN2 STD0 STD1 STD2
```
