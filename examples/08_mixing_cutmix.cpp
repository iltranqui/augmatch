// 08_mixing_cutmix — paste a box from one image into another (CutMix), using the raw API.
//
// What it shows:
//   * Multi-image ops take ImageSourceU8 descriptors instead of views.
//   * The raw API runs where the build runs: host pointers in CPU builds, device pointers
//     in CUDA builds. This example handles both, so you can see what easy:: does for you.
//   * Box coordinates are half-open pixel edges: [x1, x2) x [y1, y2).
// API used: ImageSourceU8, CutMixConfig, cutmix_u8.
// Output: 08_mixing_cutmix.ppm — the test image with an inverted 24x16 patch in the middle.

#include <iostream>
#include <vector>

#include "augmatch/augmatch.hpp"
#include "example_utils.hpp"

int main() {
  const int width = 64, height = 48, channels = 3;
  const std::size_t bytes = static_cast<std::size_t>(width) * height * channels;

  std::vector<std::uint8_t> base = example::make_test_image(width, height);
  std::vector<std::uint8_t> patch(bytes), output(bytes);
  for (std::size_t i = 0; i < bytes; ++i) patch[i] = static_cast<std::uint8_t>(255 - base[i]);  // colour negative

  augmatch::CutMixConfig cutmix;
  cutmix.width = width;                  // the raw API needs the size in the config
  cutmix.height = height;
  cutmix.channels = channels;
  cutmix.box = {20.0f, 16.0f, 44.0f, 32.0f};  // x1, y1, x2, y2: pixels inside come from `patch`

#if AUGMATCH_HAS_CUDA
  // CUDA build: the raw function reads and writes device memory, so copy there and back.
  auto device_alloc = [](std::size_t size) {                  // cudaMalloc wants a void**
    void* pointer = nullptr;
    cudaMalloc(&pointer, size);
    return static_cast<std::uint8_t*>(pointer);
  };
  std::uint8_t* d_base = device_alloc(bytes);
  std::uint8_t* d_patch = device_alloc(bytes);
  std::uint8_t* d_out = device_alloc(bytes);
  cudaMemcpy(d_base, base.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_patch, patch.data(), bytes, cudaMemcpyHostToDevice);
  const augmatch::ImageSourceU8 first{d_base, width, height, channels};
  const augmatch::ImageSourceU8 second{d_patch, width, height, channels};
  augmatch::cutmix_u8(first, second, d_out, cutmix);            // enqueued on the default stream
  cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);  // waits for the kernel
  cudaFree(d_base); cudaFree(d_patch); cudaFree(d_out);
#else
  // CPU build: plain host pointers.
  const augmatch::ImageSourceU8 first{base.data(), width, height, channels};
  const augmatch::ImageSourceU8 second{patch.data(), width, height, channels};
  augmatch::cutmix_u8(first, second, output.data(), cutmix);
#endif

  // Inside the box we expect patch pixels, outside we expect base pixels.
  const std::size_t inside = (static_cast<std::size_t>(20) * width + 30) * channels;  // (x=30, y=20)
  const std::size_t outside = 0;                                                       // (x=0, y=0)
  if (output[inside] != patch[inside] || output[outside] != base[outside]) {
    std::cerr << "cutmix produced unexpected pixels\n";
    return 1;
  }

  example::write_ppm("08_mixing_cutmix.ppm", output, width, height);
  std::cout << "wrote 08_mixing_cutmix.ppm\n";
  return 0;
}
