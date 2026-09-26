#include "augmatch/sensor/noise.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  constexpr int w = 3, h = 2, c = 1;
  const std::vector<float> input(w * h * c, 0.5f), rows(h, 0.1f), columns(w, -0.05f);
  std::vector<float> output(input.size());
  float *di = nullptr, *doo = nullptr, *dr = nullptr, *dc = nullptr;
  if (cudaMalloc(&di, input.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&doo, output.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&dr, rows.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&dc, columns.size() * sizeof(float)) != cudaSuccess) return 1;
  if (cudaMemcpy(di, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dr, rows.data(), rows.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dc, columns.data(), columns.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
  augmatch::row_column_correlated_noise_f32(di, doo, {w, h, c, 0.0f, 0.0f, dr, dc, 0.0f, 1.0f, 9});
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(output.data(), doo, output.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 3;
  for (float value : output) if (std::abs(value - 0.55f) > 1e-6f) return 4;
  augmatch::ClusteredDefect record{1, 0, 0, augmatch::ClusteredDefectKind::Dead, 0.0f, -1};
  augmatch::ClusteredDefect *device_record = nullptr;
  if (cudaMalloc(&device_record, sizeof(record)) != cudaSuccess || cudaMemcpy(device_record, &record, sizeof(record), cudaMemcpyHostToDevice) != cudaSuccess) return 5;
  augmatch::clustered_defective_pixels_f32(di, doo, {w, h, c, device_record, 1, 0, 0, 0.0f, 0.0f, 3});
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(output.data(), doo, output.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 6;
  if (output[1] != 0.0f || output[0] != 0.5f) return 7;
  cudaFree(device_record); cudaFree(dc); cudaFree(dr); cudaFree(doo); cudaFree(di);
  return 0;
}
