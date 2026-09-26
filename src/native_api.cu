// CUDA translation unit for the native API contract. The fused operation and
// batch operation reuse noise_view.cu's stateless counter-based kernel and
// preserve caller stream ordering. Host profile serialization remains usable
// in CUDA builds; device-owned profile points are handled by existing profile APIs.
#include "native_api_cpu.cpp"
