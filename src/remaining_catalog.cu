// The remaining catalog APIs have an explicit host-only contract (see header).
// Keeping the implementation in one translation unit avoids an unsafe implicit
// device-pointer fallback while still exporting the same API from CUDA builds.
#include "remaining_catalog_cpu.cpp"
