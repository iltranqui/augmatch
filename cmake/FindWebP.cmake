# Minimal package discovery for the native libwebp dependency.
find_path(WebP_INCLUDE_DIR NAMES webp/encode.h)
find_library(WebP_LIBRARY NAMES webp)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(WebP DEFAULT_MSG WebP_INCLUDE_DIR WebP_LIBRARY)

if(WebP_FOUND)
  set(WebP_INCLUDE_DIRS "${WebP_INCLUDE_DIR}")
  set(WebP_LIBRARIES "${WebP_LIBRARY}")
endif()
mark_as_advanced(WebP_INCLUDE_DIR WebP_LIBRARY)
