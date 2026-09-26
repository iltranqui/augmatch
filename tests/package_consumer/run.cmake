if(NOT DEFINED AUGMATCH_BUILD_DIR OR NOT DEFINED AUGMATCH_SOURCE_DIR)
  message(FATAL_ERROR "AUGMATCH_BUILD_DIR and AUGMATCH_SOURCE_DIR are required")
endif()

set(prefix "${AUGMATCH_BUILD_DIR}/package-consumer-prefix")
set(consumer_build "${AUGMATCH_BUILD_DIR}/package-consumer-build")
file(REMOVE_RECURSE "${prefix}" "${consumer_build}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${AUGMATCH_BUILD_DIR}" --prefix "${prefix}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(result)
  message(FATAL_ERROR "package install failed (${result})\n${output}\n${error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${AUGMATCH_SOURCE_DIR}/tests/package_consumer"
          -B "${consumer_build}" -DCMAKE_PREFIX_PATH=${prefix}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(result)
  message(FATAL_ERROR "find_package consumer configure failed (${result})\n${output}\n${error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(result)
  message(FATAL_ERROR "find_package consumer build failed (${result})\n${output}\n${error}")
endif()

execute_process(
  COMMAND "${consumer_build}/augmatch_package_consumer"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(result AND NOT result EQUAL 77)
  message(FATAL_ERROR "find_package consumer execution failed (${result})\n${output}\n${error}")
endif()
