add_executable(warpvm
  src/main.cu
)

target_include_directories(warpvm PRIVATE src)

target_compile_options(warpvm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --Werror all-warnings>
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>
)

add_test(NAME slice1_warp_arithmetic COMMAND warpvm slice1)
