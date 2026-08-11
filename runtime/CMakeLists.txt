# ---- SDL2 (host viewer, v0.1.1) -----------------------------------------
find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

add_executable(warpvm
  src/main.cu
  src/gpu/kernel.cu
  src/host/cpu_interpreter.cpp
  src/host/ptx_compiler.cpp
  src/host/life_bench.cu
  src/host/view_sdl.cu
)

target_include_directories(warpvm PRIVATE src ${SDL2_INCLUDE_DIRS})
target_compile_definitions(warpvm PRIVATE ${SDL2_CFLAGS_OTHER})
target_link_libraries(warpvm ${SDL2_LIBRARIES} CUDA::cuda_driver)

target_compile_options(warpvm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --Werror all-warnings>
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>
)

# ---- Rust language tools (assembler / disassembler) ----------------------
set(WVM_TOOLS_DIR ${CMAKE_SOURCE_DIR}/tools)
set(WVM_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/tools-rust)
set(WVM_AS ${WVM_CARGO_TARGET_DIR}/release/warpvm-as)
set(WVM_DIS ${WVM_CARGO_TARGET_DIR}/release/warpvm-dis)

add_custom_target(rust_tools ALL
  COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${WVM_CARGO_TARGET_DIR}
          cargo build --release --manifest-path ${WVM_TOOLS_DIR}/Cargo.toml
  BYPRODUCTS ${WVM_AS} ${WVM_DIS}
  WORKING_DIRECTORY ${WVM_TOOLS_DIR}
  COMMENT "Building Rust assembler/disassembler"
  VERBATIM
)

# ---- tests -----------------------------------------------------------------
add_test(NAME slice1_warp_arithmetic COMMAND warpvm slice1)
add_test(NAME slice2_interpreter COMMAND warpvm slice2)
add_test(NAME slice3_multi_vm COMMAND warpvm slice3)
add_test(NAME slice4_warp_control COMMAND warpvm slice4)
add_test(NAME slice5_control_plane COMMAND warpvm slice5)
add_test(NAME slice7_messaging COMMAND warpvm slice7)
add_test(NAME gfxa_framebuffer COMMAND warpvm gfxa)
add_test(NAME gfxb_flip COMMAND warpvm gfxb)
add_test(NAME cpu_interpreter COMMAND warpvm cpu_tests)
add_test(NAME v013_compiled_arithmetic_control_memory
  COMMAND warpvm compiled_tests)

add_test(NAME pipeline_hello
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/pipeline.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/hello.wva
          "r0  = 42" ${CMAKE_BINARY_DIR}/hello.wvm)
add_test(NAME pipeline_consts
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/pipeline.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/consts.wva
          "r0  = 123456" ${CMAKE_BINARY_DIR}/consts.wvm)
add_test(NAME pipeline_memory
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/pipeline.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/memory.wva
          "r2  = 100" ${CMAKE_BINARY_DIR}/memory.wvm)
add_test(NAME pipeline_loop32
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/pipeline.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/loop32.wva
          "r7  = 100" ${CMAKE_BINARY_DIR}/loop32.wvm)
add_test(NAME slice6_attach
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/attach.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/heartbeat.wva
          ${CMAKE_BINARY_DIR}/attach)
add_test(NAME v01_demo
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/demo.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/demo.wva
          ${CMAKE_BINARY_DIR}/demo)
add_test(NAME gfx_viewer_smoke
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gfx_viewer_smoke.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/graphics.wva
          ${CMAKE_BINARY_DIR}/gfxsmoke)
add_test(NAME gfx_capstone
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gfx_cap.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/graphics.wva
          ${CMAKE_BINARY_DIR}/gfxcap)
add_test(NAME program01_warplife
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warplife.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife.wva
          ${CMAKE_BINARY_DIR}/warplife)
add_test(NAME program01_benchmark_smoke
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warplife_bench.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife.wva
          ${CMAKE_BINARY_DIR}/warplife_bench)
add_test(NAME v012_cpu_gpu_equivalence
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warplife_equiv.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife.wva
          ${CMAKE_BINARY_DIR}/warplife_equiv)
add_test(NAME v013_compiled_warplife
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/compiled_life.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife.wva
          ${CMAKE_BINARY_DIR}/compiled_warplife)
add_test(NAME program01_native_cpu_equivalence
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warplife_native_cpu.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife.wva
          ${CMAKE_BINARY_DIR}/warplife_native_cpu)
