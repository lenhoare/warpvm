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

# ---- Rust language tools (assembler / disassembler / Warp C) -------------
set(WVM_TOOLS_DIR ${CMAKE_SOURCE_DIR}/tools)
set(WVM_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/tools-rust)
set(WVM_AS ${WVM_CARGO_TARGET_DIR}/release/warpvm-as)
set(WVM_DIS ${WVM_CARGO_TARGET_DIR}/release/warpvm-dis)
set(WVM_WARPC ${WVM_CARGO_TARGET_DIR}/release/warpc)

add_custom_target(rust_tools ALL
  COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${WVM_CARGO_TARGET_DIR}
          cargo build --release --manifest-path ${WVM_TOOLS_DIR}/Cargo.toml
  BYPRODUCTS ${WVM_AS} ${WVM_DIS} ${WVM_WARPC}
  WORKING_DIRECTORY ${WVM_TOOLS_DIR}
  COMMENT "Building Rust language tools"
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
add_test(NAME warpc_frontend_unit
  COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${WVM_CARGO_TARGET_DIR}
          cargo test --manifest-path ${WVM_TOOLS_DIR}/Cargo.toml -p warpc)
add_test(NAME warpc_integer_smoke
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_integer_smoke.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/integer_smoke.wc
          ${CMAKE_BINARY_DIR}/warpc_integer_smoke)
add_test(NAME warpc_control_flow
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_acceptance.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/control_flow.wc
          ${CMAKE_BINARY_DIR}/warpc_control_flow warpc_control_flow)
add_test(NAME warpc_switch
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_acceptance.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/switch.wc
          ${CMAKE_BINARY_DIR}/warpc_switch warpc_switch)
add_test(NAME warpc_functions
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_acceptance.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/functions.wc
          ${CMAKE_BINARY_DIR}/warpc_functions warpc_functions)
add_test(NAME warpc_memory
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_acceptance.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/memory.wc
          ${CMAKE_BINARY_DIR}/warpc_memory warpc_memory)
add_test(NAME warpc_strings
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_acceptance.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/strings.wc
          ${CMAKE_BINARY_DIR}/warpc_strings warpc_strings)
add_test(NAME warpc_uniformity
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_acceptance.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/uniformity.wc
          ${CMAKE_BINARY_DIR}/warpc_uniformity warpc_uniformity)
add_test(NAME warpc_divergent_if
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_acceptance.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/divergent_if.wc
          ${CMAKE_BINARY_DIR}/warpc_divergent_if warpc_divergent_if)
add_test(NAME warpc_graphics
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_graphics.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/hello_pixels.wc
          ${CMAKE_BINARY_DIR}/warpc_graphics)
add_test(NAME warpc_firefly
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warpc_firefly.sh
          ${WVM_WARPC} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warpc/warpc_firefly.wc
          ${CMAKE_BINARY_DIR}/warpc_firefly)

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
add_test(NAME program01_warplife_words
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warplife.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife_words.wva
          ${CMAKE_BINARY_DIR}/warplife_words)
add_test(NAME v012_words_cpu_gpu_equivalence
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warplife_equiv.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife_words.wva
          ${CMAKE_BINARY_DIR}/warplife_words_equiv)
add_test(NAME v013_compiled_warplife_words
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/compiled_life.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife_words.wva
          ${CMAKE_BINARY_DIR}/compiled_warplife_words)
add_test(NAME program01_words_native_cpu_equivalence
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/warplife_native_cpu.sh
          ${WVM_AS} $<TARGET_FILE:warpvm>
          ${CMAKE_SOURCE_DIR}/programs/warplife_words.wva
          ${CMAKE_BINARY_DIR}/warplife_words_native_cpu)
