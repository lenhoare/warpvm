add_executable(warpvm
  src/main.cu
  src/gpu/kernel.cu
)

target_include_directories(warpvm PRIVATE src)

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
