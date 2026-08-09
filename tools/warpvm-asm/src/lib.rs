// WarpVM language tools: assembler, disassembler, .wvm file format.
// The instruction contract is docs/isa.md; isa.rs mirrors the constants in
// runtime/src/gpu/warpvm.cuh. Round-trip tests police drift between them.
pub mod assemble;
pub mod disasm;
pub mod isa;
pub mod parse;
pub mod wvm;

pub use assemble::{assemble, Assembled};
