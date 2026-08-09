// Disassembler: .wvm words -> re-assemblable .wva text.
//
// Output is deliberately round-trip stable: assembling the disassembly of a
// program reproduces the original code and literal pool byte-for-byte.
// Words whose fields are out of range (invalid programs) render as `.word`
// so they still round-trip exactly.

use crate::isa::{self, Decoded};

pub fn disassemble(code: &[u32], literals: &[u32]) -> String {
    let mut out = String::new();
    out.push_str("; warpvm disassembly\n");
    out.push_str(&format!(
        "; code: {} words, literals: {}\n",
        code.len(),
        literals.len()
    ));
    for (i, &v) in literals.iter().enumerate() {
        out.push_str(&format!(".const L{i} {v}\n"));
    }
    if !literals.is_empty() {
        out.push('\n');
    }

    for (addr, &word) in code.iter().enumerate() {
        let d = isa::decode(word);
        let body = match render(&d, literals) {
            Some(b) => b,
            None => format!(".word 0x{word:08x}"),
        };
        out.push_str(&format!("{body} ; {addr:04x}\n"));
    }
    out
}

fn vreg(n: u32) -> String {
    format!("r{n}")
}
fn sreg(n: u32) -> String {
    format!("s{n}")
}
fn pred(n: u32) -> String {
    format!("p{n}")
}
fn sreg_ok(n: u32) -> bool {
    n < 8
}
fn pred_ok(n: u32) -> bool {
    n < 4
}

// Binary vector op with an immediate form.
fn bin3(d: &Decoded, name: &str, name_i: &str, is_imm: bool) -> String {
    if is_imm {
        format!("{name_i} {}, {}, #{}", vreg(d.rd), vreg(d.rs1), d.imm())
    } else {
        format!("{name} {}, {}, {}", vreg(d.rd), vreg(d.rs1), vreg(d.rs2()))
    }
}

// Returns None when the word cannot be rendered as valid assembly.
fn render(d: &Decoded, literals: &[u32]) -> Option<String> {
    let g = isa::guard_text(d.guard);
    let body: String = match d.op {
        isa::OP_NOP => "NOP".into(),
        isa::OP_HALT => "HALT".into(),
        isa::OP_YIELD => "YIELD".into(),

        isa::OP_MOV => format!("MOV {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_MOV_I => format!("MOV_I {}, #{}", vreg(d.rd), d.imm()),

        isa::OP_ADD => bin3(d, "ADD", "ADD_I", false),
        isa::OP_SUB => bin3(d, "SUB", "SUB_I", false),
        isa::OP_MUL => bin3(d, "MUL", "MUL_I", false),
        isa::OP_AND => bin3(d, "AND", "AND_I", false),
        isa::OP_OR => bin3(d, "OR", "OR_I", false),
        isa::OP_XOR => bin3(d, "XOR", "XOR_I", false),
        isa::OP_SHL => bin3(d, "SHL", "SHL_I", false),
        isa::OP_SHR => bin3(d, "SHR", "SHR_I", false),
        isa::OP_ADD_I => bin3(d, "ADD", "ADD_I", true),
        isa::OP_SUB_I => bin3(d, "SUB", "SUB_I", true),
        isa::OP_MUL_I => bin3(d, "MUL", "MUL_I", true),
        isa::OP_AND_I => bin3(d, "AND", "AND_I", true),
        isa::OP_OR_I => bin3(d, "OR", "OR_I", true),
        isa::OP_XOR_I => bin3(d, "XOR", "XOR_I", true),
        isa::OP_SHL_I => bin3(d, "SHL", "SHL_I", true),
        isa::OP_SHR_I => bin3(d, "SHR", "SHR_I", true),
        isa::OP_DIV => bin3(d, "DIV", "", false),
        isa::OP_MOD => bin3(d, "MOD", "", false),
        isa::OP_MIN => bin3(d, "MIN", "", false),
        isa::OP_MAX => bin3(d, "MAX", "", false),

        isa::OP_ABS => format!("ABS {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_NEG => format!("NEG {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_NOT => format!("NOT {}, {}", vreg(d.rd), vreg(d.rs1)),

        isa::OP_CMP_EQ | isa::OP_CMP_NE | isa::OP_CMP_LT | isa::OP_CMP_LE
        | isa::OP_CMP_GT | isa::OP_CMP_GE => {
            if !pred_ok(d.rd) {
                return None;
            }
            let name = match d.op {
                isa::OP_CMP_EQ => "CMP_EQ",
                isa::OP_CMP_NE => "CMP_NE",
                isa::OP_CMP_LT => "CMP_LT",
                isa::OP_CMP_LE => "CMP_LE",
                isa::OP_CMP_GT => "CMP_GT",
                _ => "CMP_GE",
            };
            format!("{name} {}, {}, {}", pred(d.rd), vreg(d.rs1), vreg(d.rs2()))
        }
        isa::OP_CMP_EQ_I | isa::OP_CMP_NE_I | isa::OP_CMP_LT_I | isa::OP_CMP_LE_I
        | isa::OP_CMP_GT_I | isa::OP_CMP_GE_I => {
            if !pred_ok(d.rd) {
                return None;
            }
            let name = match d.op {
                isa::OP_CMP_EQ_I => "CMP_EQ",
                isa::OP_CMP_NE_I => "CMP_NE",
                isa::OP_CMP_LT_I => "CMP_LT",
                isa::OP_CMP_LE_I => "CMP_LE",
                isa::OP_CMP_GT_I => "CMP_GT",
                _ => "CMP_GE",
            };
            format!("{name} {}, {}, #{}", pred(d.rd), vreg(d.rs1), d.imm())
        }

        isa::OP_NOTMASK => {
            if !pred_ok(d.rd) || !pred_ok(d.rs1) {
                return None;
            }
            format!("NOTMASK {}, {}", pred(d.rd), pred(d.rs1))
        }
        isa::OP_ANDMASK | isa::OP_ORMASK => {
            if !pred_ok(d.rd) || !pred_ok(d.rs1) || !pred_ok(d.rs2()) {
                return None;
            }
            let name = if d.op == isa::OP_ANDMASK { "ANDMASK" } else { "ORMASK" };
            format!("{name} {}, {}, {}", pred(d.rd), pred(d.rs1), pred(d.rs2()))
        }
        isa::OP_BALLOT => {
            if !pred_ok(d.rd) {
                return None;
            }
            format!("BALLOT {}, {}", pred(d.rd), vreg(d.rs1))
        }
        isa::OP_ANY | isa::OP_ALL => {
            if !pred_ok(d.rd) || !pred_ok(d.rs1) {
                return None;
            }
            let name = if d.op == isa::OP_ANY { "ANY" } else { "ALL" };
            format!("{name} {}, {}", pred(d.rd), pred(d.rs1))
        }

        isa::OP_LANEID => format!("LANEID {}", vreg(d.rd)),
        isa::OP_VMID => format!("VMID {}", vreg(d.rd)),
        isa::OP_CLOCK => format!("CLOCK {}", vreg(d.rd)),
        isa::OP_RAND => format!("RAND {}", vreg(d.rd)),
        isa::OP_BROADCAST => format!("BROADCAST {}, {}, #{}", vreg(d.rd), vreg(d.rs1), d.lo),
        isa::OP_SHUFFLE => format!("SHUFFLE {}, {}, {}", vreg(d.rd), vreg(d.rs1), vreg(d.rs2())),
        isa::OP_SHUFFLE_XOR => format!("SHUFFLE_XOR {}, {}, #{}", vreg(d.rd), vreg(d.rs1), d.lo),

        isa::OP_REDUCE_ADD => format!("REDUCE_ADD {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_REDUCE_MIN => format!("REDUCE_MIN {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_REDUCE_MAX => format!("REDUCE_MAX {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_REDUCE_AND => format!("REDUCE_AND {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_REDUCE_OR => format!("REDUCE_OR {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_REDUCE_XOR => format!("REDUCE_XOR {}, {}", vreg(d.rd), vreg(d.rs1)),

        isa::OP_LOAD => format!("LOAD {}, {}", vreg(d.rd), vreg(d.rs1)),
        isa::OP_STORE => format!("STORE {}, {}", vreg(d.rd), vreg(d.rs1)),

        isa::OP_LDW => {
            let note = literals
                .get(d.lo as usize)
                .map(|v| format!(" ; lit={v}"))
                .unwrap_or_default();
            format!("LDW {}, #{}{note}", vreg(d.rd), d.lo)
        }

        isa::OP_LOG => format!("LOG {}, {}", vreg(d.rs1), vreg(d.rs2())),
        isa::OP_LOG_I => format!("LOG_I {}, #{}", vreg(d.rs1), d.imm()),

        isa::OP_SEND => format!("SEND {}, {}, {}", vreg(d.rd), vreg(d.rs1), vreg(d.rs2())),
        isa::OP_TRY_RECV => {
            if !pred_ok(d.rd) {
                return None;
            }
            format!("TRY_RECV {}, {}, {}", pred(d.rd), vreg(d.rs1), vreg(d.rs2()))
        }

        isa::OP_S_MOV => {
            if !sreg_ok(d.rd) || !sreg_ok(d.rs1) {
                return None;
            }
            format!("S_MOV {}, {}", sreg(d.rd), sreg(d.rs1))
        }
        isa::OP_S_MOV_I => {
            if !sreg_ok(d.rd) {
                return None;
            }
            format!("S_MOV_I {}, #{}", sreg(d.rd), d.imm())
        }
        isa::OP_S_ADD => {
            if !sreg_ok(d.rd) || !sreg_ok(d.rs1) || !sreg_ok(d.rs2()) {
                return None;
            }
            format!("S_ADD {}, {}, {}", sreg(d.rd), sreg(d.rs1), sreg(d.rs2()))
        }
        isa::OP_S_ADD_I => {
            if !sreg_ok(d.rd) || !sreg_ok(d.rs1) {
                return None;
            }
            format!("S_ADD_I {}, {}, #{}", sreg(d.rd), sreg(d.rs1), d.imm())
        }
        isa::OP_S_LDW => {
            if !sreg_ok(d.rd) {
                return None;
            }
            format!("S_LDW {}, #{}", sreg(d.rd), d.lo)
        }
        isa::OP_S_CMP_LT => {
            if !pred_ok(d.rd) || !sreg_ok(d.rs1) || !sreg_ok(d.rs2()) {
                return None;
            }
            format!("S_CMP_LT {}, {}, {}", pred(d.rd), sreg(d.rs1), sreg(d.rs2()))
        }
        isa::OP_S_CMP_LT_I => {
            if !pred_ok(d.rd) || !sreg_ok(d.rs1) {
                return None;
            }
            format!("S_CMP_LT_I {}, {}, #{}", pred(d.rd), sreg(d.rs1), d.imm())
        }
        isa::OP_S_BCAST => {
            if !sreg_ok(d.rs1) {
                return None;
            }
            format!("S_BCAST {}, {}", vreg(d.rd), sreg(d.rs1))
        }
        isa::OP_S_GET => {
            if !sreg_ok(d.rd) {
                return None;
            }
            format!("S_GET {}, {}", sreg(d.rd), vreg(d.rs1))
        }

        isa::OP_JMP => format!("JMP #{}", d.imm()),
        isa::OP_CALL => format!("CALL #{}", d.imm()),
        isa::OP_RET => "RET".into(),
        isa::OP_JMP_IF_ANY | isa::OP_JMP_IF_ALL => {
            // Predicate lives in the guard field; only the plain (non-inverted)
            // forms are representable in assembly.
            if !(1..=4).contains(&d.guard) {
                return None;
            }
            let name = if d.op == isa::OP_JMP_IF_ANY { "JMP_IF_ANY" } else { "JMP_IF_ALL" };
            format!("{name} {}, #{}", pred(d.guard - 1), d.imm())
        }

        _ => return None,
    };

    // JMP_IF_* carry their predicate in the guard field, already rendered as
    // an operand; prepending an @-guard would be wrong and non-reassemblable.
    let suppress_guard = matches!(d.op, isa::OP_JMP_IF_ANY | isa::OP_JMP_IF_ALL);
    if g.is_empty() || suppress_guard {
        Some(body)
    } else {
        Some(format!("{g} {body}"))
    }
}
