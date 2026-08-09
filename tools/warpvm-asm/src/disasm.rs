// Disassembler: .wvm words -> re-assemblable .wva text.
//
// Output is deliberately round-trip stable: assembling the disassembly of a
// program reproduces the original code and literal pool byte-for-byte.

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
        let body = render(&d, word, literals);
        out.push_str(&format!("{body} ; {addr:04x}\n"));
    }
    out
}

fn render(d: &Decoded, word: u32, literals: &[u32]) -> String {
    let g = isa::guard_text(d.guard);
    let body = match d.op {
        isa::OP_NOP => "NOP".to_string(),
        isa::OP_HALT => "HALT".to_string(),
        isa::OP_MOV => format!("MOV r{}, r{}", d.rd, d.rs1),
        isa::OP_ADD => format!("ADD r{}, r{}, r{}", d.rd, d.rs1, d.rs2()),
        isa::OP_MOV_I => format!("MOV_I r{}, #{}", d.rd, d.imm()),
        isa::OP_ADD_I => format!("ADD_I r{}, r{}, #{}", d.rd, d.rs1, d.imm()),
        isa::OP_LOAD => format!("LOAD r{}, r{}", d.rd, d.rs1),
        isa::OP_STORE => format!("STORE r{}, r{}", d.rd, d.rs1),
        isa::OP_LDW => {
            let note = literals
                .get(d.lo as usize)
                .map(|v| format!(" ; lit={v}"))
                .unwrap_or_default();
            format!("LDW r{}, #{}{note}", d.rd, d.lo)
        }
        _ => return format!(".word 0x{word:08x}"),
    };
    if g.is_empty() {
        body
    } else {
        format!("{g} {body}")
    }
}
