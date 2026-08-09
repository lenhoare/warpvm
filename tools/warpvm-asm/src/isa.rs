// Instruction encoding constants — mirror of runtime/src/gpu/warpvm.cuh.
// The contract is docs/isa.md; keep both in sync.

pub const LANES: usize = 32;
pub const MAX_CODE_WORDS: usize = 4096;
pub const MAX_LITERALS: usize = 256;

pub const OPCODE_SHIFT: u32 = 25;
pub const GUARD_SHIFT: u32 = 21;
pub const RD_SHIFT: u32 = 17;
pub const RS1_SHIFT: u32 = 13;
pub const REG_FIELD_MASK: u32 = 0xF;
pub const LO_MASK: u32 = 0x1FFF;

pub const IMM_MIN: i64 = -4096;
pub const IMM_MAX: i64 = 4095;

pub const OP_NOP: u32 = 0x00;

pub const OP_MOV: u32 = 0x01;
pub const OP_ADD: u32 = 0x02;
pub const OP_SUB: u32 = 0x03;
pub const OP_MUL: u32 = 0x04;
pub const OP_DIV: u32 = 0x05;
pub const OP_MOD: u32 = 0x06;
pub const OP_MIN: u32 = 0x07;
pub const OP_MAX: u32 = 0x08;
pub const OP_AND: u32 = 0x09;
pub const OP_OR: u32 = 0x0A;
pub const OP_XOR: u32 = 0x0B;
pub const OP_SHL: u32 = 0x0C;
pub const OP_SHR: u32 = 0x0D;

pub const OP_MOV_I: u32 = 0x10;
pub const OP_ADD_I: u32 = 0x11;
pub const OP_SUB_I: u32 = 0x12;
pub const OP_MUL_I: u32 = 0x13;
pub const OP_AND_I: u32 = 0x14;
pub const OP_OR_I: u32 = 0x15;
pub const OP_XOR_I: u32 = 0x16;
pub const OP_SHL_I: u32 = 0x17;
pub const OP_SHR_I: u32 = 0x18;
pub const OP_LDW: u32 = 0x19;
pub const OP_S_BCAST: u32 = 0x1A;
pub const OP_S_GET: u32 = 0x1B;

pub const OP_ABS: u32 = 0x20;
pub const OP_NEG: u32 = 0x21;
pub const OP_NOT: u32 = 0x22;

pub const OP_CMP_EQ: u32 = 0x28;
pub const OP_CMP_NE: u32 = 0x29;
pub const OP_CMP_LT: u32 = 0x2A;
pub const OP_CMP_LE: u32 = 0x2B;
pub const OP_CMP_GT: u32 = 0x2C;
pub const OP_CMP_GE: u32 = 0x2D;

pub const OP_CMP_EQ_I: u32 = 0x30;
pub const OP_CMP_NE_I: u32 = 0x31;
pub const OP_CMP_LT_I: u32 = 0x32;
pub const OP_CMP_LE_I: u32 = 0x33;
pub const OP_CMP_GT_I: u32 = 0x34;
pub const OP_CMP_GE_I: u32 = 0x35;

pub const OP_NOTMASK: u32 = 0x38;
pub const OP_ANDMASK: u32 = 0x39;
pub const OP_ORMASK: u32 = 0x3A;
pub const OP_BALLOT: u32 = 0x3B;
pub const OP_ANY: u32 = 0x3C;
pub const OP_ALL: u32 = 0x3D;

pub const OP_LANEID: u32 = 0x40;
pub const OP_BROADCAST: u32 = 0x41;
pub const OP_SHUFFLE: u32 = 0x42;
pub const OP_SHUFFLE_XOR: u32 = 0x43;
pub const OP_REDUCE_ADD: u32 = 0x44;
pub const OP_REDUCE_MIN: u32 = 0x45;
pub const OP_REDUCE_MAX: u32 = 0x46;
pub const OP_REDUCE_AND: u32 = 0x47;
pub const OP_REDUCE_OR: u32 = 0x48;
pub const OP_REDUCE_XOR: u32 = 0x49;
pub const OP_VMID: u32 = 0x4A;
pub const OP_CLOCK: u32 = 0x4B;
pub const OP_RAND: u32 = 0x4C;

pub const OP_LOAD: u32 = 0x50;
pub const OP_STORE: u32 = 0x51;

pub const OP_LOG: u32 = 0x58;
pub const OP_LOG_I: u32 = 0x59;

// 0x60-0x6F reserved for messaging.

pub const OP_JMP: u32 = 0x70;
pub const OP_JMP_IF_ANY: u32 = 0x71;
pub const OP_JMP_IF_ALL: u32 = 0x72;
pub const OP_CALL: u32 = 0x73;
pub const OP_RET: u32 = 0x74;
pub const OP_HALT: u32 = 0x75;
pub const OP_YIELD: u32 = 0x76;
pub const OP_STEP_TRAP: u32 = 0x77;

pub const OP_S_MOV: u32 = 0x78;
pub const OP_S_MOV_I: u32 = 0x79;
pub const OP_S_ADD: u32 = 0x7A;
pub const OP_S_ADD_I: u32 = 0x7B;
pub const OP_S_LDW: u32 = 0x7C;
pub const OP_S_CMP_LT: u32 = 0x7D;
pub const OP_S_CMP_LT_I: u32 = 0x7E;

pub fn enc_r(op: u32, guard: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (op << OPCODE_SHIFT)
        | (guard << GUARD_SHIFT)
        | (rd << RD_SHIFT)
        | (rs1 << RS1_SHIFT)
        | (rs2 & REG_FIELD_MASK)
}

pub fn enc_i(op: u32, guard: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
    (op << OPCODE_SHIFT)
        | (guard << GUARD_SHIFT)
        | (rd << RD_SHIFT)
        | (rs1 << RS1_SHIFT)
        | ((imm as u32) & LO_MASK)
}

pub struct Decoded {
    pub op: u32,
    pub guard: u32,
    pub rd: u32,
    pub rs1: u32,
    pub lo: u32,
}

impl Decoded {
    pub fn rs2(&self) -> u32 {
        self.lo & REG_FIELD_MASK
    }

    pub fn imm(&self) -> i32 {
        sign_ext13(self.lo)
    }
}

pub fn decode(word: u32) -> Decoded {
    Decoded {
        op: (word >> OPCODE_SHIFT) & 0x7F,
        guard: (word >> GUARD_SHIFT) & 0xF,
        rd: (word >> RD_SHIFT) & REG_FIELD_MASK,
        rs1: (word >> RS1_SHIFT) & REG_FIELD_MASK,
        lo: word & LO_MASK,
    }
}

pub fn sign_ext13(lo: u32) -> i32 {
    if lo & 0x1000 != 0 {
        (lo | 0xFFFF_E000) as i32
    } else {
        lo as i32
    }
}

/// Render a guard field as assembly prefix text ("" when unconditional).
pub fn guard_text(guard: u32) -> String {
    match guard {
        0 => String::new(),
        1..=4 => format!("@p{}", guard - 1),
        5..=8 => format!("@!p{}", guard - 5),
        _ => format!("@?<{}>", guard),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_layout() {
        let w = enc_r(0x02, 0, 1, 2, 3);
        let d = decode(w);
        assert_eq!(d.op, 0x02);
        assert_eq!(d.rd, 1);
        assert_eq!(d.rs1, 2);
        assert_eq!(d.rs2(), 3);
    }

    #[test]
    fn immediate_roundtrip() {
        for imm in [-4096i32, -1, 0, 1, 42, 4095] {
            let w = enc_i(OP_ADD_I, 0, 0, 0, imm);
            assert_eq!(decode(w).imm(), imm);
        }
    }

    #[test]
    fn guard_names() {
        assert_eq!(guard_text(0), "");
        assert_eq!(guard_text(1), "@p0");
        assert_eq!(guard_text(4), "@p3");
        assert_eq!(guard_text(5), "@!p0");
        assert_eq!(guard_text(8), "@!p3");
    }
}
