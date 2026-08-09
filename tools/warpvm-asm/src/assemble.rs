// Single-pass assembler with jump fixups: .wva -> bytecode + literal pool.
//
// Design notes:
// - Encoded in one pass. Out-of-range constants are materialised through the
//   scratch register at the point of use (1 source line -> 2 words).
// - Labels record their code offset as encoding progresses, so they already
//   account for any materialisation emitted before them.
// - Forward jump references emit a placeholder target and are patched in
//   `resolve_fixups` once every label's offset is known.
// - `.const` must be defined before use (materialisation needs the literal
//   index). Labels may be referenced before they are defined.

use std::collections::HashMap;

use crate::isa;
use crate::parse::{self, Operand, Program, Stmt};

pub struct Assembled {
    pub code: Vec<u32>,
    pub literals: Vec<u32>,
}

const DEFAULT_SCRATCH: u32 = 15;
// S_LDW/S_ADD write scalar fields (0-7), so scalar materialisation needs its
// own scratch distinct from the vector scratch.
const SCRATCH_SCALAR: u32 = 7;

struct Asm {
    code: Vec<u32>,
    literals: Vec<u32>,
    lit_by_value: HashMap<u32, u32>,
    consts: HashMap<String, i64>,
    labels: HashMap<String, u32>,
    fixups: Vec<(usize, String, usize)>, // (code index, label, line)
    ldw_refs: Vec<(usize, u32)>,         // (line, literal index)
    scratch: u32,
}

enum ImmRes {
    Inline(i32),
    ViaLit(u32),
}

impl Asm {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            literals: Vec::new(),
            lit_by_value: HashMap::new(),
            consts: HashMap::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
            ldw_refs: Vec::new(),
            scratch: DEFAULT_SCRATCH,
        }
    }

    fn emit(&mut self, word: u32) {
        self.code.push(word);
    }

    fn intern(&mut self, value: u32) -> Result<u32, String> {
        if let Some(&idx) = self.lit_by_value.get(&value) {
            return Ok(idx);
        }
        if self.literals.len() >= isa::MAX_LITERALS {
            return Err("literal pool full".to_string());
        }
        let idx = self.literals.len() as u32;
        self.literals.push(value);
        self.lit_by_value.insert(value, idx);
        Ok(idx)
    }

    fn resolve_imm(&mut self, op: &Operand, line: usize) -> Result<ImmRes, String> {
        let value: i64 = match op {
            Operand::Imm(v) => *v,
            Operand::Name(n) => *self.consts.get(n).ok_or_else(|| {
                format!("line {line}: unknown constant '{n}' (define it with .const before use)")
            })?,
            _ => return Err(format!("line {line}: expected immediate or constant")),
        };
        if (isa::IMM_MIN..=isa::IMM_MAX).contains(&value) {
            return Ok(ImmRes::Inline(value as i32));
        }
        let idx = self.intern(value as u32)?;
        Ok(ImmRes::ViaLit(idx))
    }

    // ---- operand extractors -------------------------------------------------
    fn vreg(o: &Operand, line: usize, what: &str) -> Result<u32, String> {
        match o {
            Operand::VReg(n) => Ok(*n),
            _ => Err(format!("line {line}: {what} must be a vector register")),
        }
    }
    fn sreg(o: &Operand, line: usize, what: &str) -> Result<u32, String> {
        match o {
            Operand::SReg(n) => Ok(*n),
            _ => Err(format!("line {line}: {what} must be a scalar register")),
        }
    }
    fn pred(o: &Operand, line: usize, what: &str) -> Result<u32, String> {
        match o {
            Operand::Pred(n) => Ok(*n),
            _ => Err(format!("line {line}: {what} must be a predicate (p0-p3)")),
        }
    }

    // ---- instruction encoders ----------------------------------------------

    // 2-operand vector binary: MOV rD, rS1 | MOV rD, imm.
    fn encode_mov(&mut self, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        match ops {
            [rd, Operand::VReg(rs1)] => {
                let rd = Self::vreg(rd, line, "destination")?;
                self.emit(isa::enc_r(isa::OP_MOV, guard, rd, *rs1, 0));
            }
            [rd, third] => {
                let rd = Self::vreg(rd, line, "destination")?;
                match self.resolve_imm(third, line)? {
                    ImmRes::Inline(imm) => {
                        self.emit(isa::enc_i(isa::OP_MOV_I, guard, rd, 0, imm));
                    }
                    ImmRes::ViaLit(idx) => {
                        self.emit(isa::enc_i(isa::OP_LDW, 0, self.scratch, 0, idx as i32));
                        self.emit(isa::enc_r(isa::OP_MOV, guard, rd, self.scratch, 0));
                    }
                }
            }
            _ => return Err(format!("line {line}: MOV takes rD, rS1 or rD, immediate")),
        }
        Ok(())
    }

    // 3-operand vector binary. `op_i` is the immediate-form opcode when one
    // exists; ops without an I-form (DIV/MOD/MIN/MAX) materialise the
    // immediate into the scratch register instead.
    fn encode_bin3(
        &mut self,
        op_r: u32,
        op_i: Option<u32>,
        guard: u32,
        ops: &[Operand],
        line: usize,
    ) -> Result<(), String> {
        let [rd, rs1, third] = ops else {
            return Err(format!("line {line}: expected rD, rS1, rS2/immediate"));
        };
        let rd = Self::vreg(rd, line, "destination")?;
        let rs1 = Self::vreg(rs1, line, "first source")?;
        if let Operand::VReg(rs2) = third {
            self.emit(isa::enc_r(op_r, guard, rd, rs1, *rs2));
            return Ok(());
        }
        match self.resolve_imm(third, line)? {
            ImmRes::Inline(imm) => match op_i {
                Some(op_i) => self.emit(isa::enc_i(op_i, guard, rd, rs1, imm)),
                None => {
                    self.emit(isa::enc_i(isa::OP_MOV_I, 0, self.scratch, 0, imm));
                    self.emit(isa::enc_r(op_r, guard, rd, rs1, self.scratch));
                }
            },
            ImmRes::ViaLit(idx) => {
                self.emit(isa::enc_i(isa::OP_LDW, 0, self.scratch, 0, idx as i32));
                self.emit(isa::enc_r(op_r, guard, rd, rs1, self.scratch));
            }
        }
        Ok(())
    }

    // Comparison into a predicate: CMP_xx pD, rA, rB/imm.
    fn encode_cmp(
        &mut self,
        op_r: u32,
        op_i: u32,
        guard: u32,
        ops: &[Operand],
        line: usize,
    ) -> Result<(), String> {
        let [pd, rs1, third] = ops else {
            return Err(format!("line {line}: expected pD, rS1, rS2/immediate"));
        };
        let pd = Self::pred(pd, line, "destination predicate")?;
        let rs1 = Self::vreg(rs1, line, "first source")?;
        if let Operand::VReg(rs2) = third {
            self.emit(isa::enc_r(op_r, guard, pd, rs1, *rs2));
            return Ok(());
        }
        match self.resolve_imm(third, line)? {
            ImmRes::Inline(imm) => self.emit(isa::enc_i(op_i, guard, pd, rs1, imm)),
            ImmRes::ViaLit(idx) => {
                self.emit(isa::enc_i(isa::OP_LDW, 0, self.scratch, 0, idx as i32));
                self.emit(isa::enc_r(op_r, guard, pd, rs1, self.scratch));
            }
        }
        Ok(())
    }

    // Unary vector op: OP rD, rS1.
    fn encode_unary(&mut self, op: u32, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        let [rd, rs1] = ops else {
            return Err(format!("line {line}: expected rD, rS1"));
        };
        let rd = Self::vreg(rd, line, "destination")?;
        let rs1 = Self::vreg(rs1, line, "source")?;
        self.emit(isa::enc_r(op, guard, rd, rs1, 0));
        Ok(())
    }

    // Destination-only vector op: OP rD (LANEID, VMID, CLOCK, RAND).
    fn encode_dest_only(&mut self, op: u32, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        let [rd] = ops else {
            return Err(format!("line {line}: expected rD"));
        };
        let rd = Self::vreg(rd, line, "destination")?;
        self.emit(isa::enc_r(op, guard, rd, 0, 0));
        Ok(())
    }

    // Mask ops over predicates: NOTMASK pD,pS | ANDMASK/ORMASK pD,pA,pB.
    fn encode_mask2(&mut self, op: u32, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        let [pd, ps1] = ops else {
            return Err(format!("line {line}: expected pD, pS"));
        };
        let pd = Self::pred(pd, line, "destination predicate")?;
        let ps1 = Self::pred(ps1, line, "source predicate")?;
        self.emit(isa::enc_r(op, guard, pd, ps1, 0));
        Ok(())
    }
    fn encode_mask3(&mut self, op: u32, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        let [pd, pa, pb] = ops else {
            return Err(format!("line {line}: expected pD, pA, pB"));
        };
        let pd = Self::pred(pd, line, "destination predicate")?;
        let pa = Self::pred(pa, line, "first source predicate")?;
        let pb = Self::pred(pb, line, "second source predicate")?;
        self.emit(isa::enc_r(op, guard, pd, pa, pb));
        Ok(())
    }

    // BALLOT pD, rS  |  ANY/ALL pD, pS.
    fn encode_ballot(&mut self, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        let [pd, rs1] = ops else {
            return Err(format!("line {line}: expected pD, rS"));
        };
        let pd = Self::pred(pd, line, "destination predicate")?;
        let rs1 = Self::vreg(rs1, line, "source")?;
        self.emit(isa::enc_r(isa::OP_BALLOT, guard, pd, rs1, 0));
        Ok(())
    }
    fn encode_anyall(&mut self, op: u32, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        let [pd, ps1] = ops else {
            return Err(format!("line {line}: expected pD, pS"));
        };
        let pd = Self::pred(pd, line, "destination predicate")?;
        let ps1 = Self::pred(ps1, line, "source predicate")?;
        self.emit(isa::enc_r(op, guard, pd, ps1, 0));
        Ok(())
    }

    // BROADCAST rD, rS, #lane  |  SHUFFLE_XOR rD, rS, #mask.
    fn encode_lane_imm(&mut self, op: u32, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        let [rd, rs1, Operand::Imm(v)] = ops else {
            return Err(format!("line {line}: expected rD, rS, #imm"));
        };
        let rd = Self::vreg(rd, line, "destination")?;
        let rs1 = Self::vreg(rs1, line, "source")?;
        if !(0..32).contains(v) {
            return Err(format!("line {line}: lane/mask must be 0..31"));
        }
        self.emit(isa::enc_i(op, guard, rd, rs1, *v as i32));
        Ok(())
    }

    fn encode_shuffle(&mut self, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        let [rd, rs1, rs2] = ops else {
            return Err(format!("line {line}: expected rD, rS1, rS2"));
        };
        let rd = Self::vreg(rd, line, "destination")?;
        let rs1 = Self::vreg(rs1, line, "first source")?;
        let rs2 = Self::vreg(rs2, line, "second source")?;
        self.emit(isa::enc_r(isa::OP_SHUFFLE, guard, rd, rs1, rs2));
        Ok(())
    }

    // ---- control flow ------------------------------------------------------

    // Encode a jump/call target. Labels are fixed up after the pass.
    fn emit_branch(&mut self, op: u32, guard: u32, target: &Operand, line: usize) -> Result<(), String> {
        let idx = self.code.len();
        match target {
            Operand::Imm(v) => {
                if *v < 0 {
                    return Err(format!("line {line}: negative jump target"));
                }
                self.emit(isa::enc_i(op, guard, 0, 0, *v as i32));
            }
            Operand::Name(n) => {
                self.fixups.push((idx, n.clone(), line));
                self.emit(isa::enc_i(op, guard, 0, 0, 0));
            }
            _ => return Err(format!("line {line}: jump target must be a label or address")),
        }
        Ok(())
    }

    // JMP_IF_ANY/ALL take their predicate from the operand, placed in the
    // guard field of the encoding.
    fn encode_jmp_if(&mut self, op: u32, guard: u32, ops: &[Operand], line: usize) -> Result<(), String> {
        if guard != 0 {
            return Err(format!(
                "line {line}: JMP_IF_* takes its predicate as an operand, not an @-guard"
            ));
        }
        let [pn, target] = ops else {
            return Err(format!("line {line}: expected pN, target"));
        };
        let pn = Self::pred(pn, line, "condition predicate")?;
        self.emit_branch(op, 1 + pn, target, line)
    }

    fn resolve_fixups(&mut self) -> Result<(), String> {
        let len = self.code.len() as u32;
        for (idx, name, line) in std::mem::take(&mut self.fixups) {
            let target = *self
                .labels
                .get(&name)
                .ok_or_else(|| format!("line {line}: undefined label '{name}'"))?;
            if target > len {
                return Err(format!("line {line}: label '{name}' out of range"));
            }
            self.code[idx] = (self.code[idx] & !isa::LO_MASK) | (target & isa::LO_MASK);
        }
        Ok(())
    }

    fn directive(&mut self, name: &str, args: &[String], line: usize) -> Result<(), String> {
        match name {
            "const" => {
                let [sym, value] = args else {
                    return Err(format!("line {line}: .const takes NAME VALUE"));
                };
                if !parse::is_identifier(sym) {
                    return Err(format!("line {line}: bad constant name '{sym}'"));
                }
                let v = parse::parse_number(value, line)?;
                if !(0..=0xFFFF_FFFF).contains(&v) {
                    return Err(format!("line {line}: constant out of u32 range"));
                }
                if self.consts.insert(sym.clone(), v).is_some() {
                    return Err(format!("line {line}: duplicate constant '{sym}'"));
                }
                self.intern(v as u32)?;
            }
            "scratch" => {
                let [reg] = args else {
                    return Err(format!("line {line}: .scratch takes one register"));
                };
                match parse::parse_operand(reg, line)? {
                    Operand::VReg(n) => self.scratch = n,
                    _ => return Err(format!("line {line}: .scratch needs a vector register")),
                }
            }
            "word" => {
                let [value] = args else {
                    return Err(format!("line {line}: .word takes one value"));
                };
                let v = parse::parse_number(value, line)?;
                if !(0..=0xFFFF_FFFF).contains(&v) {
                    return Err(format!("line {line}: .word out of u32 range"));
                }
                self.emit(v as u32);
            }
            other => return Err(format!("line {line}: unknown directive '.{other}'")),
        }
        Ok(())
    }

    fn instr(&mut self, guard: u32, mnemonic: &str, ops: &[Operand], line: usize) -> Result<(), String> {
        use isa::*;
        match mnemonic {
            "NOP" => self.emit(enc_r(OP_NOP, guard, 0, 0, 0)),
            "HALT" => self.emit(enc_r(OP_HALT, guard, 0, 0, 0)),
            "YIELD" => self.emit(enc_r(OP_YIELD, guard, 0, 0, 0)),

            "MOV" | "MOV_I" => self.encode_mov(guard, ops, line)?,

            "ADD" | "ADD_I" => self.encode_bin3(OP_ADD, Some(OP_ADD_I), guard, ops, line)?,
            "SUB" | "SUB_I" => self.encode_bin3(OP_SUB, Some(OP_SUB_I), guard, ops, line)?,
            "MUL" | "MUL_I" => self.encode_bin3(OP_MUL, Some(OP_MUL_I), guard, ops, line)?,
            "AND" | "AND_I" => self.encode_bin3(OP_AND, Some(OP_AND_I), guard, ops, line)?,
            "OR" | "OR_I" => self.encode_bin3(OP_OR, Some(OP_OR_I), guard, ops, line)?,
            "XOR" | "XOR_I" => self.encode_bin3(OP_XOR, Some(OP_XOR_I), guard, ops, line)?,
            "SHL" | "SHL_I" => self.encode_bin3(OP_SHL, Some(OP_SHL_I), guard, ops, line)?,
            "SHR" | "SHR_I" => self.encode_bin3(OP_SHR, Some(OP_SHR_I), guard, ops, line)?,
            "DIV" => self.encode_bin3(OP_DIV, None, guard, ops, line)?,
            "MOD" => self.encode_bin3(OP_MOD, None, guard, ops, line)?,
            "MIN" => self.encode_bin3(OP_MIN, None, guard, ops, line)?,
            "MAX" => self.encode_bin3(OP_MAX, None, guard, ops, line)?,

            "ABS" => self.encode_unary(OP_ABS, guard, ops, line)?,
            "NEG" => self.encode_unary(OP_NEG, guard, ops, line)?,
            "NOT" => self.encode_unary(OP_NOT, guard, ops, line)?,

            "CMP_EQ" => self.encode_cmp(OP_CMP_EQ, OP_CMP_EQ_I, guard, ops, line)?,
            "CMP_NE" => self.encode_cmp(OP_CMP_NE, OP_CMP_NE_I, guard, ops, line)?,
            "CMP_LT" => self.encode_cmp(OP_CMP_LT, OP_CMP_LT_I, guard, ops, line)?,
            "CMP_LE" => self.encode_cmp(OP_CMP_LE, OP_CMP_LE_I, guard, ops, line)?,
            "CMP_GT" => self.encode_cmp(OP_CMP_GT, OP_CMP_GT_I, guard, ops, line)?,
            "CMP_GE" => self.encode_cmp(OP_CMP_GE, OP_CMP_GE_I, guard, ops, line)?,

            "NOTMASK" => self.encode_mask2(OP_NOTMASK, guard, ops, line)?,
            "ANDMASK" => self.encode_mask3(OP_ANDMASK, guard, ops, line)?,
            "ORMASK" => self.encode_mask3(OP_ORMASK, guard, ops, line)?,
            "BALLOT" => self.encode_ballot(guard, ops, line)?,
            "ANY" => self.encode_anyall(OP_ANY, guard, ops, line)?,
            "ALL" => self.encode_anyall(OP_ALL, guard, ops, line)?,

            "LANEID" => self.encode_dest_only(OP_LANEID, guard, ops, line)?,
            "VMID" => self.encode_dest_only(OP_VMID, guard, ops, line)?,
            "CLOCK" => self.encode_dest_only(OP_CLOCK, guard, ops, line)?,
            "RAND" => self.encode_dest_only(OP_RAND, guard, ops, line)?,
            "BROADCAST" => self.encode_lane_imm(OP_BROADCAST, guard, ops, line)?,
            "SHUFFLE_XOR" => self.encode_lane_imm(OP_SHUFFLE_XOR, guard, ops, line)?,
            "SHUFFLE" => self.encode_shuffle(guard, ops, line)?,
            "REDUCE_ADD" => self.encode_unary(OP_REDUCE_ADD, guard, ops, line)?,
            "REDUCE_MIN" => self.encode_unary(OP_REDUCE_MIN, guard, ops, line)?,
            "REDUCE_MAX" => self.encode_unary(OP_REDUCE_MAX, guard, ops, line)?,
            "REDUCE_AND" => self.encode_unary(OP_REDUCE_AND, guard, ops, line)?,
            "REDUCE_OR" => self.encode_unary(OP_REDUCE_OR, guard, ops, line)?,
            "REDUCE_XOR" => self.encode_unary(OP_REDUCE_XOR, guard, ops, line)?,

            "LOAD" | "STORE" => {
                let [a, b] = ops else {
                    return Err(format!("line {line}: {mnemonic} takes two vector registers"));
                };
                let a = Self::vreg(a, line, "first operand")?;
                let b = Self::vreg(b, line, "second operand")?;
                let op = if mnemonic == "LOAD" { OP_LOAD } else { OP_STORE };
                self.emit(enc_r(op, guard, a, b, 0));
            }

            "LDW" => {
                let [rd, Operand::Imm(idx)] = ops else {
                    return Err(format!("line {line}: LDW takes rD and a literal-pool index"));
                };
                let rd = Self::vreg(rd, line, "destination")?;
                if *idx < 0 {
                    return Err(format!("line {line}: negative literal index"));
                }
                self.ldw_refs.push((line, *idx as u32));
                self.emit(enc_i(OP_LDW, guard, rd, 0, *idx as i32));
            }

            // ---- scalar ops ----
            "S_MOV" => {
                let [sd, ss1] = ops else {
                    return Err(format!("line {line}: S_MOV takes sD, sS"));
                };
                self.emit(enc_r(OP_S_MOV, guard, Self::sreg(sd, line, "dest")?, Self::sreg(ss1, line, "source")?, 0));
            }
            "S_MOV_I" => {
                let [sd, imm] = ops else {
                    return Err(format!("line {line}: S_MOV_I takes sD, immediate"));
                };
                let sd = Self::sreg(sd, line, "destination")?;
                match self.resolve_imm(imm, line)? {
                    ImmRes::Inline(v) => self.emit(enc_i(OP_S_MOV_I, guard, sd, 0, v)),
                    ImmRes::ViaLit(idx) => self.emit(enc_i(OP_S_LDW, guard, sd, 0, idx as i32)),
                }
            }
            "S_ADD" => {
                let [sd, sa, sb] = ops else {
                    return Err(format!("line {line}: S_ADD takes sD, sA, sB"));
                };
                self.emit(enc_r(OP_S_ADD, guard, Self::sreg(sd, line, "dest")?, Self::sreg(sa, line, "src1")?, Self::sreg(sb, line, "src2")?));
            }
            "S_ADD_I" => {
                let [sd, sa, imm] = ops else {
                    return Err(format!("line {line}: S_ADD_I takes sD, sA, immediate"));
                };
                let sd = Self::sreg(sd, line, "destination")?;
                let sa = Self::sreg(sa, line, "source")?;
                match self.resolve_imm(imm, line)? {
                    ImmRes::Inline(v) => self.emit(enc_i(OP_S_ADD_I, guard, sd, sa, v)),
                    ImmRes::ViaLit(idx) => {
                        self.emit(enc_i(OP_S_LDW, 0, SCRATCH_SCALAR, 0, idx as i32));
                        self.emit(enc_r(OP_S_ADD, guard, sd, sa, SCRATCH_SCALAR));
                    }
                }
            }
            "S_LDW" => {
                let [sd, Operand::Imm(idx)] = ops else {
                    return Err(format!("line {line}: S_LDW takes sD and a literal-pool index"));
                };
                let sd = Self::sreg(sd, line, "destination")?;
                if *idx < 0 {
                    return Err(format!("line {line}: negative literal index"));
                }
                self.ldw_refs.push((line, *idx as u32));
                self.emit(enc_i(OP_S_LDW, guard, sd, 0, *idx as i32));
            }
            "S_CMP_LT" => {
                let [pd, sa, sb] = ops else {
                    return Err(format!("line {line}: S_CMP_LT takes pD, sA, sB"));
                };
                self.emit(enc_r(OP_S_CMP_LT, guard, Self::pred(pd, line, "dest predicate")?, Self::sreg(sa, line, "src1")?, Self::sreg(sb, line, "src2")?));
            }
            "S_CMP_LT_I" => {
                let [pd, sa, imm] = ops else {
                    return Err(format!("line {line}: S_CMP_LT_I takes pD, sA, immediate"));
                };
                let pd = Self::pred(pd, line, "dest predicate")?;
                let sa = Self::sreg(sa, line, "source")?;
                match self.resolve_imm(imm, line)? {
                    ImmRes::Inline(v) => self.emit(enc_i(OP_S_CMP_LT_I, guard, pd, sa, v)),
                    ImmRes::ViaLit(idx) => {
                        self.emit(enc_i(OP_S_LDW, 0, SCRATCH_SCALAR, 0, idx as i32));
                        self.emit(enc_r(OP_S_CMP_LT, guard, pd, sa, SCRATCH_SCALAR));
                    }
                }
            }
            "S_BCAST" => {
                let [rd, ss1] = ops else {
                    return Err(format!("line {line}: S_BCAST takes rD, sS"));
                };
                self.emit(enc_r(OP_S_BCAST, guard, Self::vreg(rd, line, "dest")?, Self::sreg(ss1, line, "source")?, 0));
            }
            "S_GET" => {
                let [sd, rs1] = ops else {
                    return Err(format!("line {line}: S_GET takes sD, rS"));
                };
                self.emit(enc_r(OP_S_GET, guard, Self::sreg(sd, line, "dest")?, Self::vreg(rs1, line, "source")?, 0));
            }

            // ---- control flow ----
            "JMP" => {
                let [target] = ops else {
                    return Err(format!("line {line}: JMP takes a target"));
                };
                self.emit_branch(OP_JMP, guard, target, line)?;
            }
            "CALL" => {
                let [target] = ops else {
                    return Err(format!("line {line}: CALL takes a target"));
                };
                self.emit_branch(OP_CALL, guard, target, line)?;
            }
            "RET" => self.emit(enc_r(OP_RET, guard, 0, 0, 0)),
            "JMP_IF_ANY" => self.encode_jmp_if(OP_JMP_IF_ANY, guard, ops, line)?,
            "JMP_IF_ALL" => self.encode_jmp_if(OP_JMP_IF_ALL, guard, ops, line)?,

            other => {
                return Err(format!(
                    "line {line}: unknown mnemonic '{other}' (see docs/isa.md for the v0.1 set)"
                ))
            }
        }
        Ok(())
    }

    fn pass(&mut self, prog: &Program) -> Result<(), String> {
        for stmt in &prog.stmts {
            match stmt {
                Stmt::Label { line, name } => {
                    if self.labels.insert(name.clone(), self.code.len() as u32).is_some() {
                        return Err(format!("line {line}: duplicate label '{name}'"));
                    }
                }
                Stmt::Directive { line, name, args } => self.directive(name, args, *line)?,
                Stmt::Instr { line, guard, mnemonic, operands } => {
                    self.instr(*guard, mnemonic, operands, *line)?;
                }
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Assembled, String> {
        self.resolve_fixups()?;
        for (line, idx) in &self.ldw_refs {
            if *idx as usize >= self.literals.len() {
                return Err(format!("line {line}: literal index {idx} out of range"));
            }
        }
        if self.code.len() > isa::MAX_CODE_WORDS {
            return Err(format!("program too long ({} words)", self.code.len()));
        }
        if self.code.is_empty() {
            return Err("empty program".to_string());
        }
        Ok(Assembled { code: self.code, literals: self.literals })
    }
}

pub fn assemble(src: &str) -> Result<Assembled, String> {
    let prog = parse::parse(src)?;
    let mut asm = Asm::new();
    asm.pass(&prog)?;
    asm.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disasm;
    use crate::isa::*;

    #[test]
    fn assembles_hello() {
        let src = "MOV_I r0, 7\nADD_I r0, r0, 35\nMOV r1, r0\nADD r2, r0, r1\nHALT\n";
        let a = assemble(src).unwrap();
        assert_eq!(
            a.code,
            vec![
                enc_i(OP_MOV_I, 0, 0, 0, 7),
                enc_i(OP_ADD_I, 0, 0, 0, 35),
                enc_r(OP_MOV, 0, 1, 0, 0),
                enc_r(OP_ADD, 0, 2, 0, 1),
                enc_r(OP_HALT, 0, 0, 0, 0),
            ]
        );
        assert!(a.literals.is_empty());
    }

    #[test]
    fn materialises_large_constants() {
        let src = ".const BIG 123456\nADD_I r0, r0, BIG\nHALT\n";
        let a = assemble(src).unwrap();
        assert_eq!(a.literals, vec![123456]);
        assert_eq!(
            a.code,
            vec![
                enc_i(OP_LDW, 0, DEFAULT_SCRATCH, 0, 0),
                enc_r(OP_ADD, 0, 0, 0, DEFAULT_SCRATCH),
                enc_r(OP_HALT, 0, 0, 0, 0),
            ]
        );
    }

    #[test]
    fn div_no_imm_form_materialises() {
        let src = "DIV r0, r1, 3\nHALT\n";
        let a = assemble(src).unwrap();
        assert_eq!(a.code[0], enc_i(OP_MOV_I, 0, DEFAULT_SCRATCH, 0, 3));
        assert_eq!(a.code[1], enc_r(OP_DIV, 0, 0, 1, DEFAULT_SCRATCH));
    }

    #[test]
    fn small_constants_inline() {
        let src = ".const SMALL 5\nADD_I r0, r0, SMALL\nHALT\n";
        let a = assemble(src).unwrap();
        assert_eq!(a.literals, vec![5]);
        assert_eq!(a.code[0], enc_i(OP_ADD_I, 0, 0, 0, 5));
    }

    #[test]
    fn guards_encode() {
        let a = assemble("@p2 MOV r0, r1\n@!p0 ADD r2, r3, r4\nHALT\n").unwrap();
        assert_eq!(decode(a.code[0]).guard, 3);
        assert_eq!(decode(a.code[1]).guard, 5);
    }

    #[test]
    fn memory_ops_encode() {
        let a = assemble("LOAD r3, r2\nSTORE r2, r3\nHALT\n").unwrap();
        assert_eq!(a.code[0], enc_r(OP_LOAD, 0, 3, 2, 0));
        assert_eq!(a.code[1], enc_r(OP_STORE, 0, 2, 3, 0));
    }

    #[test]
    fn labels_forward_and_back() {
        // start=0; JMP(0) NOP(1) HALT(2); end points at HALT (offset 2).
        let src = "start:\nJMP end\nNOP\nend:\nHALT\n";
        let a = assemble(src).unwrap();
        assert_eq!(a.code[0], enc_i(OP_JMP, 0, 0, 0, 2));
        assert_eq!(a.code[2], enc_r(OP_HALT, 0, 0, 0, 0));
    }

    #[test]
    fn labels_account_for_materialisation() {
        let src = ".const BIG 123456\nJMP done\nADD_I r0, r0, BIG\ndone:\nHALT\n";
        let a = assemble(src).unwrap();
        // ADD_I BIG expands to LDW+ADD (2 words), so done is at offset 3.
        assert_eq!(a.code[0], enc_i(OP_JMP, 0, 0, 0, 3));
        assert_eq!(a.code.len(), 4);
    }

    #[test]
    fn jmp_if_uses_guard_field() {
        let src = "loop:\nJMP_IF_ANY p1, loop\nHALT\n";
        let a = assemble(src).unwrap();
        let d = decode(a.code[0]);
        assert_eq!(d.op, OP_JMP_IF_ANY);
        assert_eq!(d.guard, 2); // p1 -> guard 1+1
        assert_eq!(d.imm(), 0); // loop is at 0
    }

    #[test]
    fn warp_native_ops_encode() {
        let a = assemble(
            "LANEID r0\nREDUCE_ADD r1, r0\nBROADCAST r2, r0, 5\nSHUFFLE_XOR r3, r0, 1\nCMP_LT p0, r0, 16\nHALT\n",
        )
        .unwrap();
        assert_eq!(a.code[0], enc_r(OP_LANEID, 0, 0, 0, 0));
        assert_eq!(a.code[1], enc_r(OP_REDUCE_ADD, 0, 1, 0, 0));
        assert_eq!(a.code[2], enc_i(OP_BROADCAST, 0, 2, 0, 5));
        assert_eq!(a.code[3], enc_i(OP_SHUFFLE_XOR, 0, 3, 0, 1));
        assert_eq!(a.code[4], enc_i(OP_CMP_LT_I, 0, 0, 0, 16));
    }

    #[test]
    fn scalar_ops_encode() {
        let a = assemble("S_MOV_I s0, 0\nS_ADD_I s0, s0, 32\nS_BCAST r1, s0\nS_GET s1, r1\nHALT\n").unwrap();
        assert_eq!(a.code[0], enc_i(OP_S_MOV_I, 0, 0, 0, 0));
        assert_eq!(a.code[1], enc_i(OP_S_ADD_I, 0, 0, 0, 32));
        assert_eq!(a.code[2], enc_r(OP_S_BCAST, 0, 1, 0, 0));
        assert_eq!(a.code[3], enc_r(OP_S_GET, 0, 1, 1, 0));
    }

    #[test]
    fn rejects_undefined_label() {
        assert!(assemble("JMP nowhere\nHALT\n").is_err());
    }

    #[test]
    fn rejects_const_before_def() {
        assert!(assemble("ADD_I r0, r0, LATE\n.const LATE 5\nHALT\n").is_err());
    }

    #[test]
    fn roundtrip_stable() {
        let src = ".const BIG 123456\nMOV_I r0, 7\nADD_I r0, r0, BIG\nADD_I r1, r0, -3\nSTORE r0, r1\nLOAD r2, r0\nLANEID r3\nREDUCE_ADD r4, r3\nHALT\n";
        let a = assemble(src).unwrap();
        let text = disasm::disassemble(&a.code, &a.literals);
        let b = assemble(&text).expect("re-assemble failed");
        assert_eq!(a.code, b.code);
        assert_eq!(a.literals, b.literals);
    }

    #[test]
    fn roundtrip_control_flow() {
        let src = ".const N 100\nS_MOV_I s0, 0\nloop:\nLANEID r0\nS_BCAST r1, s0\nADD r2, r0, r1\nCMP_LT p0, r2, N\n@p0 LOAD r3, r2\n@p0 MUL r3, r3, r3\n@p0 STORE r2, r3\nS_ADD_I s0, s0, 32\nS_CMP_LT_I p1, s0, N\nJMP_IF_ANY p1, loop\nCALL done\nRET\ndone:\nHALT\n";
        let a = assemble(src).unwrap();
        let text = disasm::disassemble(&a.code, &a.literals);
        let b = assemble(&text).expect("re-assemble failed");
        assert_eq!(a.code, b.code);
        assert_eq!(a.literals, b.literals);
    }

    #[test]
    fn rejects_unknown_mnemonic() {
        assert!(assemble("FOO r0, r1\n").is_err());
        assert!(assemble("ADD_I r0, r0, NOPE\n").is_err());
        assert!(assemble("LDW r0, 9\n").is_err()); // index out of range
    }
}
