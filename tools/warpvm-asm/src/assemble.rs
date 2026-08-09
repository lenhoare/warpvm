// Two-pass assembler: .wva statements -> bytecode + literal pool.
//
// Slice 2 opcode coverage: NOP, MOV/MOV_I, ADD/ADD_I, LDW, HALT. The
// parser is already general; each later slice grows the mnemonic table.
//
// NOTE for the control-flow slice: materialisation inserts extra words
// during pass 2. Once jump targets exist, materialised LDWs must be
// accounted for before labels are resolved.

use std::collections::HashMap;

use crate::isa;
use crate::parse::{self, Operand, Stmt};

pub struct Assembled {
    pub code: Vec<u32>,
    pub literals: Vec<u32>,
}

const DEFAULT_SCRATCH: u32 = 15;

pub fn assemble(src: &str) -> Result<Assembled, String> {
    let prog = parse::parse(src)?;

    let mut labels: HashMap<String, u32> = HashMap::new();
    let mut consts: HashMap<String, i64> = HashMap::new();
    let mut literals: Vec<u32> = Vec::new();
    let mut lit_by_value: HashMap<u32, u32> = HashMap::new();
    let mut scratch = DEFAULT_SCRATCH;

    let intern = |literals: &mut Vec<u32>,
                  lit_by_value: &mut HashMap<u32, u32>,
                  value: u32|
     -> Result<u32, String> {
        if let Some(&idx) = lit_by_value.get(&value) {
            return Ok(idx);
        }
        if literals.len() >= isa::MAX_LITERALS {
            return Err("literal pool full".to_string());
        }
        let idx = literals.len() as u32;
        literals.push(value);
        lit_by_value.insert(value, idx);
        Ok(idx)
    };

    // ---- pass 1: labels, constants, word count ----------------------------
    let mut pc: u32 = 0;
    for stmt in &prog.stmts {
        match stmt {
            Stmt::Label { line, name } => {
                if labels.insert(name.clone(), pc).is_some() {
                    return Err(format!("line {line}: duplicate label '{name}'"));
                }
            }
            Stmt::Directive { line, name, args } => match name.as_str() {
                "const" => {
                    let [sym, value] = args.as_slice() else {
                        return Err(format!("line {line}: .const takes NAME VALUE"));
                    };
                    if !parse::is_identifier(sym) {
                        return Err(format!("line {line}: bad constant name '{sym}'"));
                    }
                    let v = parse::parse_number(value, *line)?;
                    if !(0..=0xFFFF_FFFF).contains(&v) {
                        return Err(format!("line {line}: constant out of u32 range"));
                    }
                    if consts.insert(sym.clone(), v).is_some() {
                        return Err(format!("line {line}: duplicate constant '{sym}'"));
                    }
                    intern(&mut literals, &mut lit_by_value, v as u32)?;
                }
                "scratch" => {
                    let [reg] = args.as_slice() else {
                        return Err(format!("line {line}: .scratch takes one register"));
                    };
                    match parse::parse_operand(reg, *line)? {
                        Operand::VReg(n) => scratch = n,
                        _ => return Err(format!("line {line}: .scratch needs a vector register")),
                    }
                }
                "word" => {
                    let [value] = args.as_slice() else {
                        return Err(format!("line {line}: .word takes one value"));
                    };
                    let v = parse::parse_number(value, *line)?;
                    if !(0..=0xFFFF_FFFF).contains(&v) {
                        return Err(format!("line {line}: .word out of u32 range"));
                    }
                    pc += 1;
                }
                other => return Err(format!("line {line}: unknown directive '.{other}'")),
            },
            Stmt::Instr { .. } => pc += 1,
        }
    }

    // ---- pass 2: encode -----------------------------------------------------
    let mut code: Vec<u32> = Vec::new();
    let mut ldw_refs: Vec<(usize, u32)> = Vec::new(); // (line, literal index)

    enum Target {
        Inline(i32),
        ViaLiteral(u32),
    }

    // Resolve an operand in immediate position.
    let resolve_imm = |op: &Operand,
                           line: usize,
                           consts: &HashMap<String, i64>,
                           literals: &mut Vec<u32>,
                           lit_by_value: &mut HashMap<u32, u32>|
     -> Result<Target, String> {
        let value: i64 = match op {
            Operand::Imm(v) => *v,
            Operand::Name(n) => *consts
                .get(n)
                .ok_or_else(|| format!("line {line}: unknown constant '{n}' (jump targets arrive with the control-flow slice)"))?,
            _ => return Err(format!("line {line}: expected immediate")),
        };
        if (isa::IMM_MIN..=isa::IMM_MAX).contains(&value) {
            return Ok(Target::Inline(value as i32));
        }
        let idx = intern(literals, lit_by_value, value as u32)?;
        Ok(Target::ViaLiteral(idx))
    };

    // Encode `OP rd, rs1, imm`, materialising out-of-range constants through
    // the scratch register.
    let encode_binary_imm = |mnemonic_r: u32,
                             mnemonic_i: u32,
                             guard: u32,
                             ops: &[Operand],
                             line: usize,
                             consts: &HashMap<String, i64>,
                             literals: &mut Vec<u32>,
                             lit_by_value: &mut HashMap<u32, u32>,
                             scratch: u32,
                             code: &mut Vec<u32>|
     -> Result<(), String> {
        let [Operand::VReg(rd), Operand::VReg(rs1), third] = ops else {
            return Err(format!("line {line}: expected rD, rS1, rS2/immediate"));
        };
        if let Operand::VReg(rs2) = third {
            code.push(isa::enc_r(mnemonic_r, guard, *rd, *rs1, *rs2));
            return Ok(());
        }
        match resolve_imm(third, line, consts, literals, lit_by_value)? {
            Target::Inline(imm) => {
                code.push(isa::enc_i(mnemonic_i, guard, *rd, *rs1, imm));
            }
            Target::ViaLiteral(idx) => {
                code.push(isa::enc_i(isa::OP_LDW, 0, scratch, 0, idx as i32));
                code.push(isa::enc_r(mnemonic_r, guard, *rd, *rs1, scratch));
            }
        }
        Ok(())
    };

    for stmt in &prog.stmts {
        let Stmt::Instr {
            line,
            guard,
            mnemonic,
            operands,
        } = stmt
        else {
            if let Stmt::Directive { name, args, .. } = stmt {
                if name == "word" {
                    let v = parse::parse_number(&args[0], 0).unwrap();
                    code.push(v as u32);
                }
            }
            continue;
        };

        match mnemonic.as_str() {
            "NOP" => {
                check_arity(operands, 0, *line)?;
                code.push(isa::enc_r(isa::OP_NOP, *guard, 0, 0, 0));
            }
            "HALT" => {
                check_arity(operands, 0, *line)?;
                code.push(isa::enc_r(isa::OP_HALT, *guard, 0, 0, 0));
            }
            "MOV" | "MOV_I" => {
                match operands.as_slice() {
                    [Operand::VReg(rd), Operand::VReg(rs1)] => {
                        code.push(isa::enc_r(isa::OP_MOV, *guard, *rd, *rs1, 0));
                    }
                    [Operand::VReg(rd), third] => {
                        match resolve_imm(
                            third,
                            *line,
                            &consts,
                            &mut literals,
                            &mut lit_by_value,
                        )? {
                            Target::Inline(imm) => {
                                code.push(isa::enc_i(isa::OP_MOV_I, *guard, *rd, 0, imm));
                            }
                            Target::ViaLiteral(idx) => {
                                code.push(isa::enc_i(isa::OP_LDW, 0, scratch, 0, idx as i32));
                                code.push(isa::enc_r(isa::OP_MOV, *guard, *rd, scratch, 0));
                            }
                        }
                    }
                    _ => {
                        return Err(format!(
                            "line {line}: MOV takes rD, rS1 or rD, immediate"
                        ))
                    }
                }
            }
            "ADD" | "ADD_I" => encode_binary_imm(
                isa::OP_ADD,
                isa::OP_ADD_I,
                *guard,
                operands,
                *line,
                &consts,
                &mut literals,
                &mut lit_by_value,
                scratch,
                &mut code,
            )?,
            "LDW" => {
                let [Operand::VReg(rd), Operand::Imm(idx)] = operands.as_slice() else {
                    return Err(format!(
                        "line {line}: LDW takes rD and a literal-pool index"
                    ));
                };
                if *idx < 0 {
                    return Err(format!("line {line}: negative literal index"));
                }
                ldw_refs.push((*line, *idx as u32));
                code.push(isa::enc_i(isa::OP_LDW, *guard, *rd, 0, *idx as i32));
            }
            "LOAD" | "STORE" => {
                let [Operand::VReg(a), Operand::VReg(b)] = operands.as_slice() else {
                    return Err(format!("line {line}: {mnemonic} takes two vector registers"));
                };
                let op = if mnemonic == "LOAD" { isa::OP_LOAD } else { isa::OP_STORE };
                code.push(isa::enc_r(op, *guard, *a, *b, 0));
            }
            other => {
                return Err(format!(
                    "line {line}: unknown mnemonic '{other}' (supported: NOP MOV MOV_I ADD ADD_I LDW LOAD STORE HALT)"
                ))
            }
        }
    }

    for (line, idx) in ldw_refs {
        if idx as usize >= literals.len() {
            return Err(format!("line {line}: literal index {idx} out of range"));
        }
    }
    if code.len() > isa::MAX_CODE_WORDS {
        return Err(format!("program too long ({} words)", code.len()));
    }
    if code.is_empty() {
        return Err("empty program".to_string());
    }

    Ok(Assembled { code, literals })
}

fn check_arity(ops: &[Operand], n: usize, line: usize) -> Result<(), String> {
    if ops.len() != n {
        return Err(format!("line {line}: expected {n} operands, got {}", ops.len()));
    }
    Ok(())
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
    fn small_constants_inline() {
        let src = ".const SMALL 5\nADD_I r0, r0, SMALL\nHALT\n";
        let a = assemble(src).unwrap();
        assert_eq!(a.literals, vec![5]); // .const always takes a pool slot
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
    fn roundtrip_stable() {
        let src = ".const BIG 123456\nMOV_I r0, 7\nADD_I r0, r0, BIG\nADD_I r1, r0, -3\nSTORE r0, r1\nLOAD r2, r0\nHALT\n";
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
