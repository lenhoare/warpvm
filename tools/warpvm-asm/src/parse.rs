// .wva parsing: labels, directives, guarded instructions, operands.

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    VReg(u32),
    SReg(u32),
    Pred(u32),
    Imm(i64),
    Name(String),
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Label {
        line: usize,
        name: String,
    },
    Instr {
        line: usize,
        guard: u32,
        mnemonic: String,
        operands: Vec<Operand>,
    },
    Directive {
        line: usize,
        name: String,
        args: Vec<String>,
    },
}

pub struct Program {
    pub stmts: Vec<Stmt>,
}

pub fn parse(src: &str) -> Result<Program, String> {
    let mut stmts = Vec::new();

    for (idx, raw) in src.lines().enumerate() {
        let line = idx + 1;
        let no_comment = match raw.find(';') {
            Some(i) => &raw[..i],
            None => raw,
        };
        let trimmed = no_comment.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut tokens = trimmed.split_whitespace();
        let first = tokens.next().unwrap();

        // Optional leading label.
        let (label, mnemonic_tok) = if let Some(name) = first.strip_suffix(':') {
            if name.is_empty() || !is_ident(name) {
                return Err(format!("line {line}: bad label '{first}'"));
            }
            (Some(name.to_string()), tokens.next())
        } else {
            (None, Some(first))
        };
        if let Some(name) = label {
            stmts.push(Stmt::Label { line, name });
        }

        let Some(mut tok) = mnemonic_tok else { continue };

        // Optional guard prefix.
        let guard = if let Some(g) = tok.strip_prefix('@') {
            let parsed = parse_guard(g, line)?;
            tok = tokens
                .next()
                .ok_or_else(|| format!("line {line}: guard with no instruction"))?;
            parsed
        } else {
            0
        };

        if let Some(dir) = tok.strip_prefix('.') {
            if dir.is_empty() {
                return Err(format!("line {line}: empty directive"));
            }
            let args = tokens.map(str::to_string).collect();
            stmts.push(Stmt::Directive {
                line,
                name: dir.to_string(),
                args,
            });
            continue;
        }

        // Operands: everything after the mnemonic, comma-separated.
        let rest: Vec<&str> = tokens.collect();
        let joined = rest.join(" ");
        let mut operands = Vec::new();
        if !joined.trim().is_empty() {
            for part in joined.split(',') {
                let p = part.trim();
                if p.is_empty() {
                    return Err(format!("line {line}: empty operand"));
                }
                operands.push(parse_operand(p, line)?);
            }
        }

        stmts.push(Stmt::Instr {
            line,
            guard,
            mnemonic: tok.to_ascii_uppercase(),
            operands,
        });
    }

    Ok(Program { stmts })
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn is_identifier(s: &str) -> bool {
    is_ident(s)
}

fn parse_guard(g: &str, line: usize) -> Result<u32, String> {
    let (inv, rest) = if let Some(r) = g.strip_prefix('!') {
        (true, r)
    } else {
        (false, g)
    };
    let pred = parse_pred(rest, line)?;
    Ok(if inv { 5 + pred } else { 1 + pred })
}

fn parse_pred(s: &str, line: usize) -> Result<u32, String> {
    let n = s
        .strip_prefix('p')
        .and_then(|d| d.parse::<u32>().ok())
        .filter(|&n| n < 4)
        .ok_or_else(|| format!("line {line}: bad predicate '{s}' (want p0-p3)"))?;
    Ok(n)
}

pub fn parse_operand(s: &str, line: usize) -> Result<Operand, String> {
    if let Some(num) = s.strip_prefix('#') {
        return Ok(Operand::Imm(parse_number(num, line)?));
    }
    if let Some(cls) = s.chars().next() {
        let digits = &s[1..];
        match cls {
            'r' if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) => {
                let n: u32 = digits.parse().map_err(|_| format!("line {line}: bad register '{s}'"))?;
                if n >= 16 {
                    return Err(format!("line {line}: vector register out of range '{s}'"));
                }
                return Ok(Operand::VReg(n));
            }
            's' if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) => {
                let n: u32 = digits.parse().map_err(|_| format!("line {line}: bad register '{s}'"))?;
                if n >= 8 {
                    return Err(format!("line {line}: scalar register out of range '{s}'"));
                }
                return Ok(Operand::SReg(n));
            }
            'p' if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) => {
                return Ok(Operand::Pred(parse_pred(s, line)?));
            }
            _ => {}
        }
    }
    let first = s.chars().next().unwrap_or(' ');
    if first.is_ascii_digit() || first == '-' {
        return Ok(Operand::Imm(parse_number(s, line)?));
    }
    if is_ident(s) {
        return Ok(Operand::Name(s.to_string()));
    }
    Err(format!("line {line}: bad operand '{s}'"))
}

pub fn parse_number(s: &str, line: usize) -> Result<i64, String> {
    let (neg, rest) = if let Some(r) = s.strip_prefix('-') {
        (true, r)
    } else {
        (false, s)
    };
    let value = if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16)
    } else if let Some(bin) = rest.strip_prefix("0b").or_else(|| rest.strip_prefix("0B")) {
        i64::from_str_radix(bin, 2)
    } else {
        rest.parse::<i64>()
    };
    let value = value.map_err(|_| format!("line {line}: bad number '{s}'"))?;
    Ok(if neg { -value } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hello() {
        let src = "label:\n    @p1 MOV_I r0, 7\n    HALT ; done\n";
        let prog = parse(src).unwrap();
        assert!(matches!(&prog.stmts[0], Stmt::Label { name, .. } if name == "label"));
        match &prog.stmts[1] {
            Stmt::Instr {
                guard,
                mnemonic,
                operands,
                ..
            } => {
                assert_eq!(*guard, 2);
                assert_eq!(mnemonic, "MOV_I");
                assert_eq!(*operands, vec![Operand::VReg(0), Operand::Imm(7)]);
            }
            other => panic!("expected instr, got {other:?}"),
        }
        assert!(matches!(&prog.stmts[2], Stmt::Instr { mnemonic, .. } if mnemonic == "HALT"));
    }

    #[test]
    fn parses_directives_and_numbers() {
        let src = ".const BIG 0xDEADBEEF\n.word 0b101\n";
        let prog = parse(src).unwrap();
        match &prog.stmts[0] {
            Stmt::Directive { name, args, .. } => {
                assert_eq!(name, "const");
                assert_eq!(args, &["BIG", "0xDEADBEEF"]);
            }
            other => panic!("expected directive, got {other:?}"),
        }
        assert_eq!(parse_number("0b101", 1).unwrap(), 5);
        assert_eq!(parse_number("-42", 1).unwrap(), -42);
        assert_eq!(parse_number("0xff", 1).unwrap(), 255);
    }

    #[test]
    fn rejects_bad_operands() {
        assert!(parse("MOV r16, r0, r0").is_err());
        assert!(parse("MOV r0, r0, @").is_err());
        assert!(parse("@p9 MOV r0, r1").is_err());
    }
}
