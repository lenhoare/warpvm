// .wvm binary file format (docs/isa.md §10).

use crate::isa;

pub const MAGIC: u32 = 0x304D_5657; // 'W','V','M','0'
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 32;

pub fn encode_file(code: &[u32], literals: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_BYTES + 4 * (code.len() + literals.len()));
    push32(&mut out, MAGIC);
    push32(&mut out, VERSION);
    push32(&mut out, 0); // flags
    push32(&mut out, code.len() as u32);
    push32(&mut out, literals.len() as u32);
    push32(&mut out, 0); // entry
    push32(&mut out, 0); // reserved
    push32(&mut out, 0); // reserved
    for &w in code.iter().chain(literals.iter()) {
        push32(&mut out, w);
    }
    out
}

pub fn parse_file(bytes: &[u8]) -> Result<(Vec<u32>, Vec<u32>), String> {
    if bytes.len() < HEADER_BYTES {
        return Err("truncated header".into());
    }
    let rd = |off: usize| -> u32 {
        u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
    };
    if rd(0) != MAGIC {
        return Err("bad magic (not a .wvm file)".into());
    }
    if rd(4) != VERSION {
        return Err("unsupported .wvm version".into());
    }
    let code_len = rd(12) as usize;
    let lit_len = rd(16) as usize;
    if code_len == 0 || code_len > isa::MAX_CODE_WORDS {
        return Err("bad code_len".into());
    }
    if lit_len > isa::MAX_LITERALS {
        return Err("bad literals_len".into());
    }
    let need = HEADER_BYTES + 4 * (code_len + lit_len);
    if bytes.len() < need {
        return Err("truncated body".into());
    }
    let mut code = Vec::with_capacity(code_len);
    let mut literals = Vec::with_capacity(lit_len);
    for i in 0..code_len {
        code.push(rd(HEADER_BYTES + 4 * i));
    }
    for i in 0..lit_len {
        literals.push(rd(HEADER_BYTES + 4 * (code_len + i)));
    }
    Ok((code, literals))
}

fn push32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_roundtrip() {
        let code = vec![1u32, 2, 3];
        let lits = vec![0xDEAD_BEEF];
        let bytes = encode_file(&code, &lits);
        let (c, l) = parse_file(&bytes).unwrap();
        assert_eq!(c, code);
        assert_eq!(l, lits);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = encode_file(&[1], &[]);
        bytes[0] = 0;
        assert!(parse_file(&bytes).is_err());
    }
}
