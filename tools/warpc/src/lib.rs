pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod sema;
pub mod span;

use span::{Diagnostic, Span};

pub struct Compilation {
    pub assembly: String,
    pub ast_dump: String,
    pub uniformity_dump: String,
    pub wvm: Vec<u8>,
    pub code_words: usize,
    pub literal_words: usize,
}

pub fn compile(source: &str) -> Result<Compilation, Diagnostic> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens)?;
    let ast_dump = format!("{ast:#?}");
    let typed = sema::analyze(ast)?;
    let uniformity_dump = sema::dump_uniformity(&typed);
    let assembly = codegen::generate(&typed)?;
    let assembled = wvmasm::assemble(&assembly).map_err(|message| {
        Diagnostic::new(
            Span::new(0, 1, 1),
            format!("generated assembly rejected: {message}"),
        )
    })?;
    let wvm = wvmasm::wvm::encode_file(&assembled.code, &assembled.literals);
    Ok(Compilation {
        assembly,
        ast_dump,
        uniformity_dump,
        wvm,
        code_words: assembled.code.len(),
        literal_words: assembled.literals.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_to_round_trippable_wvm() {
        let result = compile("int main(void) { int x = 40; x += 2; return x; }").unwrap();
        let (code, literals) = wvmasm::wvm::parse_file(&result.wvm).unwrap();
        assert_eq!(code.len(), result.code_words);
        assert_eq!(literals.len(), result.literal_words);
        assert!(result.assembly.contains("HALT"));
    }
}
