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
    let source = preprocess(source)?;
    let tokens = lexer::lex(&source)?;
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

// Slice F deliberately does not grow a general C preprocessor. warp.h is a
// compiler-provided interface, so consume that one include while preserving
// every following source position for diagnostics.
fn preprocess(source: &str) -> Result<String, Diagnostic> {
    let mut output = String::with_capacity(source.len());
    let mut offset = 0;
    for (line_index, line) in source.split_inclusive('\n').enumerate() {
        let line_number = line_index + 1;
        let body = line.strip_suffix('\n').unwrap_or(line);
        let trimmed = body.trim();
        if trimmed.starts_with('#') {
            if trimmed != "#include <warp.h>" && trimmed != "#include \"warp.h\"" {
                let column = body.find('#').unwrap_or(0) + 1;
                return Err(Diagnostic::new(
                    Span::new(offset + column - 1, line_number, column),
                    "only #include <warp.h> is supported in Warp C v0.1.4",
                ));
            }
            output.push_str(&" ".repeat(body.len()));
        } else {
            output.push_str(body);
        }
        if line.ends_with('\n') {
            output.push('\n');
        }
        offset += line.len();
    }
    Ok(output)
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

    #[test]
    fn consumes_builtin_warp_header_include() {
        let result =
            compile("#include <warp.h>\nint main(void) { return WARP_VIDEO_WIDTH - 86; }").unwrap();
        assert!(result.code_words > 0);
    }

    #[test]
    fn rejects_other_preprocessor_directives() {
        let error = compile("#define ANSWER 42\nint main(void) { return ANSWER; }")
            .err()
            .unwrap();
        assert_eq!(error.span.line, 1);
        assert!(error.message.contains("only #include <warp.h>"));
    }

    #[test]
    fn lowers_graphics_intrinsics_to_existing_isa() {
        let result = compile(
            "#include <warp.h>\nint main(void) { unsigned c=warp_argb(255,1,2,3); warp_set_pixel(4,5,c); unsigned *p=warp_framebuffer(); warp_flip(); if (p[5*WARP_VIDEO_WIDTH+4]==c) return 42; return 0; }",
        )
        .unwrap();
        assert!(result.assembly.contains("STORE"));
        assert!(result.assembly.contains("FLIP"));
        assert!(result.assembly.contains("1048576"));
    }

    #[test]
    fn rejects_flip_inside_divergent_control_flow() {
        let error = compile("int main(void) { if (warp_lane_id() < 16) warp_flip(); return 42; }")
            .err()
            .unwrap();
        assert!(error.message.contains("warp_flip()"));
        assert!(error.message.contains("divergent"));
    }
}
