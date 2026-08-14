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
    let source = inject_platform_library(preprocess(source)?);
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

const WARP_MEMCPY_SOURCE: &str = r#"
void warp_memcpy(uniform unsigned *dst, uniform unsigned *src, uniform unsigned words)
{
    for (uniform unsigned base = 0; base < words; base += 32) {
        unsigned i = base + WARP;
        if (i < words)
            dst[i] = src[i];
    }
}
"#;

const WARP_MEMSET_SOURCE: &str = r#"
void warp_memset(uniform unsigned *dst, unsigned value, uniform unsigned words)
{
    for (uniform unsigned base = 0; base < words; base += 32) {
        unsigned i = base + WARP;
        if (i < words)
            dst[i] = value;
    }
}
"#;

fn inject_platform_library(mut source: String) -> String {
    let needs_memcpy = contains_identifier(&source, "warp_memcpy");
    let needs_memset = contains_identifier(&source, "warp_memset");
    if needs_memcpy {
        source.push_str(WARP_MEMCPY_SOURCE);
    }
    if needs_memset {
        source.push_str(WARP_MEMSET_SOURCE);
    }
    source
}

fn contains_identifier(source: &str, target: &str) -> bool {
    source
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .any(|word| word == target)
}

// The platform header deliberately does not grow a general C preprocessor. warp.h is a
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
                    "only #include <warp.h> is supported in Warp C v0.1.5",
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
            "#include <warp.h>\nint main(void) { uniform unsigned c=warp_argb(255,1,2,3); warp_set_pixel(4,5,c); uniform unsigned *p=warp_framebuffer(); warp_flip(); if (p[5*WARP_VIDEO_WIDTH+4]==c) return 42; return 0; }",
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

    #[test]
    fn lowers_vm_wide_mailbox_intrinsics() {
        let result = compile(
            "unsigned payload; unsigned metadata; int main(void) { warp_send(1, 7, 42); return warp_try_recv(&payload, &metadata); }",
        )
        .unwrap();
        assert!(result.assembly.contains("SEND"));
        assert!(result.assembly.contains("TRY_RECV"));
        assert!(result.assembly.contains("@p0 STORE"));
    }

    #[test]
    fn rejects_send_inside_divergent_control_flow() {
        let error =
            compile("int main(void) { if (warp_lane_id() == 0) warp_send(1, 1, 42); return 0; }")
                .err()
                .unwrap();
        assert!(error.message.contains("warp_send()"));
        assert!(error.message.contains("divergent"));
    }

    #[test]
    fn warp_is_a_divergent_non_lvalue() {
        let result =
            compile("int main(void) { int x=WARP; if (WARP<16) x=42; return 42; }").unwrap();
        assert!(result.uniformity_dump.contains("Divergent"));
        assert!(result.assembly.contains("BALLOT"));

        for source in [
            "int main(void) { WARP=3; return 0; }",
            "int main(void) { ++WARP; return 0; }",
            "int main(void) { int *p=&WARP; return 0; }",
        ] {
            let error = compile(source).err().unwrap();
            assert!(error.message.contains("lvalue"));
        }
    }

    #[test]
    fn warp_cannot_be_shadowed_or_used_as_a_constant() {
        for source in [
            "int WARP; int main(void) { return 0; }",
            "int f(int WARP) { return 0; } int main(void) { return 0; }",
            "int main(void) { int WARP=1; return 0; }",
        ] {
            let error = compile(source).err().unwrap();
            assert!(error.message.contains("reserved predefined"));
        }
        let error = compile("int a[WARP]; int main(void) { return 0; }")
            .err()
            .unwrap();
        assert!(error.message.contains("constant integer expression"));
    }

    #[test]
    fn collective_constraints_are_diagnosed() {
        let result = compile("int main(void) { return warp_shuffle_xor(WARP, 1 + 1); }").unwrap();
        assert!(result.assembly.contains("SHUFFLE_XOR"));
        assert!(result.assembly.contains(", 2"));

        let error =
            compile("int main(void) { int x=0; if (WARP<16) x=warp_reduce_add(WARP); return x; }")
                .err()
                .unwrap();
        assert!(error.message.contains("divergent control flow"));

        let error = compile("int main(void) { return warp_broadcast(WARP, WARP); }")
            .err()
            .unwrap();
        assert!(error.message.contains("lane must be uniform"));

        let error = compile("int main(void) { int mask=1; return warp_shuffle_xor(WARP, mask); }")
            .err()
            .unwrap();
        assert!(error.message.contains("constant from 0 to 31"));
    }

    #[test]
    fn shuffle_uniformity_is_visible_in_dump_and_branch_lowering() {
        let result = compile(
            "int main(void) { int value=WARP*7; uniform int fixed=warp_shuffle(value,31); int self=warp_shuffle(value,WARP); if (fixed==217 && warp_vm_id()>=0) return 42; return self-self; }",
        )
        .unwrap();
        let fixed = result
            .uniformity_dump
            .lines()
            .find(|line| line.contains(" fixed:"))
            .unwrap();
        let self_value = result
            .uniformity_dump
            .lines()
            .find(|line| line.contains(" self:"))
            .unwrap();
        assert!(fixed.ends_with("Uniform"), "{fixed}");
        assert!(self_value.ends_with("Divergent"), "{self_value}");
        assert!(result.assembly.contains("SHUFFLE"));
        assert!(result.assembly.contains("JMP_IF_ANY"));
        assert!(!result.assembly.contains("BALLOT p3"));
    }
}
