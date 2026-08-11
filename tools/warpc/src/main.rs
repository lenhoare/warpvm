use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!(
        "usage: warpc <input.wc> -o <output.wvm> [--emit-asm] [--emit-ast] [--dump-uniformity]"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut emit_asm = false;
    let mut emit_ast = false;
    let mut dump_uniformity = false;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => {
                usage();
                return ExitCode::SUCCESS;
            }
            "-o" | "--output" => {
                index += 1;
                if index >= args.len() {
                    eprintln!("warpc: -o requires an argument");
                    return ExitCode::FAILURE;
                }
                output = Some(args[index].clone());
            }
            "--emit-asm" => emit_asm = true,
            "--emit-ast" => emit_ast = true,
            "--dump-uniformity" => dump_uniformity = true,
            option if option.starts_with('-') => {
                eprintln!("warpc: unknown option '{option}'");
                usage();
                return ExitCode::FAILURE;
            }
            path => {
                if input.is_some() {
                    eprintln!("warpc: multiple input files");
                    return ExitCode::FAILURE;
                }
                input = Some(path.to_string());
            }
        }
        index += 1;
    }

    let Some(input) = input else {
        usage();
        return ExitCode::FAILURE;
    };
    if output.is_none() && !emit_asm && !emit_ast && !dump_uniformity {
        eprintln!("warpc: -o is required unless a dump mode is selected");
        return ExitCode::FAILURE;
    }
    let source = match fs::read_to_string(&input) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("warpc: {input}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let compilation = match warpc::compile(&source) {
        Ok(compilation) => compilation,
        Err(error) => {
            eprintln!("{}", error.render(&input));
            return ExitCode::FAILURE;
        }
    };

    if emit_ast {
        println!("{}", compilation.ast_dump);
    }
    if dump_uniformity {
        print!("{}", compilation.uniformity_dump);
    }
    if emit_asm {
        print!("{}", compilation.assembly);
    }
    if let Some(output) = output {
        if let Err(error) = fs::write(&output, &compilation.wvm) {
            eprintln!("warpc: {output}: {error}");
            return ExitCode::FAILURE;
        }
        println!(
            "{}: {} words, {} literals -> {}",
            input, compilation.code_words, compilation.literal_words, output
        );
    }
    ExitCode::SUCCESS
}
