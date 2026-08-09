// warpvm-as: assemble a .wva file into .wvm bytecode.

use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!("usage: warpvm-as <input.wva> -o <output.wvm>");
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                usage();
                return ExitCode::SUCCESS;
            }
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("warpvm-as: -o requires an argument");
                    return ExitCode::FAILURE;
                }
                output = Some(args[i].clone());
            }
            other if other.starts_with('-') => {
                eprintln!("warpvm-as: unknown option '{other}'");
                usage();
                return ExitCode::FAILURE;
            }
            other => {
                if input.is_some() {
                    eprintln!("warpvm-as: multiple input files");
                    return ExitCode::FAILURE;
                }
                input = Some(other.to_string());
            }
        }
        i += 1;
    }

    let (Some(input), Some(output)) = (input, output) else {
        usage();
        return ExitCode::FAILURE;
    };

    let src = match fs::read_to_string(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("warpvm-as: {input}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let assembled = match wvmasm::assemble(&src) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("warpvm-as: {input}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let bytes = wvmasm::wvm::encode_file(&assembled.code, &assembled.literals);
    if let Err(e) = fs::write(&output, &bytes) {
        eprintln!("warpvm-as: {output}: {e}");
        return ExitCode::FAILURE;
    }

    println!(
        "{}: {} words, {} literals -> {}",
        input,
        assembled.code.len(),
        assembled.literals.len(),
        output
    );
    ExitCode::SUCCESS
}
