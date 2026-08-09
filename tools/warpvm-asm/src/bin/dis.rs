// warpvm-dis: disassemble a .wvm file to stdout.

use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 1 || args[0] == "-h" || args[0] == "--help" {
        eprintln!("usage: warpvm-dis <input.wvm>");
        return if args.len() == 1 {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    }

    let input = &args[0];
    let bytes = match fs::read(input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("warpvm-dis: {input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (code, literals) = match wvmasm::wvm::parse_file(&bytes) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("warpvm-dis: {input}: {e}");
            return ExitCode::FAILURE;
        }
    };

    print!("{}", wvmasm::disasm::disassemble(&code, &literals));
    ExitCode::SUCCESS
}
