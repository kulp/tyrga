use std::env;
use std::io::{self, BufRead};

fn hexify(bytes : &str) -> String {
    let mut out = String::new();

    return out;
}

fn dehexify(bytes : &str, len : usize) -> String {
    let mut out = String::new();

    return out;
}

fn mangle(name : &str) -> String {
    return String::from(name)
}

fn demangle(name : &str) -> String {
    return String::from(name)
}

fn main() {
    let stdin = io::stdin();
    let args : Vec<_> = env::args().collect();
    if args.len() != 2 {
        panic!("Need `--mangle` or `--demangle` option");
    }
    let func = match &args[1][..] {
        "--mangle" => mangle,
        "--demangle" => demangle,
        _ => panic!("Invalid option `{}`", &args[1]),
    };
    for line in stdin.lock().lines() {
        println!("{}", func(&line.unwrap()));
    }
}

