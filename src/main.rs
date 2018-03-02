extern crate regex;
#[macro_use]
extern crate lazy_static;

use regex::Regex;
use std::env;
use std::io::{self, BufRead};

fn hexify(s : &str) -> String {
    let mut out = String::new();
    let bytes = s.as_bytes();

    for &b in bytes {
        out.push_str(&format!("{:02x}", &b));
    }

    return out;
}

fn dehexify(bytes : &str, len : usize) -> String {
    let mut out = String::new();

    return out;
}

fn mangle(name : &str) -> String {
    let mut offset = 0;
    let mut out = String::with_capacity(2 * name.len()); // heuristic
    lazy_static! {
        static ref RE_TOKEN    : Regex = Regex::new(r"^(?i)[a-z_]\w*").unwrap();
        static ref RE_NONTOKEN : Regex = Regex::new(r"^[0-9]*\W*").unwrap();
    }

    out.push('_');

    while offset < name.len() {
        match RE_TOKEN.find(&name[offset..]) {
            Some(m) => {
                let s = m.as_str();
                let len = s.len();
                offset += len;
                out.push_str(&format!("{}{}", len, s));
            },
            None => {},
        };
        match RE_NONTOKEN.find(&name[offset..]) {
            Some(m) if m.as_str().len() > 0 => {
                let s = m.as_str();
                let len = s.len();
                offset += len;
                out.push_str(&format!("0{}_{}", len, hexify(&s)));
            },
            _ => {},
        };
    }

    out.shrink_to_fit();
    return out;
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

