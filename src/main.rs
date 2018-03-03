extern crate regex;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate classfile_parser;

use classfile_parser::constant_info::*;
use regex::Regex;
use std::env;
use std::io::{self, BufRead};
use std::u8;
use std::usize;

fn hexify(s : &str) -> String {
    let mut out = String::new();
    let bytes = s.as_bytes();

    for &b in bytes {
        out.push_str(&format!("{:02x}", &b));
    }

    return out;
}

fn dehexify(s : &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 2);

    for i in 0..(s.len() / 2) {
        let hex = u8::from_str_radix(&s[i*2..i*2+2], 16)
                    .expect("Hex parse failure");
        out.push(hex);
    }

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
        if let Some(m) = RE_TOKEN.find(&name[offset..]) {
            let s = m.as_str();
            let len = s.len();
            offset += len;
            out.push_str(&format!("{}{}", len, s));
        }
        if let Some(m) = RE_NONTOKEN.find(&name[offset..]) {
            if m.as_str().len() > 0 {
                let s = m.as_str();
                let len = s.len();
                offset += len;
                out.push_str(&format!("0{}_{}", len, hexify(&s)));
            } else if offset < name.len() {
                panic!("Unable to progress");
            }
        }
    }

    out.shrink_to_fit();
    return out;
}

fn demangle(name : &str) -> String { // TODO Option<String>
    let mut offset = 0;
    let mut out = String::with_capacity(name.len());
    lazy_static! { static ref NUM : Regex = Regex::new(r"^\d+").unwrap(); }

    if &name[0..1] != "_" {
        panic!("Bad identifier (expected `_`)");
    } else {
        offset += 1;
    }

    let mut is_hex = false;
    while offset < name.len() {
        if &name[offset..offset+1] == "0" {
            offset += 1;
            is_hex = true;
        }
        let m = NUM.find(&name[offset..])
                   .expect("Bad identifier (expected number)");
        let len = usize::from_str_radix(m.as_str(), 10)
            .expect("Hex parse failure");
        offset += m.as_str().len();
        if is_hex {
            if &name[offset..offset+1] != "_" {
                panic!("Bad identifier (expected `_`)");
            }
            offset += 1;
            let nybbles = 2 * len;
            let vec = dehexify(&name[offset..offset+nybbles]);
            let utf8 = String::from_utf8(vec)
                .expect("Expected UTF-8 string");
            out.push_str(&utf8);
            offset += nybbles;
        } else {
            out.push_str(&name[offset..offset+len]);
            offset += len;
        }
        is_hex = false;
    }

    out.shrink_to_fit();
    return out;
}

fn stringify(pool : &Vec<ConstantInfo>, index : u16) -> Result<String,&str> {
    let ci = &pool[(index - 1) as usize];
    let st = stringify;
    let p = pool;

    use classfile_parser::constant_info::ConstantInfo::*;
    return match *ci {
        Utf8(ref x)               => Ok(x.utf8_string.clone()),
        Integer(ref x)            => Ok(format!("{}", x.value)),
        Float(ref x)              => Ok(format!("{}", x.value)),
        Long(ref x)               => Ok(format!("{}", x.value)),
        Double(ref x)             => Ok(format!("{}", x.value)),
        Class(ref x)              => st(&p, x.name_index),
        String(ref x)             => st(&p, x.string_index),
        FieldRef(ref x)           => Ok(st(&p, x.class_index)? + "." + &st(&p, x.name_and_type_index)?),
        MethodRef(ref x)          => Ok(st(&p, x.class_index)? + "." + &st(&p, x.name_and_type_index)?),
        InterfaceMethodRef(ref x) => Ok(st(&p, x.class_index)? + "." + &st(&p, x.name_and_type_index)?),
        NameAndType(ref x)        => Ok(st(&p, x.name_index )? + ":" + &st(&p, x.descriptor_index)?),
        //MethodHandle
        MethodType(ref x)         => st(&p, x.descriptor_index),
        //InvokeDynamic

        _ => Err("Unsupported constant p item type"),
    };
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
        println!("{}", func(&line.expect("Line read failure")));
    }
}

