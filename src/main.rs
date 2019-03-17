mod deadcode;
mod exprtree;
mod jvmtypes;
mod mangling;
mod tenyr;

use jvmtypes::*;

use classfile_parser::ClassFile;
use classfile_parser::code_attribute::{code_parser, Instruction};

fn parse_method(mut code : Vec<(usize, Instruction)>) -> Vec<Operation> {
    code.drain(..).map(decode_insn).collect()
}

#[cfg(test)]
fn parse_class(stem : &str) -> ClassFile {
    let mut name = String::from(concat!(env!("OUT_DIR"), "/"));
    name.push_str(stem);
    let name = &name;
    classfile_parser::parse_class(name).unwrap()
}

#[cfg(test)]
fn test_parse_methods(stem : &str) {
    use classfile_parser::attribute_info::code_attribute_parser;

    for method in &parse_class(stem).methods {
        let c = &method.attributes[0].info;
        let (_, code) = code_attribute_parser(c).unwrap();

        let vec = parse_method(code_parser(&code.code).unwrap().1);
        assert!(vec.len() > 0);
    }
}

#[cfg(test)]
const CLASS_LIST : &[&'static str] = &[
    "Except",
    "Expr",
    "GCD",
    "Nest",
    "Sieve",
    "Switch",
    "Tiny",
];

#[test]
fn test_parse_classes()
{
    for name in CLASS_LIST {
        test_parse_methods(name);
    }
}

fn main() {
}

