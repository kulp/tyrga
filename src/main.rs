mod deadcode;
mod exprtree;
mod jvmtypes;
mod mangling;
mod tenyr;

use jvmtypes::*;

use classfile_parser::code_attribute::{code_parser, Instruction};

fn parse_method(mut code : Vec<(usize, Instruction)>) -> Vec<Operation> {
    code.drain(..).map(decode_insn).collect()
}

#[cfg(test)]
fn test_parse_methods(stem : &str) {
    use classfile_parser::parse_class;
    use classfile_parser::attribute_info::code_attribute_parser;

    let mut name = String::from(concat!(env!("OUT_DIR"), "/"));
    name.push_str(stem);
    let name = &name;
    let class = parse_class(name).unwrap();
    for method in &class.methods {
        let c = &method.attributes[0].info;
        let (_, code) = code_attribute_parser(c).unwrap();

        let vec = parse_method(code_parser(&code.code).unwrap().1);
        assert!(vec.len() > 0);
    }
}

#[test] fn test_parse_except() { test_parse_methods("Except") }
#[test] fn test_parse_expr()   { test_parse_methods("Expr") }
#[test] fn test_parse_gcd()    { test_parse_methods("GCD") }
#[test] fn test_parse_nest()   { test_parse_methods("Nest") }
#[test] fn test_parse_sieve()  { test_parse_methods("Sieve") }
#[test] fn test_parse_switch() { test_parse_methods("Switch") }
#[test] fn test_parse_tiny()   { test_parse_methods("Tiny") }

fn main() {
}

