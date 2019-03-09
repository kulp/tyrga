mod deadcode;
mod exprtree;
mod jvmtypes;
mod mangling;
mod tenyr;

use jvmtypes::*;

#[test]
fn test_parse_first_method() {
    use classfile_parser::parse_class;
    use classfile_parser::attribute_info::code_attribute_parser;

    let name = "test/Nest";
    let class = parse_class(name).unwrap();
    let method = &class.methods[1];
    let c = &method.attributes[0].info;
    let code = code_attribute_parser(c).to_result().unwrap().code;

    let mut vec = Vec::new();
    let mut addr = 0;
    while code[addr..].len() > 0 {
        let (op, consumed) = decode_op(&code[addr..], addr as u16);
        assert!(consumed > 0);
        vec.push(op.expect("need op"));
        addr += consumed;
    }

    assert_eq!(vec.len(), 41);
}

fn main() {
}

