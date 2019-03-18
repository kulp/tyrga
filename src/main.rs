mod deadcode;
mod exprtree;
mod jvmtypes;
mod mangling;
mod tenyr;

use jvmtypes::*;

use classfile_parser::ClassFile;
use classfile_parser::attribute_info::code_attribute_parser;
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
    for method in &parse_class(stem).methods {
        let c = &method.attributes[0].info;
        let (_, code) = code_attribute_parser(c).unwrap();

        let vec = parse_method(code_parser(&code.code).unwrap().1);
        assert!(vec.len() > 0);
    }
}

#[cfg(test)]
fn test_stack_map_table(stem : &str) {
    use classfile_parser::attribute_info::stack_map_table_attribute_parser;
    use classfile_parser::constant_info::ConstantInfo::Utf8;
    use classfile_parser::attribute_info::AttributeInfo;

    let class = parse_class(stem);
    let get_constant = |n| &class.const_pool[n as usize - 1];
    let name_of = |a : &AttributeInfo|
        match get_constant(a.attribute_name_index) {
            Utf8(u) => u.utf8_string.to_string(),
            _ => panic!("not a name")
        };

    let method = &class.methods.last().unwrap();
    let code = code_attribute_parser(&method.attributes[0].info).unwrap().1;
    let attr = &code.attributes
                    .iter()
                    .find(|a| name_of(a) == "StackMapTable")
                    .unwrap();
    let map = stack_map_table_attribute_parser(&attr.info);

    use classfile_parser::attribute_info::StackMapFrame::*;
    use classfile_parser::attribute_info::StackMapFrame;
    let get_delta = |f : &StackMapFrame| match *f {
        SameFrame { frame_type } => frame_type as u16,
        SameLocals1StackItemFrame { frame_type, .. } => frame_type as u16 - 64,
        SameLocals1StackItemFrameExtended { offset_delta, .. }
            | ChopFrame { offset_delta, .. }
            | SameFrameExtended { offset_delta, .. }
            | AppendFrame { offset_delta, .. }
            | FullFrame { offset_delta, .. }
            => offset_delta,
    };
    let _deltas : Vec<u16> = map.unwrap().1.entries.iter().map(get_delta).collect();
    // for now, getting here without panicking is enough
}

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
        test_stack_map_table(name);
    }
}

fn main() {
}

