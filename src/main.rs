mod deadcode;
mod exprtree;
mod jvmtypes;
mod mangling;
mod tenyr;

use jvmtypes::*;

use classfile_parser::code_attribute::Instruction;

fn parse_method(mut code : Vec<(usize, Instruction)>) -> Vec<(usize, Operation)> {
    code.drain(..).map(decode_insn).collect()
}

#[cfg(test)]
use classfile_parser::ClassFile;
#[cfg(test)]
fn parse_class(stem : &str) -> ClassFile {
    let mut name = String::from(concat!(env!("OUT_DIR"), "/"));
    name.push_str(stem);
    classfile_parser::parse_class(&name).unwrap()
}

#[cfg(test)]
fn test_parse_methods(stem : &str) {
    use classfile_parser::attribute_info::code_attribute_parser;
    use classfile_parser::code_attribute::code_parser;
    for method in &parse_class(stem).methods {
        let c = &method.attributes[0].info;
        let (_, code) = code_attribute_parser(c).unwrap();

        let vec = parse_method(code_parser(&code.code).unwrap().1);
        assert!(vec.len() > 0);
        use std::collections::BTreeMap;
        let _bt : BTreeMap<_,_> = vec.into_iter().collect();
    }
}

use classfile_parser::attribute_info::StackMapFrame;
fn derive_slices<'a, T>(mut body : &'a [T], table : &[StackMapFrame]) -> Vec<&'a [T]> {
    use classfile_parser::attribute_info::StackMapFrame::*;
    let get_delta = |f : &StackMapFrame| match *f {
        SameFrame                           { frame_type }       => frame_type as u16,

        SameLocals1StackItemFrame           { frame_type, .. }   => frame_type as u16 - 64,

        SameLocals1StackItemFrameExtended   { offset_delta, .. }
            | ChopFrame                     { offset_delta, .. }
            | SameFrameExtended             { offset_delta, .. }
            | AppendFrame                   { offset_delta, .. }
            | FullFrame                     { offset_delta, .. } => offset_delta,
    };
    let deltas : Vec<u16> = table.iter().map(get_delta).collect();

    let splitter = |n| {
        let (first, b) = body.split_at(n as usize);
        body = b;
        first
    };
    let before = deltas.iter().take(1);
    let after  = deltas.iter().skip(1);
    let mut slices : Vec<&[T]> =
        before
            .map(|&n| n).chain(after.map(|&n| n + 1))
            .map(splitter)
            .collect();
    slices.push(body);
    slices
}

#[cfg(test)]
fn test_stack_map_table(stem : &str) {
    use classfile_parser::attribute_info::AttributeInfo;
    use classfile_parser::attribute_info::code_attribute_parser;
    use classfile_parser::attribute_info::stack_map_table_attribute_parser;
    use classfile_parser::constant_info::ConstantInfo::Utf8;

    let class = parse_class(stem);
    let get_constant = |n| &class.const_pool[n as usize - 1];
    let name_of = |a : &AttributeInfo|
        match get_constant(a.attribute_name_index) {
            Utf8(u) => u.utf8_string.to_string(),
            _ => panic!("not a name")
        };

    let method = &class.methods.last().unwrap();
    let code = code_attribute_parser(&method.attributes[0].info).unwrap().1;
    let attr = &code.attributes.iter().find(|a| name_of(a) == "StackMapTable").unwrap();
    let table = stack_map_table_attribute_parser(&attr.info).unwrap().1.entries;

    let slices = derive_slices(&code.code, &table);
    assert_eq!(code.code.iter().count(), slices.into_iter().flatten().count());
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
        test_stack_map_table(name);
    }
}

fn main() {
}

