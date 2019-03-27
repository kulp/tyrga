mod deadcode;
mod exprtree;
mod jvmtypes;
mod mangling;
mod tenyr;

use jvmtypes::*;

use classfile_parser::ClassFile;

#[cfg(test)]
fn parse_class(stem : &str) -> ClassFile {
    let mut name = String::from(concat!(env!("OUT_DIR"), "/"));
    name.push_str(stem);
    classfile_parser::parse_class(&name).unwrap()
}

use classfile_parser::attribute_info::StackMapFrame;
use std::ops::Range;
use std::collections::BTreeMap;
fn derive_ranges<'a, T>(body : &[(usize, &'a T)], table : &[StackMapFrame])
    -> (Vec<Range<usize>>, BTreeMap<usize, &'a T>)
{
    use classfile_parser::attribute_info::StackMapFrame::*;
    let get_delta = |f : &StackMapFrame| match *f {
        SameFrame                           { frame_type }       => u16::from(frame_type),

        SameLocals1StackItemFrame           { frame_type, .. }   => u16::from(frame_type) - 64,

        SameLocals1StackItemFrameExtended   { offset_delta, .. }
            | ChopFrame                     { offset_delta, .. }
            | SameFrameExtended             { offset_delta, .. }
            | AppendFrame                   { offset_delta, .. }
            | FullFrame                     { offset_delta, .. } => offset_delta,
    };
    let deltas : Vec<u16> = table.iter().map(get_delta).collect();

    let before = deltas.iter().take(1);
    let after  = deltas.iter().skip(1);
    let max = body.last().unwrap().0 + 1;
    use std::iter::once;
    let ranges =
        once(0)
            .chain(before.cloned())
            .chain(once(0)).chain(after.map(|&n| n + 1))
            .scan(0, |state, x| { *state += x; Some(usize::from(*state)) })
            .chain(once(max))
            .collect::<Vec<_>>()
            .windows(2)
            .map(|x| x[0]..x[1])
            .filter(|x| x.len() > 0)
            .collect::<Vec<_>>();

    let tree = body.iter().cloned().collect();
    (ranges, tree)
}

use classfile_parser::method_info::MethodInfo;
fn get_ranges_for_method(class : &ClassFile, method : &MethodInfo)
    -> (Vec<Range<usize>>, BTreeMap<usize, Operation>)
{
    use classfile_parser::attribute_info::AttributeInfo;
    use classfile_parser::attribute_info::code_attribute_parser;
    use classfile_parser::attribute_info::stack_map_table_attribute_parser;
    use classfile_parser::constant_info::ConstantInfo::Utf8;

    let get_constant = |n| &class.const_pool[usize::from(n) - 1];
    let name_of = |a : &AttributeInfo|
        match get_constant(a.attribute_name_index) {
            Utf8(u) => u.utf8_string.to_string(),
            _ => panic!("not a name")
        };

    let code = code_attribute_parser(&method.attributes[0].info).unwrap().1;
    let attr = &code.attributes.iter().find(|a| name_of(a) == "StackMapTable");
    let table = match attr {
        Some(attr) => stack_map_table_attribute_parser(&attr.info).unwrap().1.entries,
        _ => Vec::new(),
    };

    use classfile_parser::code_attribute::code_parser;
    let vec = code_parser(&code.code).unwrap().1;
    let refed = vec.iter().map(|(s, x)| (*s, x)).collect::<Vec<_>>();
    let (ranges, map) = derive_ranges(&refed, &table);
    let ops = map.into_iter().map(decode_insn).collect::<BTreeMap<_,_>>();
    (ranges, ops)
}

#[cfg(test)]
fn test_stack_map_table(stem : &str) {
    let class = parse_class(stem);
    for method in &class.methods {
        let (ranges, ops) = get_ranges_for_method(&class, &method);
        let _r = ranges.into_iter().map(|x| ops.range(x)).collect::<Vec<_>>();
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
        test_stack_map_table(name);
    }
}

fn main() {
}

