#![allow(dead_code)]

use classfile_parser::constant_info::*;

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

