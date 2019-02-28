extern crate classfile_parser;
#[macro_use]
extern crate enum_primitive;

extern crate num;
use num::FromPrimitive;

use classfile_parser::constant_info::*;
use classfile_parser::parse_class;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::io::{self, BufRead};
use std::u8;
use std::usize;

mod mangling;
mod jvmtypes;
mod tenyr;

use jvmtypes::*;
use tenyr::*;

const MAX_LOCALS : u16 = 6; // arbitrary limit for now

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

fn handle_op(offset : u16, bytecode : &mut std::slice::Iter<u8>) -> (usize, Option<AddressedOperation>) {
    let mut used = 1;

    let op = match bytecode.next() {
        Some(byte) => JvmOps::from_u8(*byte),
        None => None,
    };
    let op = match op {
        Some(op) => op,
        None => return (0, None),
    };

    let as_16 = |it : &mut std::slice::Iter<u8>| ((*it.next().unwrap() as i16) << 8) | (*it.next().unwrap() as i16);

    use JvmOps::*;
    use JType::*;
    use Operation::*;
    // TODO handle {F,D}cmp{g,l} (implement carefully !)
    let way = match op {
        Ifeq | IfIcmpeq => Some(Comparison::Eq),
        Ifne | IfIcmpne => Some(Comparison::Ne),
        Iflt | IfIcmplt => Some(Comparison::Lt),
        Ifge | IfIcmpge => Some(Comparison::Ge),
        Ifgt | IfIcmpgt => Some(Comparison::Gt),
        Ifle | IfIcmple => Some(Comparison::Le),
        _ => None,
    };

    let index = match op as u8 {
        // TODO figure out how to use bytecode values from enumeration in these ranges
        b @ 0x1a...0x1d => Some(b - 0x1a),  /* Iload{N} */
        b @ 0x2a...0x2d => Some(b - 0x2a),  /* Aload{N} */
        b @ 0x3b...0x3e => Some(b - 0x3b),  /* Istore{N} */
        _ => None,
    };

    let target = match op as u8 {
        0x99...0xa7 => Some(((as_16(bytecode) as i32) + (offset as i32)) as u16), /* Ifeq...Goto */
        _ => None,
    };

    let converted_op = match op {
        b @ Iload0 | b @ Iload1 | b @ Iload2        => { Load { kind: Int, index: index.unwrap() } },
        b @ Aload0 | b @ Aload1                     => { Load { kind: Object, index: index.unwrap() } },
        b @ Arraylength                             => { Length },
        b @ IfIcmpeq | b @ IfIcmple                 => { used += 2; Branch { kind : Int, way: way.unwrap(), target: target.unwrap() } },
        b @ Ifle                                    => { used += 2; Branch { kind : Int, way: way.unwrap(), target: target.unwrap() } },
        b @ Isub                                    => { Subtract { kind: Int } },
        b @ Istore0 | b @ Istore1 | b @ Istore2     => { Store { kind: Int, index: index.unwrap() } },
        b @ Goto                                    => { used += 2; Jump { target: target.unwrap() } },
        b @ Ireturn                                 => { Yield { kind: Int } },

        b @ Nop => { println!("handling {:?} (0x{:02x})", &b, b as u8); Noop },
        b @ _ => panic!("Unsupported byte 0x{:02x}", b as u8),
    };

    return (used, Some(AddressedOperation { address: offset, op: converted_op }));
}

fn parse_bytecode(code : &Vec<u8>) -> (Vec<AddressedOperation>, HashMap<u16,usize>) {
    let mut out = Vec::new();
    let mut map = HashMap::new();

    let mut bytecode = code.iter();
    let mut i : u16 = 0;
    let mut j : usize = 0;
    while let (consumed, Some(p)) = handle_op(i, &mut bytecode) {
        out.push(p);
        map.insert(i, j);
        i += consumed as u16;
        j += 1;
    }

    return (out, map);
}

fn emit_parsed(parsed : &Vec<AddressedOperation>, map : &HashMap<u16,usize>) {
    for &ref op in parsed {
        println!("{:?}", op);
    }
}

struct RegState<'a> {
    regs : &'a Vec<Register>,
    used : &'a [Register],
}

struct XlnState<'s,'l> {
    stack  : &'s RegState<'s>,
    locals : &'l RegState<'l>,
}

fn translate(op : &AddressedOperation, state : &mut XlnState) {
    use JType::*;
    use Operation::*;
    let mut stack = &state.stack.regs;
    let mut si = stack.iter();

    let s0 = si.next().unwrap();
    let s1 = si.next().unwrap();
    let next = si.next().unwrap();
    // TODO take `next` from `stack`

    let to_cmp = |way| match way {
        Comparison::Eq => ("==", false),
        Comparison::Ge => (">=", false),
        Comparison::Gt => (">" , false),
        Comparison::Le => ("<=", false),
        Comparison::Lt => ("<" , false),
        Comparison::Ne => ("==", true ),
    };

    let emit = |loc,what| { println!("L_{}:\t{}", loc, what) };

    let rstack = "O"; // TODO
    let ret = "B"; // TODO
    let xlated = match op.op {
        Load { kind: Int, index } => format!("{} <- {}", *next, Register::from_u8(index).unwrap()),
        Store { kind: Int, index } => format!("{} <- {}", Register::from_u8(index).unwrap(), *s0),
        Branch { kind: Int, way, target } => {
            let cond = "N"; // TODO
            let op = if to_cmp(way).1 { "&~" } else { "&" };
            format!("{} <- {} {} {}\n", cond, *s0, to_cmp(way).0, *s1) +
                &format!("\tP <- @+L_{} {} {} + P", target, op, cond)
        },
        Jump { target } => format!("P <- @+L_{} + P", target),
        Subtract { kind: Int } => { // TODO other binary ops
            let op = "-"; // TODO
            format!("{} <- {} {} {}", *next, *s0, op, *s1)
        }
        Yield { kind: Int } => {
            format!("{} <- {}\n", ret, *s0) +
                &format!("\t{} <- {} + 1\n", rstack, rstack) +
                &format!("\tP <- [{} - 1]", rstack)
        },
        _ => format!("/* unhandled */"),
    };
    emit(op.address, xlated);
}

fn parse(name : &str) -> String {
    use classfile_parser::attribute_info::code_attribute_parser;

    let out = String::new();

    let class = parse_class(name).unwrap();
    let method = &class.methods[1];
    let c = &method.attributes[0].info;
    let code = code_attribute_parser(c).to_result().unwrap();

    let (parsed, map) = parse_bytecode(&code.code);

    // TODO use enumeration for registers
    let regs : HashSet<_> = (0u8..16).collect();
    let special : HashSet<_> = [0u8, 15].iter().cloned().collect();
    let retvals : HashSet<_> = [1u8, 2u8].iter().cloned().collect();
    let largest_local = 3 + code.max_locals;
    if code.max_locals > MAX_LOCALS {
        panic!("Too many locals ({} requested, {} permissible)",
               code.max_locals, MAX_LOCALS);
    }
    let locals : HashSet<_> = (3u8..largest_local as u8).collect();
    let stack : HashSet<_> = regs.clone();
    let stack : HashSet<_> = stack.difference(&special).cloned().collect();
    let stack : HashSet<_> = stack.difference(&retvals).cloned().collect();
    let stack : HashSet<_> = stack.difference(&locals ).cloned().collect();

    let mut stack = stack;

    let regs = { use Register::*; &vec![ M, L, K, J, I, H ] };
    let stack  = &RegState { regs, used: &regs[0..0] };

    let regs = { use Register::*; &vec![ D, E, F, G ] };
    let locals = &RegState { regs, used: &regs[0..0] };

    let mut state = XlnState { stack, locals };

    for op in parsed {
        translate(&op, &mut state);
    }

    return out;
}

fn main() {
    let stdin = io::stdin();
    let args : Vec<_> = env::args().collect();
    if args.len() != 2 {
        panic!("Need a single option (e.g., `--parse`)");
    }
    let func = match &args[1][..] {
        "--parse" => parse,
        _ => panic!("Invalid option `{}`", &args[1]),
    };
    for line in stdin.lock().lines() {
        println!("{}", func(&line.expect("Line read failure")));
    }
}

