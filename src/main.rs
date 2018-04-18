extern crate regex;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate classfile_parser;
#[macro_use]
extern crate enum_primitive;

extern crate num;
use num::FromPrimitive;

use classfile_parser::ClassFile;
use classfile_parser::constant_info::*;
use classfile_parser::parse_class;
use regex::Regex;
use std::collections::HashMap;
use std::env;
use std::io::{self, BufRead};
use std::u8;
use std::usize;

enum_from_primitive! {
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
enum JvmOps {
    Nop             = 0x00,
    AconstNull      = 0x01,
    IconstM1        = 0x02,
    Iconst0         = 0x03,
    Iconst1         = 0x04,
    Iconst2         = 0x05,
    Iconst3         = 0x06,
    Iconst4         = 0x07,
    Iconst5         = 0x08,
    Lconst0         = 0x09,
    Lconst1         = 0x0a,
    Fconst0         = 0x0b,
    Fconst1         = 0x0c,
    Fconst2         = 0x0d,
    Dconst0         = 0x0e,
    Dconst1         = 0x0f,
    Bipush          = 0x10,
    Sipush          = 0x11,
    Ldc             = 0x12,
    LdcW            = 0x13,
    Ldc2W           = 0x14,
    Iload           = 0x15,
    Lload           = 0x16,
    Fload           = 0x17,
    Dload           = 0x18,
    Aload           = 0x19,
    Iload0          = 0x1a,
    Iload1          = 0x1b,
    Iload2          = 0x1c,
    Iload3          = 0x1d,
    Lload0          = 0x1e,
    Lload1          = 0x1f,
    Lload2          = 0x20,
    Lload3          = 0x21,
    Fload0          = 0x22,
    Fload1          = 0x23,
    Fload2          = 0x24,
    Fload3          = 0x25,
    Dload0          = 0x26,
    Dload1          = 0x27,
    Dload2          = 0x28,
    Dload3          = 0x29,
    Aload0          = 0x2a,
    Aload1          = 0x2b,
    Aload2          = 0x2c,
    Aload3          = 0x2d,
    Iaload          = 0x2e,
    Laload          = 0x2f,
    Faload          = 0x30,
    Daload          = 0x31,
    Aaload          = 0x32,
    Baload          = 0x33,
    Caload          = 0x34,
    Saload          = 0x35,
    Istore          = 0x36,
    Lstore          = 0x37,
    Fstore          = 0x38,
    Dstore          = 0x39,
    Astore          = 0x3a,
    Istore0         = 0x3b,
    Istore1         = 0x3c,
    Istore2         = 0x3d,
    Istore3         = 0x3e,
    Lstore0         = 0x3f,
    Lstore1         = 0x40,
    Lstore2         = 0x41,
    Lstore3         = 0x42,
    Fstore0         = 0x43,
    Fstore1         = 0x44,
    Fstore2         = 0x45,
    Fstore3         = 0x46,
    Dstore0         = 0x47,
    Dstore1         = 0x48,
    Dstore2         = 0x49,
    Dstore3         = 0x4a,
    Astore0         = 0x4b,
    Astore1         = 0x4c,
    Astore2         = 0x4d,
    Astore3         = 0x4e,
    Iastore         = 0x4f,
    Lastore         = 0x50,
    Fastore         = 0x51,
    Dastore         = 0x52,
    Aastore         = 0x53,
    Bastore         = 0x54,
    Castore         = 0x55,
    Sastore         = 0x56,
    Pop             = 0x57,
    Pop2            = 0x58,
    Dup             = 0x59,
    DupX1           = 0x5a,
    DupX2           = 0x5b,
    Dup2            = 0x5c,
    Dup2X1          = 0x5d,
    Dup2X2          = 0x5e,
    Swap            = 0x5f,
    Iadd            = 0x60,
    Ladd            = 0x61,
    Fadd            = 0x62,
    Dadd            = 0x63,
    Isub            = 0x64,
    Lsub            = 0x65,
    Fsub            = 0x66,
    Dsub            = 0x67,
    Imul            = 0x68,
    Lmul            = 0x69,
    Fmul            = 0x6a,
    Dmul            = 0x6b,
    Idiv            = 0x6c,
    Ldiv            = 0x6d,
    Fdiv            = 0x6e,
    Ddiv            = 0x6f,
    Irem            = 0x70,
    Lrem            = 0x71,
    Frem            = 0x72,
    Drem            = 0x73,
    Ineg            = 0x74,
    Lneg            = 0x75,
    Fneg            = 0x76,
    Dneg            = 0x77,
    Ishl            = 0x78,
    Lshl            = 0x79,
    Ishr            = 0x7a,
    Lshr            = 0x7b,
    Iushr           = 0x7c,
    Lushr           = 0x7d,
    Iand            = 0x7e,
    Land            = 0x7f,
    Ior             = 0x80,
    Lor             = 0x81,
    Ixor            = 0x82,
    Lxor            = 0x83,
    Iinc            = 0x84,
    I2l             = 0x85,
    I2f             = 0x86,
    I2d             = 0x87,
    L2i             = 0x88,
    L2f             = 0x89,
    L2d             = 0x8a,
    F2i             = 0x8b,
    F2l             = 0x8c,
    F2d             = 0x8d,
    D2i             = 0x8e,
    D2l             = 0x8f,
    D2f             = 0x90,
    I2b             = 0x91,
    I2c             = 0x92,
    I2s             = 0x93,
    Lcmp            = 0x94,
    Fcmpl           = 0x95,
    Fcmpg           = 0x96,
    Dcmpl           = 0x97,
    Dcmpg           = 0x98,
    Ifeq            = 0x99,
    Ifne            = 0x9a,
    Iflt            = 0x9b,
    Ifge            = 0x9c,
    Ifgt            = 0x9d,
    Ifle            = 0x9e,
    IfIcmpeq        = 0x9f,
    IfIcmpne        = 0xa0,
    IfIcmplt        = 0xa1,
    IfIcmpge        = 0xa2,
    IfIcmpgt        = 0xa3,
    IfIcmple        = 0xa4,
    IfAcmpeq        = 0xa5,
    IfAcmpne        = 0xa6,
    Goto            = 0xa7,
    Jsr             = 0xa8,
    Ret             = 0xa9,
    Tableswitch     = 0xaa,
    Lookupswitch    = 0xab,
    Ireturn         = 0xac,
    Lreturn         = 0xad,
    Freturn         = 0xae,
    Dreturn         = 0xaf,
    Areturn         = 0xb0,
    Return          = 0xb1,
    Getstatic       = 0xb2,
    Putstatic       = 0xb3,
    Getfield        = 0xb4,
    Putfield        = 0xb5,
    Invokevirtual   = 0xb6,
    Invokespecial   = 0xb7,
    Invokestatic    = 0xb8,
    Invokeinterface = 0xb9,
    Invokedynamic   = 0xba,
    New             = 0xbb,
    Newarray        = 0xbc,
    Anewarray       = 0xbd,
    Arraylength     = 0xbe,
    Athrow          = 0xbf,
    Checkcast       = 0xc0,
    Instanceof      = 0xc1,
    Monitorenter    = 0xc2,
    Monitorexit     = 0xc3,
    Wide            = 0xc4,
    Multianewarray  = 0xc5,
    Ifnull          = 0xc6,
    Ifnonnull       = 0xc7,
    GotoW           = 0xc8,
    JsrW            = 0xc9,
    Breakpoint      = 0xca,
    /* cb - fd reserved */
    Impdep1         = 0xfe,
    Impdep2         = 0xff,
}
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Comparison {
    Eq,
    Ne,
    Lt,
    Ge,
    Gt,
    Le,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum JType {
    Int,
    Long,
    Float,
    Double,
    Object,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Operation {
    Branch   { kind : JType, way : Comparison, target : i16 },
    Jump     { target : i16 },
    Leave,   /* i.e. void return */
    Length,  /* i.e. arraylength */
    Load     { kind : JType, index : u8 },
    Noop,
    Store    { kind : JType, index : u8 },
    Subtract { kind : JType },
    Yield    { kind : JType }, /* i.e. return */
}

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

#[derive(Debug)]
struct AddressedOperation {
    address : u16,
    op : Operation,
}

fn handle_op(bytecode : &mut std::slice::Iter<u8>) -> (usize, Option<AddressedOperation>) {
    let mut used = 1;

    let op = match bytecode.next() {
        Some(byte) => JvmOps::from_u8(*byte),
        None => None,
    };
    let op = match op {
        Some(op) => op,
        None => return (0, None),
    };

    let handle = |op| println!("handling {:?} (0x{:02x})", &op, op as u8);
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
        b @ 0x99...0xa7 => Some(as_16(bytecode)), /* Ifeq...Goto */
        _ => None,
    };

    let converted_op = match op {
        b @ Iload0 | b @ Iload1 | b @ Iload2        => { handle(b); Load { kind: Int, index: index.unwrap() } },
        b @ Aload0 | b @ Aload1                     => { handle(b); Load { kind: Object, index: index.unwrap() } },
        b @ Arraylength                             => { handle(b); Length },
        b @ IfIcmpeq | b @ IfIcmple                 => { used += 2; handle(b); Branch { kind : Int, way: way.unwrap(), target: target.unwrap() } },
        b @ Ifle                                    => { used += 2; handle(b); Branch { kind : Int, way: way.unwrap(), target: target.unwrap() } },
        b @ Isub                                    => { handle(b); Subtract { kind: Int } },
        b @ Istore0 | b @ Istore1 | b @ Istore2     => { handle(b); Store { kind: Int, index: index.unwrap() } },
        b @ Goto                                    => { used += 2; handle(b); Jump { target: target.unwrap() } },
        b @ Ireturn                                 => { handle(b); Yield { kind: Int } },

        b @ Nop => { println!("handling {:?} (0x{:02x})", &b, b as u8); Noop },
        b @ _ => panic!("Unsupported byte 0x{:02x}", b as u8),
    };

    return (used, Some(AddressedOperation { address: 999/*XXX*/, op: converted_op }));
}

fn parse_bytecode(code : &Vec<u8>) -> (Vec<AddressedOperation>, HashMap<u16,usize>) {
    let mut out = Vec::new();
    let mut map = HashMap::new();

    let mut bytecode = code.iter();
    let mut i : u16 = 0;
    let mut j : usize = 0;
    while let (consumed, Some(p)) = handle_op(&mut bytecode) {
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

fn parse(name : &str) -> String {
    use classfile_parser::attribute_info::code_attribute_parser;

    let out = String::new();

    let class = parse_class(name).unwrap();
    let method = &class.methods[1];
    let c = &method.attributes[0].info;
    let code = code_attribute_parser(c).to_result().unwrap();

    let (parsed, map) = parse_bytecode(&code.code);
    emit_parsed(&parsed, &map);

    let pool = &class.const_pool;
    for f in 1..class.const_pool_size {
        println!("{}", mangle(&stringify(&pool, f).unwrap()));
    }

    return out;
}

fn main() {
    let stdin = io::stdin();
    let args : Vec<_> = env::args().collect();
    if args.len() != 2 {
        panic!("Need a single option (e.g., `--mangle`, `--demangle`)");
    }
    let func = match &args[1][..] {
        "--parse" => parse,
        "--mangle" => mangle,
        "--demangle" => demangle,
        _ => panic!("Invalid option `{}`", &args[1]),
    };
    for line in stdin.lock().lines() {
        println!("{}", func(&line.expect("Line read failure")));
    }
}

