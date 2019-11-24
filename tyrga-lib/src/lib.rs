//! This crate provides a library interface for converting Java Virtual Machine
//! bytecode into [tenyr](http://tenyr.info) assembly language - it is an
//! ahead-of-time compiler for the JVM, targeting the tenyr machine.

#![deny(clippy::items_after_statements)]
#![deny(clippy::needless_borrow)]
#![deny(clippy::option_unwrap_used)]
#![deny(clippy::redundant_field_names)]
#![deny(clippy::result_unwrap_used)]
#![deny(unconditional_recursion)]
#![deny(clippy::unreadable_literal)]
#![deny(clippy::just_underscores_and_digits)]
#![deny(unused_imports)]
#![deny(unreachable_patterns)]
#![deny(bare_trait_objects)]
#![deny(unused_mut)]
#![deny(unused_variables)]

// make macros visible to later modules
#[macro_use]
mod tenyr;

mod exprtree;
mod jvmtypes;
pub mod mangling;
mod stack;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::io::Write;
use std::ops::Range;

use classfile_parser::attribute_info::CodeAttribute;
use classfile_parser::attribute_info::StackMapFrame;
use classfile_parser::constant_info::*;
use classfile_parser::field_info::FieldAccessFlags;
use classfile_parser::field_info::FieldInfo;
use classfile_parser::method_info::MethodAccessFlags;
use classfile_parser::method_info::MethodInfo;
use classfile_parser::ClassFile;

use args::{count_params, count_returns};
use jvmtypes::*;
use tenyr::{Instruction, Register, SmallestImmediate};
use util::{Context, Contextualizer, Manglable};

type StackManager = stack::Manager;

pub type GeneralResult<T> = Result<T, Box<dyn Error>>;

const STACK_REGS : &[Register] = {
    use Register::*;
    &[B, C, D, E, F, G, H, I, J, K, L, M, N, O]
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Destination {
    Successor,
    Address(usize),
}

fn expand_immediate_load(
    sm : &mut StackManager,
    insn : Instruction,
    imm : i32,
) -> Vec<Instruction> {
    use tenyr::InsnGeneral as Gen;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::NoLoad;
    use SmallestImmediate::*;

    let make_imm = |temp_reg, imm|
        match imm {
            Imm12(_) =>
                unimplemented!("Imm12 was supposed to be handled separately"),
            Imm20(imm) =>
                vec![ tenyr_insn!( temp_reg <- (imm) ) ], // cannot fail
            Imm32(imm) => {
                let bot = tenyr::Immediate12::try_from_bits((imm & 0xfff) as u16)
                                .unwrap_or_else(|_| unsafe { std::hint::unreachable_unchecked() }); // truly unreachable

                // The following instructions will not fail
                tenyr_insn_list!(
                    temp_reg <- (imm >> 12)         ;
                    temp_reg <- temp_reg ^^ (bot)   ;
                ).collect()
            },
        };

    match (insn.kind, imm.into()) {
        (Type3(..), Imm12(imm)) => vec![ Instruction { kind : Type3(imm.into()), ..insn } ],
        (Type3(..), Imm20(imm)) => vec![ Instruction { kind : Type3(imm), ..insn } ],
        (Type0(g) , Imm12(imm)) => vec![ Instruction { kind : Type0(Gen { imm, ..g }), ..insn } ],
        (Type1(g) , Imm12(imm)) => vec![ Instruction { kind : Type1(Gen { imm, ..g }), ..insn } ],
        (Type2(g) , Imm12(imm)) => vec![ Instruction { kind : Type2(Gen { imm, ..g }), ..insn } ],

        (kind, imm) => {
            use std::iter::once;
            use tenyr::Opcode::{Add, BitwiseOr};

            let adder  = Gen { y : Register::A, op : Add , imm : 0_u8.into() };
            let (temp, gets) = sm.reserve_one();
            let pack = make_imm(temp, imm);
            let (op, a, b, c) = match kind {
                // the Type3 case should never be reached, but provides generality
                Type3(_) => (BitwiseOr, insn.x, Register::A, temp),
                Type0(g) => (g.op, insn.x, g.y, temp),
                Type1(g) => (g.op, insn.x, temp, g.y),
                Type2(g) => (g.op, temp, insn.x, g.y),
            };
            let gen = Gen { op, y : b, imm : 0_u8.into() };
            let operate = once(Instruction { kind : Type0(gen), x : a, dd : NoLoad, z : insn.z });
            let kind = Type0(Gen { y : c, ..adder });
            let add = once(Instruction { kind, x : insn.z, ..insn });
            let release = sm.release(1);

            std::iter::empty()
                .chain(gets)
                .chain(pack)
                .chain(operate)
                .chain(add)
                .chain(release)
                .collect()
        },
    }
}

#[test]
fn test_expand() {
    use tenyr::{Instruction, InstructionType, Register};
    use InstructionType::Type0;
    use Register::{A, B, C, D, E, F, G};

    let v = [C, D, E, F, G];
    let mut sm = StackManager::new(&v);

    {
        let imm = 867_5309; // 0x845fed
        let insn = tenyr_insn!( D -> [C * B] );
        let vv = expand_immediate_load(&mut sm, insn, imm);
        let rhs = 0xffff_ffed_u32 as i32;
        let expect = tenyr_insn_list!(
             C  <-  0x845u16    ;
             C  <-  C ^^ (rhs)  ;
             D  <-  C  *  B     ;
             D  -> [D  +  C]    ;
        );
        let ee : Vec<_> = expect.collect();
        assert_eq!(vv, ee);
    }

    {
        let imm = 123;
        let insn = tenyr_insn!( D -> [C + 0i8] );
        let vv = expand_immediate_load(&mut sm, insn, imm);
        let expect = tenyr_insn_list!(
             D  -> [C + 123i8]  ;
        );

        let ee : Vec<_> = expect.collect();
        assert_eq!(vv, ee);
    }

    {
        let imm = 867_5309; // 0x845fed
        let insn = tenyr_insn!( D -> [C + 0i8] );
        let vv = expand_immediate_load(&mut sm, insn, imm);
        let rhs = 0xffff_ffed_u32 as i32;
        let expect = tenyr_insn_list!(
             C  <-  0x845u16    ;
             C  <-  C ^^ (rhs)  ;
             D  <-  C  |  A     ;
             D  -> [D  +  C]    ;
        );
        let ee : Vec<_> = expect.collect();
        assert_eq!(vv, ee);
    }

    {
        let imm = 123;
        let insn = tenyr_insn!( D -> [C * B] );
        let vv = expand_immediate_load(&mut sm, insn, imm);
        let expect = tenyr_insn_list!(
             D  -> [C  *  B + 123i8];
        );
        let ee : Vec<_> = expect.collect();
        assert_eq!(vv, ee);
        if let Type0(ref g) = vv[0].kind {
            assert_eq!(g.imm, 123_u8.into());
        } else {
            panic!("wrong type");
        }
    }
}

type InsnPair = (Vec<Instruction>, Vec<Destination>);

fn make_target(target : impl std::string::ToString) -> exprtree::Atom {
    use exprtree::Atom::*;
    use exprtree::Expr;
    use exprtree::Operation::{Add, Sub};
    use std::rc::Rc;

    let a = Variable(target.to_string());
    let b = Expression(Rc::new(Expr { a : Variable(".".to_owned()), op : Add, b : Immediate(1) }));
    Expression(Rc::new(Expr { a, op : Sub, b }))
}

fn make_int_branch(
    sm : &mut StackManager,
    invert : bool,
    target : u16,
    target_name : &str,
    mut comp : impl FnMut(&mut StackManager) -> (Register, Vec<Instruction>),
) -> InsnPair {
    use Register::P;

    let (temp_reg, sequence) = comp(sm);
    let imm = tenyr::Immediate::Expr(make_target(target_name));
    let branch =
        if invert   { tenyr_insn!(   P <- (imm) &~ temp_reg + P     ) }
        else        { tenyr_insn!(   P <- (imm) &  temp_reg + P     ) };
    let mut v = sequence;
    v.push(branch);
    let dest = vec![
        Destination::Successor,
        Destination::Address(target.into()),
    ];
    (v, dest)
}

// number of slots of data we will save between locals and stack
const SAVE_SLOTS : u8 = 1;

fn make_builtin_name(proc : &str, descriptor : &str) -> String {
    mangle(&[&"tyrga/Builtin", &proc, &descriptor])
}

fn make_jump(target : u16, target_name : &str) -> InsnPair {
    use crate::tenyr::InstructionType::Type3;
    let kind = Type3(tenyr::Immediate::Expr(make_target(target_name)));
    let insn = Instruction { kind, z : Register::P, x : Register::P, ..tenyr::NOOP_TYPE3 };
    let dests = vec![ Destination::Address(target as usize) ];
    (vec![ insn ], dests)
}

fn make_call(
    sm : &mut StackManager,
    target : &str,
    descriptor : &str,
) -> Vec<Instruction> {
    use Register::P;

    let mut insns = Vec::new();
    insns.extend(sm.freeze(count_params(descriptor).into()));

    // Save return address through current stack pointer (callee will
    // decrement stack pointer)
    let sp = sm.get_stack_ptr();
    let far = format!("@+{}", target);
    let off = tenyr::Immediate20::Expr(exprtree::Atom::Variable(far));
    insns.extend(tenyr_insn_list!(
        [sp] <- P + 1i8 ;
        P <- P + (off)  ;
    ));

    insns.extend(sm.thaw(count_returns(descriptor).into()));
    insns
}

fn make_yield(
    sm : &mut StackManager,
    kind : JType,
    target_name : &str,
    max_locals : u16,
) -> InsnPair {
    use tenyr::MemoryOpType::StoreRight;
    use Register::P;

    let mut v = Vec::new();
    for i in (0..kind.size()).rev() { // get deepest first
        let (reg, gets) = sm.get(i.into());
        v.extend(gets);
        v.push(Instruction {
            dd : StoreRight,
            ..sm.index_local(reg, i.into(), max_locals)
        })
    }
    v.extend(sm.empty());
    let ex = tenyr::Immediate::Expr(make_target(target_name));
    v.push(tenyr_insn!( P <- (ex) + P ));

    (v, vec![]) // leaving the method is not a Destination we care about
}

fn make_constant<'a>(
    sm : &mut StackManager,
    gc : impl Contextualizer<'a>,
    details : Indirection<ExplicitConstant>,
) -> Vec<Instruction> {
    use jvmtypes::Indirection::{Explicit, Indirect};

    let mut make = |slice : &[_]| {
        slice.iter().fold(Vec::new(), |mut v, &value| {
            use Register::A;

            let (reg, gets) = sm.reserve_one();
            v.extend(gets);
            let insn = Instruction { z : reg, x : A, ..tenyr::NOOP_TYPE3 };
            v.extend(expand_immediate_load(sm, insn, value));
            v
        })
    };

    match details {
        Explicit(ExplicitConstant { kind, value }) =>
            match kind {
                JType::Object => make(&[ 0 ]), // all Object constants are nulls
                JType::Int    => make(&[ value.into() ]),
                JType::Long   => make(&[ 0, value.into() ]),
                JType::Float  => make(&[ f32::from(value).to_bits() as i32 ]),
                JType::Double => {
                    let bits = f64::from(value).to_bits();
                    make(&[ (bits >> 32) as i32, bits as i32 ])
                },
                _ => unreachable!("impossible Constant configuration"),
            },
        Indirect(index) => {
            use ConstantInfo::*;
            let c = gc.get_constant(index);
            match c {
                Integer(IntegerConstant { value }) => make(&[ *value ]),
                Long   (   LongConstant { value }) => make(&[ (*value >> 32) as i32, *value as i32 ]),
                Float  (  FloatConstant { value }) => make(&[ value.to_bits() as i32 ]),
                Double ( DoubleConstant { value }) => {
                    let bits = value.to_bits();
                    make(&[ (bits >> 32) as i32, bits as i32 ])
                },
                Class       (       ClassConstant { .. }) |
                String      (      StringConstant { .. }) |
                MethodHandle(MethodHandleConstant { .. }) |
                MethodType  (MethodTypeConstant   { .. }) =>
                    // Some of these will return Err in the future under certain circumstances, so
                    // do not be tempted to make the containing function infallible
                    unimplemented!("unhandled Constant configuration"),

                _ => unreachable!("impossible Constant configuration"),
            }
        }
    }
}

fn make_negation(sm : &mut StackManager) -> Vec<Instruction> {
    use Register::A;
    let mut v = Vec::new();
    let (y, gets) = sm.get(0);
    v.extend(gets);
    v.push(tenyr_insn!( y <- A - y ));
    v
}

fn make_bitwise(sm : &mut StackManager, kind : JType, op : tenyr::Opcode) -> Vec<Instruction> {
    use tenyr::InstructionType::Type0;

    let mut v = Vec::new();
    let size : u16 = kind.size().into();
    for i in (0..size).rev() {
        let (x, get_x) = sm.get(i + size);
        let (y, get_y) = sm.get(i);
        assert!(get_y.is_empty());
        v.extend(get_x);
        let z = x;
        let imm = 0_u8.into();
        let dd = tenyr::MemoryOpType::NoLoad;
        v.push(Instruction { kind : Type0(tenyr::InsnGeneral { y, op, imm }), x, z, dd });
    }
    v.extend(sm.release(size));
    v
}

fn make_arithmetic_general(sm : &mut StackManager, op : tenyr::Opcode) -> Vec<Instruction> {
    use tenyr::InstructionType::Type0;

    let mut v = Vec::new();
    let (x, gets) = sm.get(1);
    v.extend(gets);
    let (y, gets) = sm.get(0);
    v.extend(gets);
    let z = x;
    let dd = tenyr::MemoryOpType::NoLoad;
    let imm = 0_u8.into();
    v.push(Instruction { kind : Type0(tenyr::InsnGeneral { y, op, imm }), x, z, dd });
    v.extend(sm.release(1));
    v
}

fn make_arithmetic_call(
    sm : &mut StackManager,
    kind : JType,
    op : ArithmeticOperation,
) -> GeneralResult<Vec<Instruction>> {
    use std::convert::TryInto;
    use ArithmeticOperation::*;

    let ch : char = kind.try_into().expect("invalid char kind");

    let nargs = {
        match op {
            Add | Sub | Mul | Div | Rem | Shl | Shr | Ushr | And | Or | Xor => 2,
            Neg => 1,
        }
    };
    let descriptor = format!("({}){}", std::iter::repeat(ch).take(nargs).collect::<String>(), ch);

    // TODO replace lookup table with some automatic namer
    let proc = match op {
        Add  => "Add",
        Sub  => "Sub",
        Mul  => "Mul",
        Div  => "Div",
        Rem  => "Rem",
        Neg  => "Neg",
        Shl  => "Shl",
        Shr  => "Shr",
        Ushr => "Ushr",
        And  => "And",
        Or   => "Or",
        Xor  => "Xor",
    };

    Ok(make_call(sm, &make_builtin_name(&proc.to_lowercase(), &descriptor), &descriptor))
}

fn make_arithmetic(
    sm : &mut StackManager,
    kind : JType,
    op : ArithmeticOperation,
) -> GeneralResult<Vec<Instruction>> {
    let (general_op, bitwise_op) = {
        use tenyr::Opcode::*;
        match op {
            ArithmeticOperation::Add  => (Some(Add)            , None            ),
            ArithmeticOperation::Sub  => (Some(Subtract)       , None            ),
            ArithmeticOperation::Mul  => (Some(Multiply)       , None            ),
            ArithmeticOperation::Shl  => (Some(ShiftLeft)      , None            ),
            ArithmeticOperation::Shr  => (Some(ShiftRightArith), None            ),
            ArithmeticOperation::Ushr => (Some(ShiftRightLogic), None            ),
            ArithmeticOperation::And  => (Some(BitwiseAnd)     , Some(BitwiseAnd)),
            ArithmeticOperation::Or   => (Some(BitwiseOr)      , Some(BitwiseOr )),
            ArithmeticOperation::Xor  => (Some(BitwiseXor)     , Some(BitwiseXor)),
            _ => (None, None),
        }
    };

    {
        use ArithmeticOperation::Neg;
        use JType::Int;

        match (kind, bitwise_op, general_op, op) {
            (_  , Some(op), _       , _  ) => Ok(make_bitwise               (sm, kind, op)),
            (Int, _       , Some(op), _  ) => Ok(make_arithmetic_general    (sm,       op)),
            (Int, _       , _       , Neg) => Ok(make_negation              (sm          )),
            (_  , _       , _       , _  ) =>   (make_arithmetic_call       (sm, kind, op)),
        }
    }
}

fn make_mem_op(sm : &mut StackManager, op : LocalOperation, max_locals : u16) -> Vec<Instruction> {
    use tenyr::MemoryOpType::{LoadRight, StoreRight};
    use LocalOperation::{Load, Store};

    let idx;
    let (before, dd, after) = match op {
        Load  { kind, index } => { idx = index; (Some(kind.size()), LoadRight, None) },
        Store { kind, index } => { idx = index; (None, StoreRight, Some(kind.size())) },
    };

    let mut v = Vec::new();
    let size = before.xor(after).unwrap_or(0).into();
    v.extend(sm.reserve(before.unwrap_or(0).into()));
    for i in (0 .. size).rev() {
        let (reg, gets) = sm.get(i);
        v.extend(gets);
        v.push(Instruction { dd, ..sm.index_local(reg, (idx + i).into(), max_locals) })
    }
    v.extend(sm.release(after.unwrap_or(0).into()));
    v
}

fn make_increment(
    sm : &mut StackManager,
    index : u16,
    value : i16,
    max_locals : u16,
) -> GeneralResult<Vec<Instruction>> {
    use tenyr::MemoryOpType;

    let (temp_reg, mut v) = sm.reserve_one();
    let insn = sm.index_local(temp_reg, index.into(), max_locals);
    v.push(Instruction { dd : MemoryOpType::LoadRight, ..insn.clone() });
    v.push(tenyr_insn!( temp_reg <- temp_reg + (value) ));
    v.push(Instruction { dd : MemoryOpType::StoreRight, ..insn });
    v.extend(sm.release(1));

    Ok(v)
}

fn make_branch(
    sm : &mut StackManager,
    ops : OperandCount,
    way : Comparison,
    target : u16,
    target_name : &str,
) -> InsnPair {
    use tenyr::*;

    let (op, swap) = match way {
        jvmtypes::Comparison::Eq => (Opcode::CompareEq, false),
        jvmtypes::Comparison::Ne => (Opcode::CompareEq, false),
        jvmtypes::Comparison::Lt => (Opcode::CompareLt, false),
        jvmtypes::Comparison::Ge => (Opcode::CompareGe, false),
        jvmtypes::Comparison::Gt => (Opcode::CompareLt, true ),
        jvmtypes::Comparison::Le => (Opcode::CompareGe, true ),
    };
    let invert = way == jvmtypes::Comparison::Ne;

    make_int_branch(sm, invert, target, target_name, |sm| {
        use OperandCount::{Single, Double};

        let mut v = Vec::new();
        let count = ops as u16;
        let (lhs, gets) = sm.get(count - 1);
        v.extend(gets);
        let (rhs, gets) = match ops { Single => (Register::A, vec![]), Double => sm.get(0) };
        v.extend(gets);
        let temp_reg = lhs;
        let (rhs, lhs) = if swap { (lhs, rhs) } else { (rhs, lhs) };

        v.extend(sm.release(count));
        v.push(Instruction {
            kind : InstructionType::Type0(
                InsnGeneral {
                    y : rhs,
                    op,
                    imm : 0_u8.into(),
                }),
            z : temp_reg,
            x : lhs,
            dd : MemoryOpType::NoLoad,
        });
        (temp_reg, v)
    })
}

fn make_switch_lookup(
    sm : &mut StackManager,
    namer : impl Fn(&dyn Display) -> GeneralResult<String>,
    there : impl Fn(i32) -> u16,
    default : i32,
    pairs : Vec<(i32, i32)>,
) -> GeneralResult<InsnPair> {
    let (top, mut insns) = sm.get(0);
    let (temp_reg, gets) = sm.reserve_one(); // need a persistent temporary
    insns.extend(gets);

    let make = |(imm, target)|
        GeneralResult::Ok(make_int_branch(sm, false, there(target), &namer(&there(target))?,
            |sm| (temp_reg, expand_immediate_load(sm, tenyr_insn!(temp_reg <- top == 0i8), imm))));

    pairs
        .into_iter()
        .map(make)
        .chain(std::iter::once(Ok(make_jump(there(default), &namer(&there(default))?))))
        .try_fold((insns, Vec::new()), |(mut insns, mut dests), tup| {
            let (i, d) = tup?;
            insns.extend(i);
            dests.extend(d);
            Ok((insns, dests))
        })
}

fn make_switch_table(
    sm : &mut StackManager,
    namer : impl Fn(&dyn Display) -> GeneralResult<String>,
    there : impl Fn(i32) -> u16,
    default : i32,
    low : i32,
    high : i32,
    offsets : Vec<i32>,
) -> GeneralResult<InsnPair> {
    use std::iter::once;
    use tenyr::InstructionType::{Type1, Type2};
    use tenyr::*;

    type InsnType = dyn Fn(InsnGeneral) -> InstructionType;

    let (top, mut insns) = sm.get(0);
    let (temp_reg, gets) = sm.reserve_one(); // need a persistent temporary
    insns.extend(gets);

    let maker = |kind : &'static InsnType, imm : i32| {
        move |sm : &mut StackManager| {
            let insn = Instruction {
                kind : kind(
                    InsnGeneral {
                        y : Register::A,
                        op : Opcode::CompareLt,
                        imm : 0_u8.into(), // placeholder
                    }),
                z : temp_reg,
                x : top,
                dd : MemoryOpType::NoLoad,
            };
            let insns = expand_immediate_load(sm, insn, imm);
            (temp_reg, insns)
        }
    };

    let mut default_maker = |maker|
        GeneralResult::Ok(make_int_branch(sm, false, there(default), &namer(&there(default))?, maker));

    let insn = {
        use Register::P;
        tenyr_insn!( P <- top - 0i8 + P )
    };

    let offsets = offsets.into_iter().map(|far| Ok(make_jump(there(far), &namer(&there(far))?)));

    std::iter::empty()
        .chain(once(default_maker(maker(&Type1, low))))
        .chain(once(default_maker(maker(&Type2, high))))
        .chain(once(Ok((expand_immediate_load(sm, insn, low), vec![]))))
        .chain(offsets)
        .chain(once(Ok((sm.release(1), vec![])))) // release temporary
        .try_fold((insns, Vec::new()), |(mut insns, mut dests), tup| {
            let (i, d) = tup?;
            insns.extend(i);
            dests.extend(d);
            Ok((insns, dests))
        })
}

fn make_switch(
    sm : &mut StackManager,
    params : SwitchParams,
    namer : impl Fn(&dyn Display) -> GeneralResult<String>,
    addr : usize,
) -> GeneralResult<InsnPair> {
    use jvmtypes::SwitchParams::{Lookup, Table};

    let there = |x| (x + (addr as i32)) as u16;

    match params {
        Lookup { default, pairs } =>
            make_switch_lookup(sm, namer, there, default, pairs),
        Table { default, low, high, offsets } =>
            make_switch_table(sm, namer, there, default, low, high, offsets),
    }
}

fn make_array_op(sm : &mut StackManager, op : ArrayOperation) -> Vec<Instruction> {
    use jvmtypes::ArrayOperation::{GetLength, Load, Store};
    use tenyr::InstructionType::{Type1, Type3};
    use tenyr::MemoryOpType::{LoadRight, StoreRight};
    use tenyr::{InsnGeneral, Opcode};

    let mut v = Vec::new();
    let array_params = |sm : &mut StackManager, v : &mut Vec<Instruction>| {
        let (arr, gets) = sm.get(1);
        v.extend(gets);
        let (idx, gets) = sm.get(0);
        v.extend(gets);
        v.extend(sm.release(2));
        (idx, arr)
    };
    let kind;
    let (x, y, z, dd) = match op {
        GetLength => {
            // TODO document layout of arrays
            // This implementation assumes a reference to an array points to its first element, and
            // that one word below that element is a word containing the number of elements.
            let (top, gets) = sm.get(0);
            v.extend(gets);
            v.push(Instruction { kind : Type3((-1_i8).into()), dd : LoadRight, z : top, x : top });
            return v; // bail out early
        }
        Load(k) => {
            kind = k;
            let (idx, arr) = array_params(sm, &mut v);
            let (res, gets) = sm.reserve_one();
            v.extend(gets);
            (idx, arr, res, LoadRight)
        },
        Store(k) => {
            kind = k;
            let (val, gets) = sm.get(0);
            v.extend(gets);
            v.extend(sm.release(1));
            let (idx, arr) = array_params(sm, &mut v);
            (idx, arr, val, StoreRight)
        },
    };
    // For now, all arrays of int or smaller are stored unpacked (i.e. one bool/short/char
    // per 32-bit tenyr word)
    let (op, imm) = match kind.size() {
        1 => (Opcode::BitwiseOr, 0_u8),
        2 => (Opcode::ShiftLeft, 1_u8),
        _ => unreachable!(), // impossible size
    };
    let imm = imm.into();
    let kind = Type1(InsnGeneral { y, op, imm });
    let insn = Instruction { kind, z, x, dd };
    v.push(insn);
    v
}

fn make_invocation_virtual(
    sm : &mut StackManager,
    descriptor : &str,
    method_name : impl Manglable,
) -> GeneralResult<Vec<Instruction>> {
    use tenyr::Immediate20;
    use Register::P;

    let mut insns = Vec::new();
    // Save return address through current stack pointer (callee will
    // decrement stack pointer)
    let sp = sm.get_stack_ptr();
    let param_count = u16::from(count_params(descriptor));
    let (obj, gets) = sm.get(param_count);
    insns.extend(gets);
    let stack_count = param_count + 1; // extra "1" for `this`

    insns.extend(sm.freeze(stack_count));

    let (temp, gets) = sm.reserve_one();
    insns.extend(gets);
    let far = format!("@{}", mangle(&[&method_name, &"vslot"]));
    let off = Immediate20::Expr(exprtree::Atom::Variable(far));

    insns.extend(tenyr_insn_list!(
            temp <- [obj - 1i8] ;
            [sp] <- P + 1i8     ;
            P <- [temp + (off)] ;
        ));
    insns.extend(sm.release(1));

    insns.extend(sm.thaw(count_returns(descriptor).into()));

    Ok(insns)
}

fn make_invocation<'a>(
    sm : &mut StackManager,
    kind : InvokeKind,
    index : u16,
    gc : impl Contextualizer<'a>,
) -> GeneralResult<Vec<Instruction>> {
    let get_method_parts = || {
        if let ConstantInfo::MethodRef(mr) = gc.get_constant(index) {
            if let ConstantInfo::Class(cl) = gc.get_constant(mr.class_index) {
                if let ConstantInfo::NameAndType(nt) = gc.get_constant(mr.name_and_type_index) {
                    return GeneralResult::Ok((
                            gc.get_string(cl.name_index).ok_or("bad class name")?,
                            gc.get_string(nt.name_index).ok_or("bad method name")?,
                            gc.get_string(nt.descriptor_index).ok_or("bad method descriptor")?,
                        ))
                }
            }
        }

        Err("error during constant pool lookup".into())
    };

    let (class, method, descriptor) = get_method_parts()?;
    let name = &mangle(&[ &class, &method, &descriptor ]);

    match kind {
        // TODO fully handle Special (this is dumb partial handling)
        InvokeKind::Special =>
            Ok(make_call(sm, name, &descriptor).into_iter().chain(sm.release(1)).collect()),
        InvokeKind::Static =>
            Ok(make_call(sm, name, &descriptor)),
        // TODO vet handling of Virtual against JVM spec
        InvokeKind::Virtual => {
            if let ConstantInfo::MethodRef(mr) = gc.get_constant(index) {
                impl Manglable for Context<'_, &MethodRefConstant> {
                    fn pieces(&self) -> Vec<String> {
                        self.get_pieces(self.as_ref().class_index, self.as_ref().name_and_type_index)
                    }
                }

                make_invocation_virtual(sm, &descriptor, gc.contextualize(mr))
            } else {
                Err("bad constant kind".into())
            }
        },
        _ => unimplemented!("unhandled invocation kind {:?}", kind),
    }
}

fn make_stack_op(
    sm : &mut StackManager,
    op : StackOperation,
    size : OperandCount,
) -> Vec<Instruction> {
    use StackOperation::*;
    let mut copy = |i| sm.get_copy(i).1;
    match (op, size as u16) {
        (Pop, size) => sm.release(size),
        (Dup,    1) => copy(0),
        (Dup,    2) => [ copy(1), copy(1) ].concat(),
        _ => unimplemented!(),
    }
}

fn make_allocation<'a>(
    sm : &mut StackManager,
    details : AllocationKind,
    gc : impl Contextualizer<'a>,
) -> GeneralResult<Vec<Instruction>> {
    match details {
        AllocationKind::Array { kind, dims } => {
            use jvmtypes::Indirection::{Explicit, Indirect};

            match (kind, dims) {
                (Explicit(kind), 1) => {
                    let mut pre = match kind.size() {
                        1 => vec![],
                        // insert an instruction that doubles the top-of-stack count
                        2 => {
                            let mut v = Vec::new();
                            let (top, gets) = sm.get(0);
                            v.extend(gets);
                            v.push(tenyr_insn!( top <- top + top ));
                            v
                        },
                        _ => unreachable!("impossible size"),
                    };
                    let descriptor = "(I)Ljava.lang.Object;";
                    let name = make_builtin_name("alloc", descriptor);
                    pre.extend(make_call(sm, &name, descriptor));
                    Ok(pre)
                },
                (Indirect(_index), _) => unimplemented!(),
                _ => Err("invalid allocation configuration".into()),
            }
        },
        AllocationKind::Element { index } => {
            let class = gc.get_constant(index);
            if let ConstantInfo::Class(cc) = class {
                let name = gc.get_string(cc.name_index).ok_or("no class name")?;
                let desc = format!("()L{};", name);
                let call = mangle(&[&name, &"new"]);
                Ok(make_call(sm, &call, &desc))
            } else {
                Err("invalid ConstantInfo kind".into())
            }
        },
    }
}

fn make_compare(
    sm : &mut StackManager,
    kind : JType,
    nans : Option<NanComparisons>,
) -> GeneralResult<Vec<Instruction>> {
    use std::convert::TryInto;

    let ch : char = kind.try_into().expect("invalid char kind");

    let (gc, mut v) = sm.reserve_one();
    let n = match nans {
        Some(NanComparisons::Greater) => 1,
        Some(NanComparisons::Less) => -1,
        _ => 0,
    };
    v.push(tenyr_insn!( gc <- (n) ));

    let desc = format!("({}{}I)I", ch, ch);
    v.extend(make_call(sm, &make_builtin_name("cmp", &desc), &desc));
    Ok(v)
}

fn make_conversion(
    sm : &mut StackManager,
    from : JType,
    to : JType,
) -> GeneralResult<Vec<Instruction>> {
    use std::convert::TryInto;
    use tenyr::InsnGeneral;
    use tenyr::InstructionType::Type1;
    use tenyr::MemoryOpType::NoLoad;
    use tenyr::Opcode::{ShiftLeft, ShiftRightArith, ShiftRightLogic};
    use JType::{Byte, Char, Int, Long, Short};

    match (from, to) {
        (Int, Byte) |
        (Int, Char) |
        (Int, Short) => {
            let (top, mut insns) = sm.get(0);
            let op = match to { Byte | Short => ShiftRightArith, _ => ShiftRightLogic };
            let amount : u8 = match to { Byte => 24, _ => 16 };

            let make_kind = |op, imm| Type1(InsnGeneral { op, y : Register::A, imm });
            let make_insn = |kind| Instruction { dd : NoLoad, z : top, x : top, kind };

            insns.push(make_insn(make_kind(ShiftLeft, amount.into())));
            insns.push(make_insn(make_kind(op       , amount.into())));
            Ok(insns)
        },
        (Int, Long) => {
            let (low, get_actions) = sm.get(0);
            let (top, get_second) = sm.reserve_one();

            let moves = tenyr_insn_list!(
                top <- low          ;
                low <- top @ 31u8   ;
            );

            let insns = std::iter::empty()
                .chain(get_actions)
                .chain(get_second)
                .chain(moves)
                .collect();
            Ok(insns)
        },
        _ => {
            let ch_from : char = from.try_into().expect("invalid char kind");
            let ch_to   : char = to  .try_into().expect("invalid char kind");
            let name = format!("into_{}", ch_to); // TODO improve naming
            let desc = format!("({}){}", ch_from, ch_to);
            Ok(make_call(sm, &make_builtin_name(&name, &desc), &desc))
        },
    }
}

fn make_varaction<'a>(
    sm : &mut StackManager,
    op : VarOp,
    kind : VarKind,
    index : u16,
    gc : impl Contextualizer<'a>,
) -> GeneralResult<Vec<Instruction>> {
    use classfile_parser::constant_info::ConstantInfo::FieldRef;
    use tenyr::MemoryOpType::{LoadRight, StoreRight};

    if let FieldRef(fr) = gc.get_constant(index) {
        use exprtree::Atom;

        impl Manglable for Context<'_, &FieldRefConstant> {
            fn pieces(&self) -> Vec<String> {
                self.get_pieces(self.as_ref().class_index, self.as_ref().name_and_type_index)
            }
        }

        let len = {
            use classfile_parser::constant_info::ConstantInfo::NameAndType;
            use std::convert::TryFrom;

            let r = match gc.get_constant(fr.name_and_type_index) {
                NameAndType(nt) =>
                    gc.get_string(nt.descriptor_index)
                        .and_then(|x| x.chars().next())
                        .ok_or("bad descriptor")
                        .and_then(JType::try_from)
                        .map_err(Into::into),
                _ => GeneralResult::Err("unexpected kind".into()),
            };

            r?.size()
        };

        let make_off = |base, i| {
            let a = base;
            let b = Atom::Immediate(i);
            let op = exprtree::Operation::Add;
            let e = exprtree::Expr { a, b, op };
            tenyr::Immediate20::Expr(Atom::Expression(std::rc::Rc::new(e)))
        };

        let op_depth = match op { VarOp::Get => 0, VarOp::Put => len.into() };

        let format = |suff| format!("@{}", mangle(&[ &gc.contextualize(fr), &suff ]));

        let (drops, (reg, mut insns), base) = match kind {
            VarKind::Static => ( 0, (Register::P, vec![]), make_target(   format("static"      )) ),
            VarKind::Field  => ( 1, sm.get(op_depth)     , Atom::Variable(format("field_offset")) ),
        };

        let mut range = 0_i32..len.into();
        let mut reversed = range.clone().rev();
        let (prior, post, memop, iter) : (_, _, _, &mut dyn Iterator<Item=_>) = match op {
            VarOp::Get => (1, 0, LoadRight , &mut range   ),
            VarOp::Put => (0, 1, StoreRight, &mut reversed),
        };

        for it in iter {
            use crate::tenyr::InstructionType::Type3;

            let x = reg;
            let kind = Type3(make_off(base.clone(), it));
            insns.extend(sm.reserve(prior));
            let (z, gets) = sm.get(0);
            insns.extend(gets);
            insns.push(Instruction { dd : memop, z, x, kind });
            insns.extend(sm.release(post));
        }

        insns.extend(sm.release(drops));

        Ok(insns)
    } else {
        Err("invalid ConstantInfo kind".into())
    }
}

fn make_instructions<'a>(
    sm : &mut StackManager,
    (addr, op) : (usize, Operation),
    namer : impl Fn(&dyn Display) -> GeneralResult<String>,
    gc : impl Contextualizer<'a>,
    max_locals : u16,
) -> GeneralResult<InsnPair> {
    use Operation::*;

    // We need to track destinations and return them so that the caller can track stack state
    // through the chain of control flow, possibly cloning the StackManager state along the way to
    // follow multiple destinations. Each basic block needs to be visited only once, however, since
    // the JVM guarantees that every instance of every instruction within a method always sees the
    // same depth of the operand stack every time that instance is executed.
    let branching = |x| x;
    let no_branch = |x| Ok((x, vec![Destination::Successor]));

    let make_jump   = |_sm, target| Ok(make_jump(target, &namer(&target)?));
    let make_noop   = |_sm| vec![tenyr::NOOP_TYPE0];
    let make_branch = |sm, ops, way, target| Ok(make_branch(sm, ops, way, target, &namer(&target)?));
    let make_yield  = |sm, kind| Ok(make_yield(sm, kind, &namer(&"epilogue")?, max_locals));

    match op {
        Allocation { 0 : details      } => no_branch( make_allocation ( sm, details, gc              )?),
        Arithmetic { kind, op         } => no_branch( make_arithmetic ( sm, kind, op                 )?),
        ArrayOp    { 0 : aop          } => no_branch( make_array_op   ( sm, aop                      ) ),
        Branch     { ops, way, target } => branching( make_branch     ( sm, ops, way, target         ) ),
        Compare    { kind, nans       } => no_branch( make_compare    ( sm, kind, nans               )?),
        Constant   { 0 : details      } => no_branch( make_constant   ( sm, gc, details              ) ),
        Conversion { from, to         } => no_branch( make_conversion ( sm, from, to                 )?),
        Increment  { index, value     } => no_branch( make_increment  ( sm, index, value, max_locals )?),
        Invocation { kind, index      } => no_branch( make_invocation ( sm, kind, index, gc          )?),
        Jump       { target           } => branching( make_jump       ( sm, target                   ) ),
        LocalOp    { 0 : op           } => no_branch( make_mem_op     ( sm, op, max_locals           ) ),
        Noop       {                  } => no_branch( make_noop       ( sm                           ) ),
        StackOp    { op, size         } => no_branch( make_stack_op   ( sm, op, size                 ) ),
        Switch     { 0 : params       } => branching( make_switch     ( sm, params, namer, addr      ) ),
        VarAction  { op, kind, index  } => no_branch( make_varaction  ( sm, op, kind, index, gc      )?),
        Yield      { kind             } => branching( make_yield      ( sm, kind                     ) ),

        Unhandled  { ..               } => unimplemented!("unhandled operation {:?}", op)
    }
}

#[test]
fn test_make_instruction() -> GeneralResult<()> {
    use classfile_parser::constant_info::ConstantInfo;
    use classfile_parser::constant_info::ConstantInfo::Unusable;
    use jvmtypes::Indirection::Explicit;
    use tenyr::InstructionType::Type3;
    use tenyr::MemoryOpType::NoLoad;
    use Instruction;
    use Register::{A, B};
    struct Useless;
    impl<'a> Contextualizer<'a> for Useless {
        fn get_constant(&self, _ : u16) -> &'a ConstantInfo { &Unusable }
        fn contextualize<U>(&self, _ : U) -> Context<'a, U> {
            unreachable!("this code is for testing only")
        }
    }

    let mut sm = StackManager::new(STACK_REGS);
    let op = Operation::Constant(Explicit(ExplicitConstant { kind : JType::Int, value : 5 }));
    let namer = |x : &dyn Display| Ok(format!("{}:{}", "test", x.to_string()));
    let insn = make_instructions(&mut sm, (0, op), namer, Useless, 0)?;
    let imm = 5_u8.into();
    let rhs = Instruction { kind : Type3(imm), z : STACK_REGS[0], x : A, dd : NoLoad };
    assert_eq!(insn.0, vec![ rhs ]);
    assert_eq!(insn.0[0], tenyr_insn!( B <- 5u8 ));

    Ok(())
}

fn derive_ranges<'a>(
    max : usize,
    table : impl IntoIterator<Item = &'a StackMapFrame>,
) -> Vec<Range<usize>> {
    use classfile_parser::attribute_info::StackMapFrame::*;
    let mut deltas = table.into_iter().map(|f| match *f {
        SameFrame                           { frame_type }       => frame_type.into(),

        SameLocals1StackItemFrame           { frame_type, .. }   => u16::from(frame_type) - 64,

        SameLocals1StackItemFrameExtended   { offset_delta, .. }
            | ChopFrame                     { offset_delta, .. }
            | SameFrameExtended             { offset_delta, .. }
            | AppendFrame                   { offset_delta, .. }
            | FullFrame                     { offset_delta, .. } => offset_delta,
    });

    std::iter::once(0)
        .chain(deltas.next())
        .chain(deltas.map(|n| n + 1))
        .scan(0, |state, x| { *state += x; Some(usize::from(*state)) })
        .chain(std::iter::once(max))
        .collect::<Vec<_>>()
        .windows(2)
        .filter(|x| x[1] > x[0])
        .map(|x| x[0]..x[1])
        .collect()
}

fn get_method_code(method : &MethodInfo) -> GeneralResult<CodeAttribute> {
    use classfile_parser::attribute_info::code_attribute_parser;
    Ok(code_attribute_parser(&method.attributes[0].info).or(Err("error while parsing code attribute"))?.1)
}

mod util {
    use crate::mangling;
    use crate::GeneralResult;
    use classfile_parser::constant_info::ConstantInfo;
    use classfile_parser::constant_info::{ClassConstant, NameAndTypeConstant};
    use classfile_parser::field_info::FieldInfo;
    use classfile_parser::method_info::MethodInfo;
    use classfile_parser::ClassFile;
    use std::rc::Rc;

    // Previously, NAME_SEPARATOR was a colon, but in the JVM a colon is
    // technically a valid character in a method name.
    const NAME_SEPARATOR : &str = ";";

    pub(in super) trait Described {
        fn name_index(&self)       -> u16;
        fn descriptor_index(&self) -> u16;
    }

    impl Described for MethodInfo {
        fn name_index(&self)       -> u16 { self.name_index }
        fn descriptor_index(&self) -> u16 { self.descriptor_index }
    }

    impl Described for FieldInfo  {
        fn name_index(&self)       -> u16 { self.name_index }
        fn descriptor_index(&self) -> u16 { self.descriptor_index }
    }

    impl Described for NameAndTypeConstant {
        fn name_index(&self)       -> u16 { self.name_index }
        fn descriptor_index(&self) -> u16 { self.descriptor_index }
    }

    impl Manglable for Context<'_, &ClassConstant> {
        fn pieces(&self) -> Vec<String> {
            vec![ self.get_string(self.as_ref().name_index).expect("invalid constant pool string reference") ]
        }
    }

    impl Manglable for &str {
        fn pieces(&self) -> Vec<String> { vec![ self.to_string() ] }
    }

    impl Manglable for String {
        fn pieces(&self) -> Vec<String> { (self.as_ref() as &str).pieces() }
    }

    pub(in super) trait Contextualizer<'a> {
        fn contextualize<U>(&self, nested : U) -> Context<'a, U>;
        fn get_constant(&self, index : u16) -> &'a ConstantInfo;
        fn get_string(&self, i : u16) -> Option<String> {
            if let classfile_parser::constant_info::ConstantInfo::Utf8(u) = self.get_constant(i) {
                Some(u.utf8_string.to_string())
            } else {
                None
            }
        }
    }

    #[derive(Clone)]
    pub(in super) struct Context<'a, T> {
        constant_getter : Rc<dyn Fn(u16) -> &'a ConstantInfo + 'a>,
        nested : Rc<T>,
    }

    impl<'a, T> Context<'a, T> {
        pub fn get_class(&self, index : u16) -> GeneralResult<Context<'a, &'a ClassConstant>> {
            match self.get_constant(index) {
                classfile_parser::constant_info::ConstantInfo::Class(cl) => Ok(self.contextualize(cl)),
                _ => Err("not a class".into()),
            }
        }

        pub fn get_pieces(&self, ci : u16, nat : u16) -> Vec<String> {
            use classfile_parser::constant_info::ConstantInfo::{Class, NameAndType};

            if let Class(ni) = self.get_constant(ci) {
                let ni = ni.name_index;
                let ss = self.get_string(ni).expect("invalid constant pool string reference");
                if let NameAndType(nt) = self.get_constant(nat) {
                    let nt = self.contextualize(nt);
                    std::iter::once(ss).chain(nt.pieces()).collect()
                } else {
                    panic!("invalid constant pool name-and-type reference")
                }
            } else {
                panic!("invalid constant pool class reference")
            }
        }
    }

    impl<'a, T> Contextualizer<'a> for &Context<'a, T> {
        fn contextualize<U>(&self, nested : U) -> Context<'a, U> {
            Context { constant_getter : self.constant_getter.clone(), nested : Rc::new(nested) }
        }
        fn get_constant(&self, index : u16) -> &'a ConstantInfo {
            (self.constant_getter)(index)
        }
    }

    impl<T> AsRef<T> for Context<'_, T> {
        fn as_ref(&self) -> &T { &self.nested }
    }

    pub(in super) fn get_constant_getter<'a>(nested : &'a ClassFile) -> Context<'a, &ClassFile> {
        let gc = move |n| &nested.const_pool[usize::from(n) - 1];
        Context { constant_getter : Rc::new(gc), nested : Rc::new(nested) }
    }

    pub(in super) trait Manglable {
        fn pieces(&self) -> Vec<String>;
        fn stringify(&self) -> String {
            self.pieces().join(NAME_SEPARATOR)
        }
        fn mangle(&self) -> String {
            mangling::mangle(self.stringify().bytes())
        }
    }

    impl<T : Described> Manglable for Context<'_, &T> {
        fn pieces(&self) -> Vec<String> {
            vec![
                self.get_string(self.as_ref().name_index()      ).expect("invalid constant pool string reference"),
                self.get_string(self.as_ref().descriptor_index()).expect("invalid constant pool string reference"),
            ]
        }
    }

    impl Manglable for &[&dyn Manglable] {
        fn pieces(&self) -> Vec<String> {
            self.iter().map(|x| x.stringify()).collect() // TODO flatten
        }
    }
}

impl StackManager {
    pub fn index_local(&self, reg : Register, idx : i32, max_locals : u16) -> Instruction {
        let saved : u16 = SAVE_SLOTS.into();
        self.get_frame_offset(reg, idx - i32::from(saved + max_locals))
    }
}

type RangeList = Vec<Range<usize>>;
type OperationMap = BTreeMap<usize, Operation>;

fn get_ranges_for_method(
    method : &Context<'_, &MethodInfo>,
) -> GeneralResult<(RangeList, OperationMap)> {
    use classfile_parser::attribute_info::stack_map_table_attribute_parser;
    use classfile_parser::attribute_info::AttributeInfo;
    use classfile_parser::code_attribute::code_parser;
    use classfile_parser::constant_info::ConstantInfo::Utf8;

    let attribute_namer = |a : &AttributeInfo|
        match method.get_constant(a.attribute_name_index) {
            Utf8(u) => Ok((a.info.clone(), &u.utf8_string)),
            _ => Err("not a name"),
        };

    let code = get_method_code(method.as_ref())?;
    let names : Result<Vec<_>,_> = code.attributes.iter().map(attribute_namer).collect();
    let info = names?.into_iter().find(|(_, name)| name == &"StackMapTable").map(|t| t.0);
    let (_, vec) = code_parser(&code.code).or(Err("error while parsing method code"))?;
    let (max, _) = vec.last().ok_or("body unexpectedly empty")?;
    let max = max + 1; // convert address to an exclusive bound
    let ops = vec.into_iter().map(decode_insn).collect();
    match info {
        Some(info) => {
            let (_, keep) = stack_map_table_attribute_parser(&info).or(Err("error while parsing stack map"))?;
            Ok((derive_ranges(max, &keep.entries), ops))
        },
        _ =>
            Ok((vec![ 0..max ], ops)),
    }
}

fn mangle(list : &[&dyn Manglable]) -> String { list.mangle() }

fn make_label(
    class : &Context<'_, &ClassConstant>,
    method : &Context<'_, &MethodInfo>,
    suffix : impl Display,
) -> String {
    format!(".L{}", mangle(&[ class, method, &format!("__{}", suffix) ]))
}

fn make_basic_block(
    class : &Context<'_, &ClassConstant>,
    method : &Context<'_, &MethodInfo>,
    list : impl IntoIterator<Item = InsnPair>,
    range : &Range<usize>,
) -> (tenyr::BasicBlock, BTreeSet<usize>) {
    use tenyr::BasicBlock;
    use Destination::{Address, Successor};

    let mut insns = Vec::new();
    let mut exits = BTreeSet::new();

    let mut includes_successor = false;
    for insn in list {
        let does_branch = |&e| if let Address(n) = e { Some(n) } else { None };

        let (ins, exs) = insn;

        // update the state of includes_successor each time so that the last instruction's behavior
        // is captured
        includes_successor = exs.iter().any(|e| if let Successor = e { true } else { false });

        exits.extend(exs.iter().filter_map(does_branch).filter(|e| !range.contains(e)));
        insns.extend(ins);
    }
    let label = make_label(class, method, range.start);

    if includes_successor {
        exits.insert(range.end);
    }

    (BasicBlock { label, insns }, exits)
}

// The incoming StackManager represents a "prototype" StackManager which should be empty, and
// which will be cloned each time a new BasicBlock is seen.
fn make_blocks_for_method<'a, 'b>(
    class : &'a Context<'b, &'b ClassConstant>,
    method : &'a Context<'b, &'b MethodInfo>,
    sm : &StackManager,
    max_locals : u16,
) -> GeneralResult<Vec<tenyr::BasicBlock>> {
    use std::iter::FromIterator;

    struct Params<'a, 'b> {
        class : &'a Context<'b, &'b ClassConstant>,
        method : &'a Context<'b, &'b MethodInfo>,
        rangemap : &'a BTreeMap<usize, Range<usize>>,
        ops : &'a BTreeMap<usize, Operation>,
        max_locals : u16,
    }

    fn make_blocks(
        params : &Params,
        seen : &mut HashSet<usize>,
        mut sm : StackManager,
        which : &Range<usize>,
    ) -> GeneralResult<Vec<tenyr::BasicBlock>> {
        let (class, method, rangemap, ops, max_locals) =
            (params.class, params.method, params.rangemap, params.ops, params.max_locals);
        if seen.contains(&which.start) {
            return Ok(vec![]);
        }
        seen.insert(which.start);

        let block : GeneralResult<Vec<_>> =
            ops .range(which.clone())
                // TODO obviate clone by doing .remove() (no .drain on BTreeMap ?)
                .map(|(&u, o)| (u, o.clone()))
                .map(|x| make_instructions(&mut sm, x, |y| Ok(make_label(class, method, y)), class, max_locals))
                .collect();
        let (bb, ee) = make_basic_block(class, method, block?, which);
        let mut out = Vec::new();
        out.push(bb);

        for exit in &ee {
            // intentional clone of StackManager
            out.extend(make_blocks(params, seen, sm.clone(), &rangemap[exit])?);
        }

        Ok(out)
    }

    let (ranges, ops) = get_ranges_for_method(method)?;
    let rangemap = &BTreeMap::from_iter(ranges.into_iter().map(|r| (r.start, r)));
    let ops = &ops;

    let params = Params { class, method, rangemap, ops, max_locals };

    let mut seen = HashSet::new();

    let sm = sm.clone(); // intentional clone of StackManager
    make_blocks(&params, &mut seen, sm, &rangemap[&0])
}

#[test]
fn test_parse_classes() -> GeneralResult<()> {
    fn parse_class(path : &std::path::Path) -> GeneralResult<ClassFile> {
        let p = path.with_extension("");
        let p = p.to_str().ok_or("bad path")?;
        classfile_parser::parse_class(p).map_err(Into::into)
    }

    fn test_stack_map_table(path : &std::path::Path) -> GeneralResult<()> {
        let class = parse_class(path)?;
        let methods = class.methods.iter();
        for method in methods.filter(|m| !m.access_flags.contains(MethodAccessFlags::NATIVE)) {
            let sm = StackManager::new(STACK_REGS);
            let class = &util::get_constant_getter(&class).get_class(class.this_class)?;
            let max_locals = get_method_code(method)?.max_locals;
            let method = class.contextualize(method);
            let bbs = make_blocks_for_method(class, &method, &sm, max_locals)?;
            for bb in &bbs {
                eprintln!("{}", bb);
            }
        }

        Ok(())
    }

    let is_dir_or_class = |e : &walkdir::DirEntry| {
        e.metadata().map(|e| e.is_dir()).unwrap_or(false) ||
            e.file_name().to_str().map(|s| s.ends_with(".class")).unwrap_or(false)
    };
    for class in walkdir::WalkDir::new(env!("OUT_DIR")).into_iter().filter_entry(is_dir_or_class) {
        let class = class?;
        if ! class.path().metadata()?.is_dir() {
            let path = class.path();
            let name = class.file_name().to_str().ok_or("no name")?;
            eprintln!("Testing {} ({}) ...", path.display(), name);
            test_stack_map_table(path)?;
        }
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Method {
    name : String,
    prologue : tenyr::BasicBlock,
    blocks : Vec<tenyr::BasicBlock>,
    epilogue : tenyr::BasicBlock,
}

impl Display for Method {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, ".global {}", self.name)?;
        writeln!(f, "{}:", self.name)?;
        write!(f, "{}", self.prologue)?;
        for bb in &self.blocks {
            write!(f, "{}", bb)?
        }
        write!(f, "{}", self.epilogue)?;
        Ok(())
    }
}

mod args {
    use crate::JType;
    use std::convert::TryFrom;

    pub fn field_size(ch : char) -> Result<u8, &'static str> {
        JType::try_from(ch).map(JType::size)
    }

    fn count_internal(s : &str) -> Result<u8, String> {
        fn eat(s : &str) -> Result<usize, String> {
            let ch = s.chars().next().ok_or("string ended too soon")?;
            match ch {
                'B' | 'C' | 'F' | 'I' | 'S' | 'Z' | 'D' | 'J' | 'V' => Ok(1),
                'L' => Ok(1 + s.find(';').ok_or("string ended too soon")?),
                '[' => Ok(1 + eat(&s[1..])?),
                _ => Err(format!("unexpected character {}", ch)),
            }
        }

        if s.is_empty() {
            return Ok(0);
        }
        let ch = s.chars().next().unwrap_or('x'); // .next() cannot fail since s is not empty
        let mine = field_size(ch);
        Ok(mine? + count_internal(&s[eat(s)?..])?)
    }

    // JVM limitations restrict the count of method parameters to 255 at most
    pub fn count_params(descriptor : &str) -> u8 {
        let open = 1; // byte index of open parenthesis is 0
        let close = descriptor.rfind(')').expect("descriptor missing closing parenthesis");
        count_internal(&descriptor[open..close]).expect("parse error in descriptor")
    }

    // JVM limitations restrict the count of return values to 1 at most, of size 2 at most
    pub fn count_returns(descriptor : &str) -> u8 {
        let close = descriptor.rfind(')').expect("descriptor missing closing parenthesis");
        count_internal(&descriptor[close+1..]).expect("parse error in descriptor")
    }
}

#[test]
fn test_count_params() {
    assert_eq!(3, count_params("(III)V"));
    assert_eq!(4, count_params("(JD)I"));
    assert_eq!(2, count_params("(Lmetasyntactic;Lvariable;)I"));
    assert_eq!(1, count_params("([[[I)I"));
    assert_eq!(0, count_params("()Lplaceholder;"));
    assert_eq!(0, count_params("()D"));
}

#[test]
#[should_panic]
fn test_count_params_panic() { let _ = count_params("("); }

#[test]
fn test_count_returns() {
    assert_eq!(0, count_returns("(III)V"));
    assert_eq!(1, count_returns("(JD)I"));
    assert_eq!(1, count_returns("(Lmetasyntactic;Lvariable;)I"));
    assert_eq!(1, count_returns("([[[I)I"));
    assert_eq!(1, count_returns("()Lplaceholder;"));
    assert_eq!(2, count_returns("()D"));
}

#[test]
#[should_panic]
fn test_count_returns_panic() {
    let _ = count_returns("(");
}

fn translate_method<'a, 'b>(
        class : &'a Context<'b, &'b ClassConstant>,
        method : &'a Context<'b, &'b MethodInfo>,
    ) -> GeneralResult<Method>
{
    let mr = method.as_ref();
    let total_locals = get_method_code(mr)?.max_locals;
    let descriptor = class.get_string(mr.descriptor_index).ok_or("method descriptor missing")?;
    let num_returns = count_returns(&descriptor).into();
    // Pretend we have at least as many locals as we have return-slots, so we have somewhere to
    // store our results when we Yield.
    let max_locals = total_locals.max(num_returns);

    let sm = &StackManager::new(STACK_REGS);
    let sp = sm.get_stack_ptr();
    let max_locals_i32 = i32::from(max_locals);

    let prologue = {
        let name = "prologue";
        let off = -(max_locals_i32 - i32::from(count_params(&descriptor)) + i32::from(SAVE_SLOTS));
        let insns = vec![ tenyr_insn!( sp <-  sp + (off) ) ];
        let label = make_label(class, method, name);
        tenyr::BasicBlock { label, insns }
    };

    let epilogue = {
        let name = "epilogue";
        let off = i32::from(SAVE_SLOTS) + i32::from(total_locals) - i32::from(num_returns);
        let down = i32::from(num_returns) - max_locals_i32;
        let rp = Register::P;
        let mv = if off != 0 { Some(tenyr_insn!( sp <-  sp + (off) )) } else { None };
        let insns = mv.into_iter().chain(std::iter::once(tenyr_insn!( rp <- [sp + (down)] ))).collect();
        let label = make_label(class, method, name);
        tenyr::BasicBlock { label, insns }
    };

    let blocks = make_blocks_for_method(class, method, sm, max_locals)?;
    let name = mangle(&[ class, method ]);
    Ok(Method { name, prologue, blocks, epilogue })
}

fn write_method_table(
    class : &Context<'_, &ClassConstant>,
    methods : &[MethodInfo],
    outfile : &mut dyn Write,
) -> std::io::Result<()> {
    let label = ".Lmethod_table";
    writeln!(outfile, "{}:", label)?;

    let names = methods.iter().map(|method| mangle(&[ class, &class.contextualize(method) ]) );
    let lengths : Vec<_> = names.map(|s| (s.len(), s)).collect();
    let width = lengths.iter().fold(0, |c, (len, _)| c.max(*len));

    for (method, (_, mangled_name)) in methods.iter().zip(lengths) {
        let flags = method.access_flags;

        writeln!(outfile, "    .word @{:width$} - {}, {:#06x}",
            mangled_name, label, flags.bits(), width=width)?;
    }

    writeln!(outfile, "{}_end:", label)?;
    writeln!(outfile, "    .zero 0")?;
    writeln!(outfile)?;
    Ok(())
}

fn write_vslot_list(
    class : &Context<'_, &ClassConstant>,
    methods : &[MethodInfo],
    outfile : &mut dyn Write,
) -> std::io::Result<()> {
    let non_virtual = MethodAccessFlags::STATIC | MethodAccessFlags::PRIVATE;

    let virtuals = methods.iter().filter(|m| (m.access_flags & non_virtual).is_empty());
    let names = virtuals.map(|m| mangle(&[ class, &class.contextualize(m), &"vslot" ]) );
    let lengths : Vec<_> = names.map(|s| (s.len(), s)).collect();
    let width = lengths.iter().fold(0, |c, (len, _)| c.max(*len));

    for (index, (_, mangled_name)) in lengths.iter().enumerate() {
        writeln!(outfile, "    .global {}", mangled_name)?;
        writeln!(outfile, "    .set    {:width$}, {}", mangled_name, index, width=width)?;
    }
    if ! lengths.is_empty() {
        writeln!(outfile)?;
    }

    Ok(())
}

fn write_field_list(
    class : &Context<'_, &ClassConstant>,
    fields : &[FieldInfo],
    outfile : &mut dyn Write,
    suff : &str,
    selector : impl Fn(&&FieldInfo) -> bool,
    generator : impl Fn(&mut dyn Write, &str, usize, usize, usize) -> GeneralResult<()>,
) -> GeneralResult<()> {
    let tuples = fields.iter().filter(selector).map(|f| {
        let s = class.get_string(f.descriptor_index).ok_or("missing descriptor")?;
        let desc = s.chars().next().ok_or("empty descriptor")?;
        let size = args::field_size(desc)?.into();
        let name = mangle(&[ class, &class.contextualize(f), &suff ]);
        Ok((size, f, name))
    });

    let tuples : Vec<_> = tuples.flat_map(GeneralResult::into_iter).collect();
    let sums = tuples.iter().scan(0, |off, tup| { let old = *off; *off += tup.0; Some(old) });
    let width = tuples.iter().map(|t| t.2.len()).fold(0, usize::max);

    for ((size, _, mangled_name), offset) in tuples.iter().zip(sums) {
        generator(outfile, mangled_name, offset, *size, width)?;
    }
    if ! tuples.is_empty() {
        writeln!(outfile)?;
    }

    Ok(())
}

/// Emits tenyr assembly language corresponding to the given input class.
pub fn translate_class(class : ClassFile, outfile : &mut dyn Write) -> GeneralResult<()> {
    if class.major_version < 50 {
        return Err("need classfile version ≥50.0 for StackMapTable attributes".into());
    }

    let fields = &class.fields;
    let methods = &class.methods;
    let class = &util::get_constant_getter(&class).get_class(class.this_class)?;

    write_method_table(class, methods, outfile)?;
    write_vslot_list(class, methods, outfile)?;

    let is_static = |f : &&FieldInfo| f.access_flags.contains(FieldAccessFlags::STATIC);
    let print_field = |outfile : &mut dyn Write, slot_name : &str, offset, _size, width| {
        writeln!(outfile, "    .global {}", slot_name)?;
        writeln!(outfile, "    .set    {:width$}, {}", slot_name, offset, width=width)?;
        Ok(())
    };
    write_field_list(class, fields, outfile, "field_offset", |f| ! is_static(f), print_field)?;
    let print_static = |outfile : &mut dyn Write, slot_name : &str, _offset, size, width| {
        writeln!(outfile, "    .global {}", slot_name)?;
        writeln!(outfile, "    {:width$}: .zero {}", slot_name, size, width=width)?;
        Ok(())
    };
    write_field_list(class, fields, outfile, "static", &is_static, &print_static)?;

    for method in methods.iter().filter(|m| !m.access_flags.contains(MethodAccessFlags::NATIVE)) {
        let method = class.contextualize(method);
        let mm = translate_method(class, &method)?;
        writeln!(outfile, "{}", mm)?;
    }

    Ok(())
}

