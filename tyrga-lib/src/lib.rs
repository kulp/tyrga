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
use util::{Context, ContextConstantGetter, Contextualizer, Manglable};

type StackManager = stack::Manager;

const STACK_REGS : &[Register] = {
    use Register::*;
    &[B, C, D, E, F, G, H, I, J, K, L, M, N, O]
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Destination {
    Successor,
    Address(usize),
}

fn expand_immediate_load(sm : &mut StackManager, insn : Instruction, imm : i32)
    -> GeneralResult<Vec<Instruction>>
{
    use tenyr::InsnGeneral as Gen;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::NoLoad;
    use SmallestImmediate::*;

    fn make_imm(temp_reg : Register, imm : SmallestImmediate) -> GeneralResult<Vec<Instruction>> {
        use Register::A;

        let result = match imm {
            Imm12(imm) => // This path is fairly useless, but it completes generality
                vec![ tenyr_insn!( temp_reg <- A | A + (imm) )? ],
            Imm20(imm) =>
                vec![ tenyr_insn!( temp_reg <- (imm) )? ],
            Imm32(imm) => {
                let bot = tenyr::Immediate12::try_from_bits((imm & 0xfff) as u16)?; // cannot fail

                tenyr_insn_list!(
                    temp_reg <- (imm >> 12)         ;
                    temp_reg <- temp_reg ^^ (bot)   ;
                ).collect()
            },
        };
        Ok(result)
    };

    let v = match (insn.kind, imm.into()) {
        (Type3(..), Imm12(imm)) => vec![ Instruction { kind : Type3(imm.into()), ..insn } ],
        (Type3(..), Imm20(imm)) => vec![ Instruction { kind : Type3(imm), ..insn } ],
        (Type0(g) , Imm12(imm)) => vec![ Instruction { kind : Type0(Gen { imm, ..g }), ..insn } ],
        (Type1(g) , Imm12(imm)) => vec![ Instruction { kind : Type1(Gen { imm, ..g }), ..insn } ],
        (Type2(g) , Imm12(imm)) => vec![ Instruction { kind : Type2(Gen { imm, ..g }), ..insn } ],

        (kind, imm) => {
            use std::iter::once;
            use tenyr::Opcode::{Add, BitwiseOr};

            let adder  = Gen { y : Register::A, op : Add , imm : 0_u8.into() };
            let reserve = sm.reserve(1).into_iter();
            let (temp, gets) = sm.get(0);
            let pack = make_imm(temp, imm)?.into_iter();
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
            let release = sm.release(1).into_iter();

            std::iter::empty()
                .chain(reserve)
                .chain(gets)
                .chain(pack)
                .chain(operate)
                .chain(add)
                .chain(release)
                .collect()
        },
    };

    Ok(v)
}

#[test]
fn test_expand() -> GeneralResult<()> {
    use tenyr::{Instruction, InstructionType, Register};
    use InstructionType::Type0;
    use Register::{A, B, C, D, E, F, G};

    let v = [C, D, E, F, G];
    let mut sm = StackManager::new(&v);

    {
        let imm = 867_5309; // 0x845fed
        let insn = tenyr_insn!( D -> [C * B] )?;
        let vv = expand_immediate_load(&mut sm, insn, imm)?;
        let rhs = 0xffff_ffed_u32 as i32;
        let expect = tenyr_insn_list!(
             C  <-  0x845       ;
             C  <-  C ^^ (rhs)  ;
             D  <-  C  *  B     ;
             D  -> [D  +  C]    ;
        );
        let ee : Vec<_> = expect.collect();
        assert_eq!(vv, ee);
    }

    {
        let imm = 123;
        let insn = tenyr_insn!( D -> [C + 0] )?;
        let vv = expand_immediate_load(&mut sm, insn, imm)?;
        let expect = tenyr_insn_list!(
             D  -> [C + 123]    ;
        );

        let ee : Vec<_> = expect.collect();
        assert_eq!(vv, ee);
    }

    {
        let imm = 867_5309; // 0x845fed
        let insn = tenyr_insn!( D -> [C + 0] )?;
        let vv = expand_immediate_load(&mut sm, insn, imm)?;
        let rhs = 0xffff_ffed_u32 as i32;
        let expect = tenyr_insn_list!(
             C  <-  0x845       ;
             C  <-  C ^^ (rhs)  ;
             D  <-  C  |  A     ;
             D  -> [D  +  C]    ;
        );
        let ee : Vec<_> = expect.collect();
        assert_eq!(vv, ee);
    }

    {
        let imm = 123;
        let insn = tenyr_insn!( D -> [C * B] )?;
        let vv = expand_immediate_load(&mut sm, insn, imm)?;
        let expect = tenyr_insn_list!(
             D  -> [C  *  B + 123]  ;
        );
        let ee : Vec<_> = expect.collect();
        assert_eq!(vv, ee);
        if let Type0(ref g) = vv[0].kind {
            assert_eq!(g.imm, 123_u8.into());
        } else {
            return Err("wrong type".into());
        }
    }

    Ok(())
}

type Namer<'a> = dyn Fn(&dyn fmt::Display) -> GeneralResult<String> + 'a;
type InsnPair = (Vec<Instruction>, Vec<Destination>);
type MakeInsnResult = GeneralResult<InsnPair>;

fn make_target(target : &dyn std::string::ToString) -> GeneralResult<exprtree::Atom> {
    use exprtree::Atom::*;
    use exprtree::Expr;
    use exprtree::Operation::{Add, Sub};
    use std::rc::Rc;

    let a = Variable(target.to_string());
    let b = Expression(Rc::new(Expr { a : Variable(".".to_owned()), op : Add, b : Immediate(1) }));
    Ok(Expression(Rc::new(Expr { a, op : Sub, b })))
}

fn make_int_branch(
    sm : &mut StackManager,
    invert : bool,
    target : u16,
    target_namer : &Namer,
    mut comp : impl FnMut(&mut StackManager) -> GeneralResult<(Register, Vec<Instruction>)>
) -> MakeInsnResult
{
    use Register::P;

    let (temp_reg, sequence) = comp(sm)?;
    let imm = tenyr::Immediate::Expr(make_target(&target_namer(&target)?)?);
    let branch =
        if invert   { tenyr_insn!(   P <- (imm) &~ temp_reg + P     ) }
        else        { tenyr_insn!(   P <- (imm) &  temp_reg + P     ) };
    let mut v = sequence;
    v.push(branch?);
    let dest = vec![
        Destination::Successor,
        Destination::Address(target.into()),
    ];
    Ok((v, dest))
}

// number of slots of data we will save between locals and stack
const SAVE_SLOTS : u8 = 1;

fn make_builtin_name(proc : &str, descriptor : &str) -> GeneralResult<String> {
    mangle(&[&"tyrga/Builtin", &proc, &descriptor])
}

fn make_jump(target : u16, target_namer : &Namer) -> GeneralResult<Instruction> {
    use crate::tenyr::InstructionType::Type3;
    use Register::P;
    Ok(Instruction {
        kind : Type3(tenyr::Immediate::Expr(make_target(&target_namer(&target)?)?)),
        z : P,
        x : P,
        ..tenyr::NOOP_TYPE3
    })
}

fn make_call(sm : &mut StackManager, target : &str, descriptor : &str) -> GeneralResult<Vec<Instruction>> {
    use Register::P;

    let mut insns = Vec::new();
    insns.extend(sm.freeze(count_params(descriptor)?.into()));

    // Save return address through current stack pointer (callee will
    // decrement stack pointer)
    let sp = sm.get_stack_ptr();
    let far = format!("@+{}", target);
    let off = tenyr::Immediate20::Expr(exprtree::Atom::Variable(far));
    insns.extend(tenyr_insn_list!(
        [sp] <- P + 1   ;
        P <- P + (off)  ;
    ));

    insns.extend(sm.thaw(count_returns(descriptor)?.into()));
    Ok(insns)
}

fn make_yield(
    sm : &mut StackManager,
    kind : JType,
    target_namer : &Namer,
    max_locals : u16,
) -> MakeInsnResult
{
    use tenyr::MemoryOpType::StoreRight;
    use util::index_local;
    use Register::P;

    let mut v = Vec::new();
    for i in (0 .. kind.size()).rev() { // get deepest first
        let (reg, gets) = sm.get(i.into());
        v.extend(gets);
        v.push(Instruction { dd : StoreRight, ..index_local(sm, reg, i.into(), max_locals) })
    }
    v.extend(sm.empty());
    let ex = tenyr::Immediate::Expr(make_target(&target_namer(&"epilogue")?)?);
    v.push(tenyr_insn!( P <- (ex) + P )?);

    Ok((v, vec![])) // leaving the method is not a Destination we care about
}

fn make_constant<'a, T>(
    sm : &mut StackManager,
    gc : &T,
    details : Indirection<ExplicitConstant>,
) -> GeneralResult<Vec<Instruction>>
where
    T : ContextConstantGetter<'a>
{
    use jvmtypes::Indirection::{Explicit, Indirect};

    let mut make = |slice : &[_]| {
        slice.iter().fold(
            Ok(vec![]),
            |v : GeneralResult<Vec<_>>, &value| {
                use Register::A;

                let mut v = v?;
                v.extend(sm.reserve(1));
                let (reg, gets) = sm.get(0);
                v.extend(gets);
                let insn = Instruction { z : reg, x : A, ..tenyr::NOOP_TYPE3 };
                v.extend(expand_immediate_load(sm, insn, value)?);
                Ok(v)
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
                _ => Err("encountered impossible Constant configuration".into()),
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
                    unimplemented!("unhandled Constant configuration"),

                _ => Err("encountered impossible Constant configuration".into()),
            }
        }
    }
}

fn make_arithmetic(
    sm : &mut StackManager,
    kind : JType,
    op : ArithmeticOperation
) -> GeneralResult<Vec<Instruction>>
{
    use tenyr::InstructionType::Type0;

    fn make_descriptor(kind : JType, op : ArithmeticOperation) -> GeneralResult<String> {
        use std::convert::TryInto;
    
        let ch : char = kind.try_into()?;
    
        let nargs = {
            use ArithmeticOperation::*;
            match op {
                Add | Sub | Mul | Div | Rem | Shl | Shr | Ushr | And | Or | Xor => 2,
                Neg => 1,
            }
        };
        Ok(format!("({}){}", std::iter::repeat(ch).take(nargs).collect::<String>(), ch))
    }

    // TODO replace make_op_name with some automatic namer
    fn make_op_name(op : ArithmeticOperation) -> &'static str {
        use ArithmeticOperation::*;
        match op {
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
        }
    }

    let make_name = |kind, op| {
        let descriptor = make_descriptor(kind, op)?;
        let proc = make_op_name(op).to_lowercase();
        make_builtin_name(&proc, &descriptor)
    };

    let make_op = |x| {
        use tenyr::Opcode::*;
        match x {
            ArithmeticOperation::Add  => Some(Add),
            ArithmeticOperation::Sub  => Some(Subtract),
            ArithmeticOperation::Mul  => Some(Multiply),
            ArithmeticOperation::Shl  => Some(ShiftLeft),
            ArithmeticOperation::Shr  => Some(ShiftRightArith),
            ArithmeticOperation::Ushr => Some(ShiftRightLogic),
            ArithmeticOperation::And  => Some(BitwiseAnd),
            ArithmeticOperation::Or   => Some(BitwiseOr),
            ArithmeticOperation::Xor  => Some(BitwiseXor),
            _ => None,
        }
    };

    match (kind, op, make_op(op)) {
        (JType::Int, ArithmeticOperation::Neg, _) => {
            use Register::A;
            let mut v = Vec::new();
            let (y, gets) = sm.get(0);
            v.extend(gets);
            v.push(tenyr_insn!( y <- A - y )?);
            Ok(v)
        },
        (JType::Int, _, Some(op)) => {
            use tenyr::{InsnGeneral, MemoryOpType};
            let mut v = Vec::new();
            let (x, gets) = sm.get(1);
            v.extend(gets);
            let (y, gets) = sm.get(0);
            v.extend(gets);
            let z = x;
            let dd = MemoryOpType::NoLoad;
            let imm = 0_u8.into();
            v.push(Instruction { kind : Type0(InsnGeneral { y, op, imm }), x, z, dd });
            v.extend(sm.release(1));
            Ok(v)
        }
        _ => make_call(sm, &make_name(kind, op)?, &make_descriptor(kind, op)?),
    }
}

fn make_mem_op(
    sm : &mut StackManager,
    op : LocalOperation,
    max_locals : u16,
) -> GeneralResult<Vec<Instruction>>
{
    use LocalOperation::{Load, Store};
    use tenyr::MemoryOpType::{LoadRight, StoreRight};

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
        v.push(Instruction { dd, ..util::index_local(sm, reg, (idx + i).into(), max_locals) })
    }
    v.extend(sm.release(after.unwrap_or(0).into()));
    Ok(v)
}

fn make_increment(
    sm : &mut StackManager,
    index : u16,
    value : i16,
    max_locals : u16,
) -> GeneralResult<Vec<Instruction>>
{
    use tenyr::MemoryOpType;
    use util::index_local;

    let mut v = Vec::new();
    v.extend(sm.reserve(1));
    let (temp_reg, gets) = sm.get(0);
    v.extend(gets);
    let insn = index_local(sm, temp_reg, index.into(), max_locals);
    v.push(Instruction { dd : MemoryOpType::LoadRight, ..insn.clone() });
    v.push(tenyr_insn!( temp_reg <- temp_reg + (value) )?);
    v.push(Instruction { dd : MemoryOpType::StoreRight, ..insn });
    v.extend(sm.release(1));

    Ok(v)
}

fn make_branch(
    sm : &mut StackManager,
    ops : OperandCount,
    way : Comparison,
    target : u16,
    target_namer : &Namer,
) -> MakeInsnResult
{
    use tenyr::*;

    let (op, swap, invert) = match way {
        jvmtypes::Comparison::Eq => (Opcode::CompareEq, false, false),
        jvmtypes::Comparison::Ne => (Opcode::CompareEq, false, true ),
        jvmtypes::Comparison::Lt => (Opcode::CompareLt, false, false),
        jvmtypes::Comparison::Ge => (Opcode::CompareGe, false, false),
        jvmtypes::Comparison::Gt => (Opcode::CompareLt, true , false),
        jvmtypes::Comparison::Le => (Opcode::CompareGe, true , false),
    };

    let opper = |sm : &mut StackManager| {
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
        Ok((temp_reg, v))
    };

    make_int_branch(sm, invert, target, target_namer, opper)
}

fn make_switch(
    sm : &mut StackManager,
    params : SwitchParams,
    target_namer : &Namer,
    addr : usize,
) -> MakeInsnResult
{
    use jvmtypes::SwitchParams::{Lookup, Table};

    let here = addr as i32;

    let mut dests = Vec::new();
    let mut insns = Vec::new();
    let (top, gets) = sm.get(0);

    insns.extend(gets);
    insns.extend(sm.reserve(1)); // need a persistent temporary

    let (temp_reg, gets) = sm.get(0);
    insns.extend(gets);

    match params {
        Lookup { default, pairs } => {
            let (i, d) : (Vec<_>, Vec<_>) =
                pairs
                    .into_iter()
                    .map(|(imm, target)|
                        make_int_branch(sm, false, (target + here) as u16, target_namer,
                            |sm| Ok((
                                temp_reg,
                                expand_immediate_load(sm, tenyr_insn!(temp_reg <- top == 0)?, imm)?
                            ))
                        ))
                    .collect::<Result<Vec<_>,_>>()?
                    .into_iter()
                    .unzip();

            insns.extend(i.concat());
            dests.extend(d.concat());

            let far = (default + here) as u16;
            insns.push(make_jump(far, target_namer)?);
            dests.push(Destination::Address(far.into()));

            Ok((insns, dests))
        },
        Table { default, low, high, offsets } => {
            use tenyr::*;
            use tenyr::InstructionType::{Type1, Type2};
            type InsnType = dyn Fn(InsnGeneral) -> InstructionType;

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
                    let insns = expand_immediate_load(sm, insn, imm)?;
                    Ok((temp_reg, insns))
                }
            };

            let far = (default + here) as u16;
            let (lo_insns, lo_dests) =
                make_int_branch(sm, false, far, target_namer, maker(&Type1, low))?;
            let (hi_insns, hi_dests) =
                make_int_branch(sm, false, far, target_namer, maker(&Type2, high))?;

            insns.extend(lo_insns);
            insns.extend(hi_insns);

            let insn = {
                use Register::P;
                tenyr_insn!( P <- top - 0 + P )
            };
            insns.extend(expand_immediate_load(sm, insn?, low)?);

            let (i, d) : (Vec<_>, Vec<_>) =
                offsets
                    .into_iter()
                    .map(|n| GeneralResult::Ok((
                        make_jump((n + here) as u16, target_namer)?,
                        Destination::Address((n + here) as usize))
                    ))
                    .collect::<Result<Vec<_>,_>>()?
                    .into_iter()
                    .unzip();

            insns.extend(i);
            dests.extend(d);

            dests.extend(lo_dests);
            dests.extend(hi_dests);

            insns.extend(sm.release(1)); // release temporary

            Ok((insns, dests))
        },
    }
}

fn make_array_op(
    sm : &mut StackManager,
    op : ArrayOperation,
) -> GeneralResult<Vec<Instruction>>
{
    use jvmtypes::ArrayOperation::{Load, Store, GetLength};
    use tenyr::{InsnGeneral, Opcode};
    use tenyr::InstructionType::{Type1, Type3};
    use tenyr::MemoryOpType::{LoadRight, StoreRight};

    let mut v = Vec::new();
    let array_params = |sm : &mut StackManager, v : &mut Vec<Instruction>| {
        let (arr, gets) = sm.get(1);
        v.extend(gets);
        let (idx, gets) = sm.get(0);
        v.extend(gets);
        v.extend(sm.release(2));
        GeneralResult::Ok((idx, arr))
    };
    let kind;
    let (x, y, z, dd) = match op {
        GetLength => {
            // TODO document layout of arrays
            // This implementation assumes a reference to an array points to its first element, and
            // that one word below that element is a word containing the number of elements.
            let mut v = Vec::new();
            let (top, gets) = sm.get(0);
            v.extend(gets);
            let insn = Instruction { kind : Type3((-1_i8).into()), dd : LoadRight, z : top, x : top };
            v.push(insn);
            return Ok(v); // bail out early
        }
        Load(k) => {
            kind = k;
            let (idx, arr) = array_params(sm, &mut v)?;
            v.extend(sm.reserve(1));
            let (res, gets) = sm.get(0);
            v.extend(gets);
            (idx, arr, res, LoadRight)
        },
        Store(k) => {
            kind = k;
            let (val, gets) = sm.get(0);
            v.extend(gets);
            v.extend(sm.release(1));
            let (idx, arr) = array_params(sm, &mut v)?;
            (idx, arr, val, StoreRight)
        },
    };
    // For now, all arrays of int or smaller are stored unpacked (i.e. one bool/short/char
    // per 32-bit tenyr word)
    let (op, imm) = match kind.size() {
        1 => Ok((Opcode::BitwiseOr, 0_u8)),
        2 => Ok((Opcode::ShiftLeft, 1_u8)),
        _ => Err("bad kind size"),
    }?;
    let imm = imm.into();
    let kind = Type1(InsnGeneral { y, op, imm });
    let insn = Instruction { kind, z, x, dd };
    v.push(insn);
    Ok(v)
}

fn make_invocation<'a, T>(
    sm : &mut StackManager,
    kind : InvokeKind,
    index : u16,
    gc : &T,
) -> GeneralResult<Vec<Instruction>>
where
    T : ContextConstantGetter<'a> + Contextualizer<'a>
{
    match kind {
        // TODO fully handle Special (this is dumb partial handling)
        InvokeKind::Special => {
            let mut insns =
                make_call(sm, &make_callable_name(gc, index)?, &get_method_parts(gc, index)?[2])?;
            insns.extend(sm.release(1));
            Ok(insns)
        },
        InvokeKind::Static =>
            make_call(sm, &make_callable_name(gc, index)?, &get_method_parts(gc, index)?[2]),
        // TODO vet handling of Virtual against JVM spec
        InvokeKind::Virtual => {
            if let ConstantInfo::MethodRef(mr) = gc.get_constant(index) {
                use tenyr::Immediate20;
                use Register::P;

                let mut insns = Vec::new();
                // Save return address through current stack pointer (callee will
                // decrement stack pointer)
                let sp = sm.get_stack_ptr();
                let descriptor = &get_method_parts(gc, index)?[2];
                let param_count = u16::from(count_params(descriptor)?);
                let (obj, gets) = sm.get(param_count);
                insns.extend(gets);
                let stack_count = param_count + 1; // extra "1" for `this`

                insns.extend(sm.freeze(stack_count));
                insns.extend(sm.reserve(1));

                let (temp, gets) = sm.get(0);
                insns.extend(gets);
                let far = format!("@{}", mangle(&[&gc.contextualize(mr), &"vslot"])?);
                let off = Immediate20::Expr(exprtree::Atom::Variable(far));

                insns.extend(tenyr_insn_list!(
                    temp <- [obj - 1]   ;
                    [sp] <- P + 1       ;
                    P <- [temp + (off)] ;
                ));
                insns.extend(sm.release(1));

                insns.extend(sm.thaw(count_returns(descriptor)?.into()));

                Ok(insns)
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
) -> GeneralResult<Vec<Instruction>>
{
    match op {
        StackOperation::Pop =>
            Ok(sm.release(size as u16)),
        StackOperation::Dup => {
            let size = size as u16;
            let (_, gets) = sm.get(size - 1); // ensure spills are reloaded
            let old : Vec<_> = (0..size).map(|i| sm.get(i).0).collect();
            let res = sm.reserve(size);
            let put : GeneralResult<Vec<_>> = (0..size).map(|i| {
                let (new, _) = sm.get(i); // already forced gets above
                let t = old[i as usize];
                tenyr_insn!( new <- t )
            }).collect();

            Ok([ gets, res, put? ].concat())
        },
        _ => unimplemented!(),
    }
}

fn make_allocation<'a, T>(
    sm : &mut StackManager,
    details : AllocationKind,
    gc : &T,
) -> GeneralResult<Vec<Instruction>>
where
    T : ContextConstantGetter<'a>
{
    match details {
         AllocationKind::Array { kind, dims } => {
            use jvmtypes::Indirection::Explicit;

            let descriptor = "(I)Ljava.lang.Object;";
            let proc = "alloc";
            let name = make_builtin_name(proc, descriptor)?;
            let regs : Vec<_> =
                (0..dims.into())
                    .map(|r| sm.get(r))
                    .collect();

            match (kind, dims) {
                (Explicit(kind), 1) => {
                    let mut pre = match kind.size() {
                        1 => Ok(vec![]),
                        // insert an instruction that doubles the top-of-stack count
                        2 => {
                            let mut v = Vec::new();
                            let (top, gets) = regs[0].clone();
                            v.extend(gets);
                            v.push(tenyr_insn!( top <- top + top )?);
                            Ok(v)
                        },
                        _ => Err("impossible size"),
                    }?;
                    let v = make_call(sm, &name, descriptor)?;
                    pre.extend(v);
                    Ok(pre)
                },
                _ => unimplemented!(),
            }
        },
        AllocationKind::Element { index } => {
            use util::get_string;

            let class = gc.get_constant(index);
            if let ConstantInfo::Class(cc) = class {
                let name = get_string(gc, cc.name_index).ok_or("no class name")?;
                let desc = format!("()L{};", name);
                let call = mangle(&[&name, &"new"])?;
                make_call(sm, &call, &desc)
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
) -> GeneralResult<Vec<Instruction>>
{
    use std::convert::TryInto;

    let mut v = sm.reserve(1);
    let name = "cmp";
    let ch : char = kind.try_into()?;

    let (gc, gets) = sm.get(0);
    v.extend(gets);
    let n = match nans {
        Some(NanComparisons::Greater) => 1,
        Some(NanComparisons::Less) => -1,
        _ => 0,
    };
    v.push(tenyr_insn!( gc <- (n) )?);

    let desc = format!("({}{}I)I", ch, ch);
    let insns = make_call(sm, &make_builtin_name(name, &desc)?, &desc)?;
    v.extend(insns);
    Ok(v)
}

fn make_conversion(
    sm : &mut StackManager,
    from : JType,
    to : JType,
) -> GeneralResult<Vec<Instruction>>
{
    use JType::{Byte, Char, Int, Short};
    use std::convert::TryInto;
    use tenyr::InsnGeneral;
    use tenyr::InstructionType::Type1;
    use tenyr::MemoryOpType::NoLoad;
    use tenyr::Opcode::{ShiftLeft, ShiftRightArith, ShiftRightLogic};

    let top;
    let mut insns;
    match (from, to) {
        (Int, Byte) |
        (Int, Char) |
        (Int, Short) => {
            let (t, v) = sm.get(0);
            top = t;
            insns = v;
        },
        _ => {
            let ch_from : char = from.try_into()?;
            let ch_to   : char = to  .try_into()?;
            let name = format!("into_{}", ch_to); // TODO improve naming
            let desc = format!("({}){}", ch_from, ch_to);
            return Ok(make_call(sm, &make_builtin_name(&name, &desc)?, &desc)?)
        },
    }

    let op = match to { Byte | Short => ShiftRightArith, _ => ShiftRightLogic };
    let amount : u8 = match to { Byte => 24, _ => 16 };

    let make_kind = |op, imm| Type1(InsnGeneral { op, y : Register::A, imm });
    let make_insn = |kind| Instruction { dd : NoLoad, z : top, x : top, kind };

    insns.push(make_insn(make_kind(ShiftLeft, amount.into())));
    insns.push(make_insn(make_kind(op       , amount.into())));
    Ok(insns)
}

fn make_varaction<'a, T>(
    sm : &mut StackManager,
    op : VarOp,
    kind : VarKind,
    index : u16,
    gc : &T,
) -> GeneralResult<Vec<Instruction>>
where
    T : ContextConstantGetter<'a> + Contextualizer<'a>
{
    use classfile_parser::constant_info::ConstantInfo::FieldRef;
    use tenyr::MemoryOpType::{LoadRight, StoreRight};

    if let FieldRef(fr) = gc.get_constant(index) {
        use exprtree::Atom;

        let fr = gc.contextualize(fr);
        let mut insns = Vec::new();

        let len = util::field_type(&fr)?.size();

        let make_off = |base, i| {
            use exprtree::Operation::Add;
            use std::rc::Rc;
            use tenyr::Immediate20;

            let e = exprtree::Expr { a : base, b : Atom::Immediate(i), op : Add };
            Immediate20::Expr(Atom::Expression(Rc::new(e)))
        };

        let op_depth = match op { VarOp::Get => 0, VarOp::Put => 1 };

        let format = |suff|
            GeneralResult::Ok(format!("@{}", mangle(&[ &fr, &suff ])?));

        let ((reg, gets), base, drops) = match kind {
            VarKind::Static =>
                (   (Register::P, vec![]),
                    make_target(&format("static")?)?,
                    0,
                ),
            VarKind::Field =>
                (   sm.get((op_depth * len).into()),
                    Atom::Variable(format("field_offset")?),
                    1, // drop object reference
                ),
        };
        insns.extend(gets);

        let mut range = 0_i32..len.into();
        let mut reversed = range.clone().rev();
        let (prior, post, memop, iter) : (_, _, _, &mut dyn Iterator<Item=_>) = match op {
            VarOp::Get => (1, 0, LoadRight , &mut range   ),
            VarOp::Put => (0, 1, StoreRight, &mut reversed),
        };

        for it in iter {
            let imm = make_off(base.clone(), it);
            insns.extend(sm.reserve(prior));
            let (top, gets) = sm.get(0);
            insns.extend(gets);
            insns.push(Instruction { dd : memop, ..tenyr_insn!( top <- [reg + (imm)] )? });
            insns.extend(sm.release(post));
        }

        insns.extend(sm.release(drops));

        Ok(insns)
    } else {
        Err("invalid ConstantInfo kind".into())
    }
}

fn make_instructions<'a, T>(
    sm : &mut StackManager,
    (addr, op) : (usize, Operation),
    target_namer : &Namer,
    gc : &T,
    max_locals : u16,
) -> MakeInsnResult
where
    T : ContextConstantGetter<'a> + Contextualizer<'a>
{
    use Operation::*;

    // We need to track destinations and return them so that the caller can track stack state
    // through the chain of control flow, possibly cloning the StackManager state along the way to
    // follow multiple destinations. Each basic block needs to be visited only once, however, since
    // the JVM guarantees that every instance of every instruction within a method always sees the
    // same depth of the operand stack every time that instance is executed.
    let no_branch = |x| Ok((x, vec![Destination::Successor]));

    match op {
        Allocation(details) =>
            no_branch(make_allocation(sm, details, gc)?),
        Arithmetic { kind, op } =>
            no_branch(make_arithmetic(sm, kind, op)?),
        ArrayOp(aop) =>
            no_branch(make_array_op(sm, aop)?),
        Branch { ops, way, target } =>
            make_branch(sm, ops, way, target, target_namer),
        Compare { kind, nans } =>
            no_branch(make_compare(sm, kind, nans)?),
        Constant(details) =>
            no_branch(make_constant(sm, gc, details)?),
        Conversion { from, to } =>
            no_branch(make_conversion(sm, from, to)?),
        Increment { index, value } =>
            no_branch(make_increment(sm, index, value, max_locals)?),
        Invocation { kind, index } =>
            no_branch(make_invocation(sm, kind, index, gc)?),
        Jump { target } =>
            Ok((vec![ make_jump(target, target_namer)? ], vec![ Destination::Address(target as usize) ])),
        LocalOp(op) =>
            no_branch(make_mem_op(sm, op, max_locals)?),
        Noop =>
            no_branch(vec![ tenyr::NOOP_TYPE0 ]),
        StackOp { op, size } =>
            no_branch(make_stack_op(sm, op, size)?),
        Switch(params) =>
            make_switch(sm, params, target_namer, addr),
        VarAction { op, kind, index } =>
            no_branch(make_varaction(sm, op, kind, index, gc)?),
        Yield { kind } =>
            make_yield(sm, kind, target_namer, max_locals),

        Unhandled( .. ) =>
            unimplemented!("unhandled operation {:?}", op)
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
    use Register::A;
    struct Useless;
    impl<'a> ContextConstantGetter<'a> for Useless {
        fn get_constant(&self, _ : u16) -> &'a ConstantInfo { &Unusable }
    }
    impl<'a> Contextualizer<'a> for Useless {
        fn contextualize<U>(&self, _ : U) -> Context<'a, U> {
            unimplemented!("this code is for testing only")
        }
    }

    let mut sm = StackManager::new(STACK_REGS);
    let op = Operation::Constant(Explicit(ExplicitConstant { kind : JType::Int, value : 5 }));
    let namer = |x : &dyn fmt::Display| Ok(format!("{}:{}", "test", x.to_string()));
    let insn = make_instructions(&mut sm, (0, op), &namer, &Useless, 0)?;
    let imm = 5_u8.into();
    let rhs = Instruction { kind : Type3(imm), z : STACK_REGS[0], x : A, dd : NoLoad };
    assert_eq!(insn.0, vec![ rhs ]);
    assert_eq!(insn.0[0].to_string(), " B  <-  5");

    Ok(())
}

pub type GeneralResult<T> = std::result::Result<T, Box<dyn Error>>;

fn generic_error(e : impl Error) -> Box<dyn Error> { format!("unknown error: {}", e).into() }

type RangeMap<T> = (Vec<Range<usize>>, BTreeMap<usize, T>);

fn derive_ranges<'a, T>(body : Vec<(usize, T)>, table : impl IntoIterator<Item=&'a StackMapFrame>)
    -> GeneralResult<RangeMap<T>>
{
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

    let max = body.last().ok_or("body unexpectedly empty")?.0 + 1;

    let ranges =
        std::iter::once(0)
            .chain(deltas.next())
            .chain(deltas.map(|n| n + 1))
            .scan(0, |state, x| { *state += x; Some(usize::from(*state)) })
            .chain(std::iter::once(max))
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|x| x[1] > x[0])
            .map(|x| x[0]..x[1])
            .collect();

    let tree = body.into_iter().collect();
    Ok((ranges, tree))
}

fn get_method_code(method : &MethodInfo) -> GeneralResult<CodeAttribute> {
    use classfile_parser::attribute_info::code_attribute_parser;
    Ok(code_attribute_parser(&method.attributes[0].info).map_err(generic_error)?.1)
}

mod util {
    use super::jvmtypes::JType;
    use super::mangling;
    use super::tenyr::Instruction;
    use super::tenyr::Register;
    use super::GeneralResult;
    use super::StackManager;
    use classfile_parser::constant_info::ConstantInfo;
    use classfile_parser::constant_info::FieldRefConstant;
    use classfile_parser::constant_info::{ClassConstant, MethodRefConstant, NameAndTypeConstant};
    use classfile_parser::field_info::FieldInfo;
    use classfile_parser::method_info::MethodInfo;
    use classfile_parser::ClassFile;
    use std::convert::TryFrom;
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
        fn pieces(&self) -> GeneralResult<Vec<String>> {
            let r : GeneralResult<String> =
                get_string(self, self.as_ref().name_index).ok_or_else(|| "no name".into());
            Ok(vec![ r? ])
        }
    }

    impl Manglable for &str {
        fn pieces(&self) -> GeneralResult<Vec<String>> { Ok(vec![ self.to_string() ]) }
    }

    impl Manglable for String {
        fn pieces(&self) -> GeneralResult<Vec<String>> { (self.as_ref() as &str).pieces() }
    }

    fn from_nat<T>(gc : &Context<'_, &T>, ci : u16, nat : u16) -> GeneralResult<Vec<String>> {
        use classfile_parser::constant_info::ConstantInfo::{Class, NameAndType};

        if let Class(ni) = gc.get_constant(ci) {
            let ni = ni.name_index;
            let ss = get_string(gc, ni).ok_or("no such name")?;
            if let NameAndType(nt) = gc.get_constant(nat) {
                let nt = gc.contextualize(nt);
                Ok(std::iter::once(ss).chain(nt.pieces()?.into_iter()).collect())
            } else {
                Err("invalid ConstantInfo kind".into())
            }
        } else {
            Err("invalid ConstantInfo kind".into())
        }
    }

    impl Manglable for Context<'_, &FieldRefConstant> {
        fn pieces(&self) -> GeneralResult<Vec<String>> {
            from_nat(self, self.as_ref().class_index, self.as_ref().name_and_type_index)
        }
    }

    impl Manglable for Context<'_, &MethodRefConstant> {
        fn pieces(&self) -> GeneralResult<Vec<String>> {
            from_nat(self, self.as_ref().class_index, self.as_ref().name_and_type_index)
        }
    }

    pub(in super) type ConstantGetter<'a> = dyn Fn(u16) -> &'a ConstantInfo + 'a;

    pub(in super) trait Contextualizer<'a> {
        fn contextualize<U>(&self, nested : U) -> Context<'a, U>;
    }

    pub(in super) trait ContextConstantGetter<'a> {
        fn get_constant(&self, index : u16) -> &'a ConstantInfo;
    }

    #[derive(Clone)]
    pub(in super) struct Context<'a, T> {
        get_constant : Rc<ConstantGetter<'a>>,
        nested : Rc<T>,
    }

    impl<'a, T> Contextualizer<'a> for Context<'a, T> {
        fn contextualize<U>(&self, nested : U) -> Context<'a, U> {
            Context { get_constant : self.get_constant.clone(), nested : Rc::new(nested) }
        }
    }

    impl<'a, T> ContextConstantGetter<'a> for Context<'a, T> {
        fn get_constant(&self, index : u16) -> &'a ConstantInfo {
            (self.get_constant)(index)
        }
    }

    impl<T> AsRef<T> for Context<'_, T> {
        fn as_ref(&self) -> &T { &self.nested }
    }

    pub(in super) fn get_constant_getter<'a>(nested : &'a ClassFile) -> Context<'a, &ClassFile> {
        let gc = move |n| &nested.const_pool[usize::from(n) - 1];
        Context { get_constant : Rc::new(gc), nested : Rc::new(nested) }
    }

    pub(in super) fn get_string(g : &dyn ContextConstantGetter, i : u16) -> Option<String> {
        if let classfile_parser::constant_info::ConstantInfo::Utf8(u) = g.get_constant(i) {
            Some(u.utf8_string.to_string())
        } else {
            None
        }
    }

    pub(in super) trait Manglable {
        fn pieces(&self) -> GeneralResult<Vec<String>>;
        fn stringify(&self) -> GeneralResult<String> {
            Ok(self.pieces()?.join(NAME_SEPARATOR))
        }
        fn mangle(&self) -> GeneralResult<String> {
            mangling::mangle(self.stringify()?.bytes())
        }
    }

    impl<T : Described> Manglable for Context<'_, &T> {
        fn pieces(&self) -> GeneralResult<Vec<String>> {
            Ok(vec![
                get_string(self, self.as_ref().name_index()      ).ok_or("no name")?,
                get_string(self, self.as_ref().descriptor_index()).ok_or("no desc")?,
            ])
        }
    }

    impl Manglable for &[&dyn Manglable] {
        fn pieces(&self) -> GeneralResult<Vec<String>> {
            self.iter().map(|x| x.stringify()).collect() // TODO flatten
        }
    }

    pub(in super) fn field_type(fr : &Context<'_, &FieldRefConstant>) -> GeneralResult<JType> {
        use classfile_parser::constant_info::ConstantInfo::NameAndType;
        if let NameAndType(nt) = fr.get_constant(fr.as_ref().name_and_type_index) {
            let desc = get_string(fr, nt.descriptor_index).ok_or("no description")?;
            let ch = desc.chars().next().ok_or("descriptor too short")?;
            let kind = JType::try_from(ch)?;
            Ok(kind)
        } else {
            Err("unexpected kind".into())
        }
    }

    pub(in super) fn index_local(sm : &StackManager, reg : Register, idx : i32, max_locals : u16) -> Instruction {
        let saved : u16 = super::SAVE_SLOTS.into();
        sm.get_frame_offset(reg, idx - i32::from(saved + max_locals))
    }
}

fn get_ranges_for_method(method : &Context<'_, &MethodInfo>)
    -> GeneralResult<RangeMap<Operation>>
{
    use classfile_parser::attribute_info::AttributeInfo;
    use classfile_parser::attribute_info::stack_map_table_attribute_parser;
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
    let keep;
    let table = match info {
        Some(info) => {
            keep = stack_map_table_attribute_parser(&info).map_err(generic_error)?;
            &keep.1.entries
        },
        _ => &[] as &[StackMapFrame],
    };

    let vec = code_parser(&code.code).map_err(generic_error)?.1;
    let (ranges, map) = derive_ranges(vec, table)?;
    let ops = map.into_iter().map(decode_insn).collect();
    Ok((ranges, ops))
}

fn get_method_parts(g : &dyn ContextConstantGetter, pool_index : u16)
    -> GeneralResult<[String ; 3]>
{
    use classfile_parser::constant_info::ConstantInfo::{Class, MethodRef, NameAndType};
    use util::get_string;

    let get_string = |n| get_string(g, n);

    if let MethodRef(mr) = g.get_constant(pool_index) {
        if let Class(cl) = g.get_constant(mr.class_index) {
            if let NameAndType(nt) = g.get_constant(mr.name_and_type_index) {
                return Ok([
                        get_string(cl.name_index).ok_or("bad class name")?,
                        get_string(nt.name_index).ok_or("bad method name")?,
                        get_string(nt.descriptor_index).ok_or("bad method descriptor")?,
                    ]);
            }
        }
    }

    Err("error during constant pool lookup".into())
}

fn make_callable_name(g : &dyn ContextConstantGetter, pool_index : u16) -> GeneralResult<String> {
    let parts = get_method_parts(g, pool_index)?;
    // TODO do this generically, without explicit indexing
    mangle(&[ &parts[0], &parts[1], &parts[2] ])
}

fn mangle(list : &[&dyn Manglable]) -> GeneralResult<String> { list.mangle() }

fn get_class<'a, T>(ctx : util::Context<'a, T>, index : u16)
    -> GeneralResult<Context<'a, &'a ClassConstant>>
{
    match ctx.get_constant(index) {
        classfile_parser::constant_info::ConstantInfo::Class(cl) => Ok(ctx.contextualize(cl)),
        _ => Err("not a class".into()),
    }
}

fn make_label(
        class : &Context<'_, &ClassConstant>,
        method : &Context<'_, &MethodInfo>,
        suffix : &dyn Display,
    ) -> GeneralResult<String>
{
    Ok(format!(".L{}", mangle(&[ class, method, &format!("__{}", &suffix) ])?))
}

fn make_basic_block(
        class : &Context<'_, &ClassConstant>,
        method : &Context<'_, &MethodInfo>,
        list : impl IntoIterator<Item=InsnPair>,
        range : &Range<usize>
    ) -> GeneralResult<(tenyr::BasicBlock, BTreeSet<usize>)>
{
    use Destination::{Address, Successor};
    use tenyr::BasicBlock;

    let mut insns = Vec::with_capacity(range.len() * 2); // heuristic
    let mut exits = BTreeSet::new();

    let inside = |addr| addr >= range.start && addr < range.end;

    let mut includes_successor = false;
    for insn in list {
        let does_branch = |&e| if let Address(n) = e { Some(n) } else { None };

        let (ins, exs) = insn;

        // update the state of includes_successor each time so that the last instruction's behavior
        // is captured
        includes_successor = exs.iter().any(|e| if let Successor = e { true } else { false });

        exits.extend(exs.iter().filter_map(does_branch).filter(|&e| !inside(e)));
        insns.extend(ins);
    }
    let label = make_label(class, method, &range.start)?;

    if includes_successor {
        exits.insert(range.end);
    }

    insns.shrink_to_fit();

    Ok((BasicBlock { label, insns }, exits))
}

// The incoming StackManager represents a "prototype" StackManager which should be empty, and
// which will be cloned each time a new BasicBlock is seen.
fn make_blocks_for_method<'a, 'b>(
        class : &'a Context<'b, &'b ClassConstant>,
        method : &'a Context<'b, &'b MethodInfo>,
        sm : &StackManager,
        max_locals : u16,
    ) -> GeneralResult<Vec<tenyr::BasicBlock>>
{
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
        ) -> GeneralResult<Vec<tenyr::BasicBlock>>
    {
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
                .map(|x| make_instructions(&mut sm, x, &|y| make_label(class, method, y), class, max_locals))
                .collect();
        let (bb, ee) = make_basic_block(class, method, block?, which)?;
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
fn test_parse_classes() -> GeneralResult<()>
{
    fn parse_class(path : &std::path::Path) -> GeneralResult<ClassFile> {
        let p = path.with_extension("");
        let p = p.to_str().ok_or("bad path")?;
        classfile_parser::parse_class(p).map_err(Into::into)
    }

    fn test_stack_map_table(path : &std::path::Path) -> GeneralResult<()> {
        use util::get_constant_getter;

        let class = parse_class(path)?;
        let methods = class.methods.iter();
        for method in methods.filter(|m| !m.access_flags.contains(MethodAccessFlags::NATIVE)) {
            let sm = StackManager::new(STACK_REGS);
            let get_constant = get_constant_getter(&class);
            let class = get_class(get_constant, class.this_class)?;
            let max_locals = get_method_code(method)?.max_locals;
            let method = class.contextualize(method);
            let bbs = make_blocks_for_method(&class, &method, &sm, max_locals)?;
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

impl fmt::Display for Method {
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
    use super::GeneralResult;
    use super::JType;
    use std::convert::TryFrom;

    pub fn field_size(ch : char) -> GeneralResult<u8> {
        JType::try_from(ch).map(JType::size).map_err(Into::into)
    }

    fn count_internal(s : &str) -> GeneralResult<u8> {
        fn eat(s : &str) -> GeneralResult<usize> {
            let ch = s.chars().next().ok_or("string ended too soon")?;
            match ch {
                'B' | 'C' | 'F' | 'I' | 'S' | 'Z' | 'D' | 'J' | 'V' => Ok(1),
                'L' => Ok(1 + s.find(';').ok_or("string ended too soon")?),
                '[' => Ok(1 + eat(&s[1..])?),
                _ => Err(format!("unexpected character {}", ch).into()),
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
    pub fn count_params(descriptor : &str) -> GeneralResult<u8> {
        let open = 1; // byte index of open parenthesis is 0
        let close = descriptor.rfind(')').ok_or("descriptor missing closing parenthesis")?;
        count_internal(&descriptor[open..close])
    }

    // JVM limitations restrict the count of return values to 1 at most, of size 2 at most
    pub fn count_returns(descriptor : &str) -> GeneralResult<u8> {
        let close = descriptor.rfind(')').ok_or("descriptor missing closing parenthesis")?;
        count_internal(&descriptor[close+1..])
    }
}

#[test]
fn test_count_params() -> GeneralResult<()> {
    assert_eq!(3, count_params("(III)V")?);
    assert_eq!(4, count_params("(JD)I")?);
    assert_eq!(2, count_params("(Lmetasyntactic;Lvariable;)I")?);
    assert_eq!(1, count_params("([[[I)I")?);
    assert_eq!(0, count_params("()Lplaceholder;")?);
    assert_eq!(0, count_params("()D")?);
    Ok(())
}

#[test]
fn test_count_returns() -> GeneralResult<()> {
    assert_eq!(0, count_returns("(III)V")?);
    assert_eq!(1, count_returns("(JD)I")?);
    assert_eq!(1, count_returns("(Lmetasyntactic;Lvariable;)I")?);
    assert_eq!(1, count_returns("([[[I)I")?);
    assert_eq!(1, count_returns("()Lplaceholder;")?);
    assert_eq!(2, count_returns("()D")?);
    Ok(())
}

fn translate_method<'a, 'b>(
        class : &'a Context<'b, &'b ClassConstant>,
        method : &'a Context<'b, &'b MethodInfo>,
    ) -> GeneralResult<Method>
{
    use util::get_string;

    let mr = method.as_ref();
    let total_locals = get_method_code(mr)?.max_locals;
    let descriptor = get_string(class, mr.descriptor_index).ok_or("method descriptor missing")?;
    let num_returns = count_returns(&descriptor)?.into();
    // Pretend we have at least as many locals as we have return-slots, so we have somewhere to
    // store our results when we Yield.
    let max_locals = total_locals.max(num_returns);

    let sm = &StackManager::new(STACK_REGS);
    let sp = sm.get_stack_ptr();
    let max_locals_i32 = i32::from(max_locals);

    let prologue = {
        let name = "prologue";
        let off = -(max_locals_i32 - i32::from(count_params(&descriptor)?) + i32::from(SAVE_SLOTS));
        let insns = vec![ tenyr_insn!( sp <-  sp + (off) )? ];
        let label = make_label(class, method, &name)?;
        tenyr::BasicBlock { label, insns }
    };

    let epilogue = {
        let name = "epilogue";
        let off = i32::from(SAVE_SLOTS) + i32::from(total_locals) - i32::from(num_returns);
        let down = i32::from(num_returns) - max_locals_i32;
        let rp = Register::P;
        let mv = if off != 0 { Some(tenyr_insn!( sp <-  sp + (off) )?) } else { None };
        let insns = mv.into_iter().chain(std::iter::once(tenyr_insn!( rp <- [sp + (down)] )?)).collect();
        let label = make_label(class, method, &name)?;
        tenyr::BasicBlock { label, insns }
    };

    let blocks = make_blocks_for_method(class, method, sm, max_locals)?;
    let name = mangle(&[ class, method ])?;
    Ok(Method { name, prologue, blocks, epilogue })
}

fn write_method_table(
        class : &Context<'_, &ClassConstant>,
        methods : &[MethodInfo],
        outfile : &mut dyn Write,
    ) -> GeneralResult<()>
{
    let label = ".Lmethod_table";
    writeln!(outfile, "{}:", label)?;

    let names = methods.iter().map(|method| Ok(mangle(&[ class, &class.contextualize(method) ])?) );
    let lengths : GeneralResult<Vec<_>> =
        names.map(|s : GeneralResult<String>| {
            let s = s?;
            let len = s.len();
            Ok((s, len))
        }).collect();
    let lengths = lengths?;
    let width = lengths.iter().fold(0, |c, (_, len)| c.max(*len));

    for (method, (mangled_name, _)) in methods.iter().zip(lengths) {
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
    ) -> GeneralResult<()>
{
    let non_virtual = MethodAccessFlags::STATIC | MethodAccessFlags::PRIVATE;

    let virtuals = methods.iter().filter(|m| (m.access_flags & non_virtual).is_empty());
    let names = virtuals.map(|m| Ok(mangle(&[ class, &class.contextualize(m), &"vslot" ])?) );
    let lengths : GeneralResult<Vec<_>> =
        names.map(|s : GeneralResult<String>| {
            let s = s?;
            let len = s.len();
            Ok((s, len))
        }).collect();
    let lengths = lengths?;
    let width = lengths.iter().fold(0, |c, (_, len)| c.max(*len));

    for (index, (mangled_name, _)) in lengths.iter().enumerate() {
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
    ) -> GeneralResult<()>
{
    use util::get_string;

    let tuples = fields.iter().filter(selector).map(|f| {
        let s = get_string(class, f.descriptor_index).ok_or("missing descriptor")?;
        let desc = s.chars().next().ok_or("empty descriptor")?;
        let size = args::field_size(desc)?.into();
        let name = mangle(&[ class, &class.contextualize(f), &suff ])?;
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
    use util::get_constant_getter;

    if class.major_version < 50 {
        return Err("need classfile version ≥50.0 for StackMapTable attributes".into());
    }

    let fields = &class.fields;
    let methods = &class.methods;
    let get_constant = get_constant_getter(&class);
    let class = get_class(get_constant, class.this_class)?;

    write_method_table(&class, methods, outfile)?;
    write_vslot_list(&class, methods, outfile)?;

    let is_static = |f : &&FieldInfo| f.access_flags.contains(FieldAccessFlags::STATIC);
    let print_field = |outfile : &mut dyn Write, slot_name : &str, offset, _size, width| {
        writeln!(outfile, "    .global {}", slot_name)?;
        writeln!(outfile, "    .set    {:width$}, {}", slot_name, offset, width=width)?;
        Ok(())
    };
    write_field_list(&class, fields, outfile, "field_offset", |f| ! is_static(f), print_field)?;
    let print_static = |outfile : &mut dyn Write, slot_name : &str, _offset, size, width| {
        writeln!(outfile, "    .global {}", slot_name)?;
        writeln!(outfile, "    {:width$}: .zero {}", slot_name, size, width=width)?;
        Ok(())
    };
    write_field_list(&class, fields, outfile, "static", &is_static, &print_static)?;

    for method in methods.iter().filter(|m| !m.access_flags.contains(MethodAccessFlags::NATIVE)) {
        let method = class.contextualize(method);
        let mm = translate_method(&class, &method)?;
        writeln!(outfile, "{}", mm)?;
    }

    Ok(())
}

