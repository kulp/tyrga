#![deny(clippy::items_after_statements)]
#![deny(clippy::needless_borrow)]
#![deny(clippy::option_unwrap_used)]
#![deny(clippy::redundant_field_names)]
#![deny(clippy::result_unwrap_used)]
#![deny(unconditional_recursion)]

mod exprtree;
mod jvmtypes;
pub mod mangling;
mod stack;
#[macro_use] mod tenyr;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::io::Write;
use std::ops::Range;

#[cfg(test)]
use std::path::Path;
#[cfg(test)]
use walkdir::WalkDir;

use classfile_parser::attribute_info::CodeAttribute;
use classfile_parser::attribute_info::StackMapFrame;
use classfile_parser::constant_info::*;
use classfile_parser::field_info::FieldAccessFlags;
use classfile_parser::field_info::FieldInfo;
use classfile_parser::method_info::MethodAccessFlags;
use classfile_parser::method_info::MethodInfo;

use args::*;
use jvmtypes::*;
use stack::*;
use tenyr::{Instruction, Register, SmallestImmediate};
use util::*;

const STACK_PTR : Register = Register::O;
const STACK_REGS : &[Register] = { use Register::*; &[ B, C, D, E, F, G, H, I, J, K, L, M, N ] };

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Destination {
    Successor,
    Address(usize),
}

pub trait Named     { fn name_index(&self)       -> u16; }
pub trait Described { fn descriptor_index(&self) -> u16; }

impl Named     for MethodInfo { fn name_index(&self)       -> u16 { self.name_index } }
impl Described for MethodInfo { fn descriptor_index(&self) -> u16 { self.descriptor_index } }

impl Named     for FieldInfo  { fn name_index(&self)       -> u16 { self.name_index } }
impl Described for FieldInfo  { fn descriptor_index(&self) -> u16 { self.descriptor_index } }

impl Named for ClassConstant { fn name_index(&self) -> u16 { self.name_index } }

impl Named     for NameAndTypeConstant { fn name_index(&self)       -> u16 { self.name_index } }
impl Described for NameAndTypeConstant { fn descriptor_index(&self) -> u16 { self.descriptor_index } }

// TODO deduplicate this implementation with the one for &dyn Named
impl Manglable for Context<'_, &ClassConstant> {
    fn pieces(&self) -> GeneralResult<Vec<String>> {
        let r : GeneralResult<String> = get_string(self, self.as_ref().name_index()).ok_or_else(|| "no name".into());
        Ok(vec![ r? ])
    }
}

impl Manglable for &str {
    fn pieces(&self) -> GeneralResult<Vec<String>> { Ok(vec![ self.to_string() ]) }
}

impl Manglable for Context<'_, &FieldRefConstant> {
    fn pieces(&self) -> GeneralResult<Vec<std::string::String>> {
        use classfile_parser::constant_info::ConstantInfo::*;

        let fr = self.as_ref();
        if let Class(ni) = self.get_constant(fr.class_index) {
            let ni = ni.name_index;
            let ss = get_string(self, ni).ok_or("no such name")?;
            if let NameAndType(nt) = self.get_constant(fr.name_and_type_index) {
                let nt = self.contextualize(nt);
                Ok(std::iter::once(ss).chain(nt.pieces()?.into_iter()).collect())
            } else {
                Err("invalid ConstantInfo kind".into())
            }
        } else {
            Err("invalid ConstantInfo kind".into())
        }
    }
}

fn expand_immediate_load(sm : &mut stack::Manager, insn : Instruction, imm : i32)
    -> GeneralResult<Vec<Instruction>>
{
    use tenyr::InsnGeneral;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::*;
    use SmallestImmediate::*;

    fn make_imm(temp_reg : Register, imm : SmallestImmediate) -> GeneralResult<Vec<Instruction>> {
        use tenyr::Register::A;

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
        (Type0(g) , Imm12(imm)) => vec![ Instruction { kind : Type0(InsnGeneral { imm, ..g }), ..insn } ],
        (Type1(g) , Imm12(imm)) => vec![ Instruction { kind : Type1(InsnGeneral { imm, ..g }), ..insn } ],
        (Type2(g) , Imm12(imm)) => vec![ Instruction { kind : Type2(InsnGeneral { imm, ..g }), ..insn } ],

        (kind, imm) => {
            use std::iter::once;
            use tenyr::Opcode::*;

            let adder  = InsnGeneral { y : Register::A, op : Add , imm : 0_u8.into() };
            let reserve = sm.reserve(1).into_iter();
            let temp = sm.get(0).ok_or("stack unexpectedly empty")?;
            let pack = make_imm(temp, imm)?.into_iter();
            let (op, a, b, c) = match kind {
                Type3(_) => (BitwiseOr, insn.x, Register::A, temp), // should never be reached, but provides generality
                Type0(g) => (g.op, insn.x, g.y, temp),
                Type1(g) => (g.op, insn.x, temp, g.y),
                Type2(g) => (g.op, temp, insn.x, g.y),
            };
            let operate = once(Instruction { kind : Type0(InsnGeneral { op, y : b, imm : 0_u8.into() }), x : a, dd : NoLoad, z : insn.z });
            let add = once(Instruction { kind : Type0(InsnGeneral { y : c, ..adder }), x : insn.z, ..insn });
            let release = sm.release(1).into_iter();

            reserve
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
    use tenyr::*;
    use Register::*;
    use InstructionType::*;

    let v = vec![ C, D, E, F, G ];
    let mut sm = stack::Manager::new(5, O, v.clone());

    {
        let imm = 867_5309;
        let insn = tenyr_insn!( D -> [C * B] )?;
        let vv = expand_immediate_load(&mut sm, insn, imm)?;
        eprintln!("{:?}", vv);
        assert_eq!(vv.len(), 4);
    }

    {
        let imm = 123;
        let insn = tenyr_insn!( D -> [C + 0] )?;
        let vv = expand_immediate_load(&mut sm, insn.clone(), imm)?;
        eprintln!("{:?}", vv);
        assert_eq!(vv.len(), 1);
        // TODO more robust test
    }

    {
        let imm = 867_5309;
        let insn = tenyr_insn!( D -> [C + 0] )?;
        let vv = expand_immediate_load(&mut sm, insn.clone(), imm)?;
        eprintln!("{:?}", vv);
        assert_eq!(vv.len(), 4);
        // TODO more robust test
    }

    {
        let imm = 123;
        let insn = tenyr_insn!( D -> [C * B] )?;
        let vv = expand_immediate_load(&mut sm, insn.clone(), imm)?;
        eprintln!("{:?}", vv);
        assert_eq!(vv.len(), 1);
        if let Type0(ref g) = vv[0].kind {
            assert_eq!(g.imm, 123_u8.into());
        } else {
            return Err("wrong type".into());
        }
    }

    Ok(())
}

type Namer<'a> = dyn Fn(&dyn fmt::Display) -> GeneralResult<String> + 'a;
type InsnTriple = (usize, Vec<Instruction>, Vec<Destination>);
type MakeInsnResult = GeneralResult<InsnTriple>;

fn make_target(target : &dyn std::string::ToString) -> GeneralResult<exprtree::Atom> {
    use exprtree::Atom::*;
    use exprtree::Expr;
    use exprtree::Operation::*;
    use std::rc::Rc;

    let a = Variable(target.to_string());
    let b = Expression(Rc::new(Expr { a : Variable(".".to_owned()), op : Add, b : Immediate(1) }));
    Ok(Expression(Rc::new(Expr { a, op : Sub, b })))
}

fn make_int_branch(
        sm : &mut stack::Manager,
        addr : usize,
        invert : bool,
        target : u16,
        target_namer : &Namer,
        mut comp : impl FnMut(&mut stack::Manager) -> GeneralResult<(tenyr::Register, Vec<Instruction>)>
    ) -> MakeInsnResult
{
    use tenyr::*;
    use tenyr::Register::P;

    let o = make_target(&target_namer(&target)?)?;

    let (temp_reg, sequence) = comp(sm)?;
    let imm = tenyr::Immediate::Expr(o);
    let branch =
        if invert   { tenyr_insn!(   P <- (imm) &~ temp_reg + P     ) }
        else        { tenyr_insn!(   P <- (imm) &  temp_reg + P     ) };
    let mut v = sequence;
    v.push(branch?);
    let dest = vec![
        Destination::Successor,
        Destination::Address(target.into()),
    ];
    Ok((addr, v, dest))
}

fn make_instructions<'a, T>(
        sm : &mut stack::Manager,
        (addr, op) : (&usize, &Operation),
        target_namer : &Namer,
        gc : &T,
    ) -> MakeInsnResult
    where T : ContextConstantGetter<'a> + Contextualizer<'a>
{
    use Operation::*;
    use jvmtypes::SwitchParams::*;
    use std::convert::TryInto;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::*;

    // We need to track destinations and return them so that the caller can track stack state
    // through the chain of control flow, possibly cloning the stack::Manager state along the way to
    // follow multiple destinations. Each basic block needs to be visited only once, however, since
    // the JVM guarantees that every instance of every instruction within a method always sees the
    // same depth of the operand stack every time that instance is executed.
    let default_dest = vec![Destination::Successor];

    let get_reg = |t : Option<_>| t.ok_or("asked but did not receive");

    let translate_arithmetic_op =
        |x| {
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

    let make_mov  = |to, from| Instruction { z : to, x : from, ..tenyr::NOOP_TYPE3 };
    let make_load = |to, from| Instruction { dd : LoadRight , ..make_mov(to, from) };

    let make_jump = |target| {
        let result : GeneralResult<Instruction> = Ok(Instruction {
            kind : Type3(tenyr::Immediate::Expr(make_target(&target_namer(&target)?)?)),
            ..make_mov(tenyr::Register::P, tenyr::Register::P)
        });
        result
    };

    let make_call = |sm : &mut stack::Manager, target : &str, descriptor| {
        use Register::P;

        let mut insns = Vec::new();
        insns.extend(sm.freeze());

        // Save return address into bottom of register-based stack
        let bottom = sm.get_regs()[0];
        insns.push(Instruction {
            kind : Type3(1_u8.into()),
            ..make_mov(bottom, tenyr::Register::P)
        });

        let far = format!("@+{}", target);
        let off : tenyr::Immediate20 = tenyr::Immediate::Expr(exprtree::Atom::Variable(far));
        insns.push( tenyr_insn!( P <- P + (off) )? );

        // adjust stack for returned values
        sm.release_frozen(count_params(descriptor)?.into());
        sm.reserve_frozen(count_returns(descriptor)?.into());
        insns.extend(sm.thaw());
        Ok((*addr, insns, default_dest.clone()))
    };

    let name_op = |op| {
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
    };
    let make_arithmetic_descriptor = |kind : JType, op| {
        let ch : char = kind.try_into()?;

        let nargs = {
            use ArithmeticOperation::*;
            match op {
                Add | Sub | Mul | Div | Rem | Shl | Shr | Ushr | And | Or | Xor => 2,
                Neg => 1,
            }
        };
        let result = format!("({}){}", std::iter::repeat(ch).take(nargs).collect::<String>(), ch);
        Ok(result) as GeneralResult<String>
    };
    let make_builtin_name = move |proc : &str, descriptor : &str| {
        mangle(&[ &"tyrga/Builtin", &proc, &descriptor ])
    };
    let make_arithmetic_name = |kind, op| {
        let descriptor = make_arithmetic_descriptor(kind, op)?;
        let proc = name_op(op).to_lowercase();
        make_builtin_name(&proc, &descriptor)
    };

    let mut make_constant = |slice : &[i32]| {
        let f = |v : GeneralResult<Vec<_>>, &value| {
            let mut v = v?;
            v.extend(sm.reserve(1));
            let insn = make_mov(get_reg(sm.get(0))?, Register::A);
            v.extend(expand_immediate_load(sm, insn, value)?);
            Ok(v)
        };
        let v = slice.iter().fold(Ok(vec![]), f)?;
        Ok((*addr, v, default_dest.clone()))
    };

    match op.clone() { // TODO obviate clone
        Constant { kind : JType::Object, value } |
        Constant { kind : JType::Int   , value } => make_constant(&[ value.into() ]),
        Constant { kind : JType::Long  , value } => make_constant(&[ 0, value.into() ]),
        Constant { kind : JType::Float , value } => make_constant(&[ f32::from(value).to_bits() as i32 ]),
        Constant { kind : JType::Double, value } => {
            let bits = f64::from(value).to_bits();
            make_constant(&[ (bits >> 32) as i32, bits as i32 ])
        },
        Constant { .. } =>
            Err("encountered impossible Constant configuration".into()),
        Yield { kind } => {
            use Register::P;
            let mut v = Vec::new();
            for i in 0 .. kind.size() {
                v.push(store_local(sm, get_reg(sm.get(i.into()))?, i.into()));
            }
            v.extend(sm.empty());
            let ex = tenyr::Immediate::Expr(make_target(&target_namer(&"epilogue")?)?);
            v.push(tenyr_insn!( P <- (ex) + P )?);

            Ok((*addr, v, vec![])) // leaving the method is not a Destination we care about
        },
        Arithmetic { kind : JType::Int, op : ArithmeticOperation::Neg } => {
            use tenyr::*;
            use Register::A;
            let y = get_reg(sm.get(0))?;
            let v = vec![ tenyr_insn!( y <- A - y )? ];
            Ok((*addr, v, default_dest))
        },
        Arithmetic { kind : JType::Int, op } if translate_arithmetic_op(op).is_some() => {
            use tenyr::*;
            let y = get_reg(sm.get(0))?;
            let x = get_reg(sm.get(1))?;
            let z = x;
            let op = translate_arithmetic_op(op).ok_or("no op for this opcode")?;
            let dd = MemoryOpType::NoLoad;
            let imm = 0_u8.into();
            let mut v = Vec::new();
            v.push(Instruction { kind : Type0(InsnGeneral { y, op, imm }), x, z, dd });
            v.extend(sm.release(1));
            Ok((*addr, v, default_dest))
        },
        Arithmetic { kind, op } =>
            make_call(sm, &make_arithmetic_name(kind, op)?, &make_arithmetic_descriptor(kind, op)?),
        LoadLocal { kind, index } => {
            let mut v = Vec::new();
            let size = kind.size().into();
            v.extend(sm.reserve(size));
            for i in 0 .. size {
                v.push(load_local(sm, get_reg(sm.get(i))?, (index + i).into()));
            }
            Ok((*addr, v, default_dest))
        },
        StoreLocal { kind, index } => {
            let mut v = Vec::new();
            let size = kind.size().into();
            for i in 0 .. size {
                v.push(store_local(sm, get_reg(sm.get(i))?, (index + i).into()));
            }
            v.extend(sm.release(size));
            Ok((*addr, v, default_dest))
        },
        Increment { index, value } => {
            use tenyr::*;
            // This reserving of a stack slot may exceed the "maximum depth" statistic on the
            // method, but we should try to avoid dedicated temporary registers.
            let mut v = Vec::new();
            v.extend(sm.reserve(1));
            let temp_reg = get_reg(sm.get(0))?;
            let stack_ptr = sm.get_stack_ptr();
            let offset = sm.get_frame_offset(index.into());
            v.extend(tenyr_insn_list!(
                temp_reg <- [stack_ptr + (offset.clone())] ;
                temp_reg <- temp_reg + (value)             ;
                temp_reg -> [stack_ptr + (offset.clone())] ;
            ));
            v.extend(sm.release(1));

            Ok((*addr, v, default_dest))
        },
        Branch { kind : JType::Object, ops, way, target } |
        Branch { kind : JType::Int   , ops, way, target } => {
            use tenyr::*;

            let (op, swap, invert) = match way {
                jvmtypes::Comparison::Eq => (Opcode::CompareEq, false, false),
                jvmtypes::Comparison::Ne => (Opcode::CompareEq, false, true ),
                jvmtypes::Comparison::Lt => (Opcode::CompareLt, false, false),
                jvmtypes::Comparison::Ge => (Opcode::CompareGe, false, false),
                jvmtypes::Comparison::Gt => (Opcode::CompareLt, true , false),
                jvmtypes::Comparison::Le => (Opcode::CompareGe, true , false),
            };

            let opper = move |sm : &mut stack::Manager| {
                let count = ops as u16;
                let lhs = get_reg(sm.get(count - 1))?;
                let rhs = if ops == OperandCount::_2 { get_reg(sm.get(0))? } else { Register::A };
                let temp_reg = lhs;
                let (rhs, lhs) = if swap { (lhs, rhs) } else { (rhs, lhs) };

                let mut v = sm.release(count);
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

            make_int_branch(sm, *addr, invert, target, target_namer, opper)
        },
        Branch { .. } =>
            Err("encountered impossible Branch configuration".into()),
        Switch(Lookup { default, pairs }) => {
            use tenyr::*;

            let here = *addr as i32;
            let far = (default + here) as u16;

            let mut dests = Vec::new();
            let top = get_reg(sm.get(0))?;
            let mut insns = sm.reserve(1); // need a persistent temporary
            let temp_reg = get_reg(sm.get(0))?;

            let maker = |imm : i32| {
                move |sm : &mut stack::Manager| {
                    let insn = tenyr_insn!( temp_reg <- top == 0 );
                    let insns = expand_immediate_load(sm, insn?, imm)?;
                    Ok((temp_reg, insns))
                }
            };

            let brancher = |(compare, target)| {
                let result = make_int_branch(sm, *addr, false, (target + here) as u16, target_namer, maker(compare));
                let (_, insns, dests) = result?;
                Ok((insns, dests)) as GeneralResult<(_,_)>
            };
            let (i, d) : (Vec<_>, Vec<_>) =
                pairs
                    .into_iter()
                    .map(brancher)
                    .collect::<Result<Vec<_>,_>>()?
                    .into_iter()
                    .unzip();

            let i = i.concat();
            let d = d.concat();

            insns.extend(i);
            dests.extend(d);

            insns.push(make_jump(far)?);
            dests.push(Destination::Address(far.into()));

            Ok((*addr, insns, dests))
        },
        Switch(Table { default, low, high, offsets }) => {
            use tenyr::*;
            use tenyr::InstructionType::*;
            type InsnType = dyn Fn(InsnGeneral) -> InstructionType;

            let here = *addr as i32;
            let far = (default + here) as u16;

            let mut dests = Vec::new();
            let top = get_reg(sm.get(0))?;
            let mut insns = sm.reserve(1); // need a persistent temporary
            let temp_reg = get_reg(sm.get(0))?;

            let maker = |kind : &'static InsnType, imm : i32| {
                move |sm : &mut stack::Manager| {
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

            let (lo_addr, lo_insns, lo_dests) =
                make_int_branch(sm, *addr, false, far, target_namer, maker(&Type1, low))?;
            let (      _, hi_insns, hi_dests) =
                make_int_branch(sm, *addr, false, far, target_namer, maker(&Type2, high))?;

            let addr = lo_addr;

            insns.extend(lo_insns);
            insns.extend(hi_insns);

            let insn = {
                use tenyr::Register::P;
                tenyr_insn!( P <- top - 0 + P )
            };
            insns.extend(expand_immediate_load(sm, insn?, low)?);

            let make_pairs = |n|
                Ok((make_jump((n + here) as u16)?, Destination::Address((n + here) as usize))) as GeneralResult<_>;
            let (i, d) : (Vec<_>, Vec<_>) =
                offsets
                    .into_iter()
                    .map(make_pairs)
                    .collect::<Result<Vec<_>,_>>()?
                    .into_iter()
                    .unzip();

            insns.extend(i);
            dests.extend(d);

            dests.extend(lo_dests);
            dests.extend(hi_dests);

            insns.extend(sm.release(1)); // release temporary

            Ok((addr, insns, dests))
        },
        Jump { target } => Ok((*addr, vec![ make_jump(target)? ], vec![ Destination::Address(target as usize) ])),
        LoadArray(kind) | StoreArray(kind) => {
            use tenyr::*;

            let mut v = Vec::new();
            let array_params = |sm : &mut stack::Manager, v : &mut Vec<Instruction>| {
                let idx = get_reg(sm.get(0))?;
                let arr = get_reg(sm.get(1))?;
                v.extend(sm.release(2));
                Ok((idx, arr)) as GeneralResult<(Register, Register)>
            };
            let (x, y, z, dd) = match *op {
                LoadArray(_) => {
                    let (idx, arr) = array_params(sm, &mut v)?;
                    v.extend(sm.reserve(1));
                    let res = get_reg(sm.get(0))?;
                    (idx, arr, res, LoadRight)
                },
                StoreArray(_) => {
                    let val = get_reg(sm.get(0))?;
                    v.extend(sm.release(1));
                    let (idx, arr) = array_params(sm, &mut v)?;
                    (idx, arr, val, StoreRight)
                },
                _ => unreachable!(),
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
            Ok((*addr, v, default_dest))
        },
        Noop => Ok((*addr, vec![ tenyr::NOOP_TYPE0 ], default_dest)),
        Length => {
            // TODO document layout of arrays
            // This implementation assumes a reference to an array points to its first element, and
            // that one word below that element is a word containing the number of elements.
            let top = get_reg(sm.get(0))?;
            let insn = Instruction { kind : Type3((-1_i8).into()), ..make_load(top, top) };
            Ok((*addr, vec![ insn ], default_dest))
        },
        // TODO fully handle Special (this is dumb partial handling)
        Invocation { kind : InvokeKind::Special, index } => {
            let (addr, mut insns, dest) =
                make_call(sm, &make_callable_name(gc, index)?, &get_method_parts(gc, index)?[2])?;
            insns.extend(sm.release(1));
            Ok((addr, insns, dest))
        },
        Invocation { kind : InvokeKind::Static, index } =>
            make_call(sm, &make_callable_name(gc, index)?, &get_method_parts(gc, index)?[2]),
        StackOp { op : StackOperation::Pop, size } => {
            let v = sm.release(size as u16);
            Ok((*addr, v, default_dest))
        },
        StackOp { op : StackOperation::Dup, size } => {
            let size = size as u16;
            let old : Vec<_> = (0..size).map(|i| get_reg(sm.get(i))).collect();
            let res : Vec<_> = (0..size).map(|_| sm.reserve(1)).flatten().collect();
            let put : GeneralResult<Vec<_>> = (0..size).map(|i| {
                let new = get_reg(sm.get(i))?;
                let t = old[i as usize]?;
                tenyr_insn!( new <- t )
            }).collect();
            let v = [ res, put? ].concat();
            Ok((*addr, v, default_dest))
        },
        ArrayAlloc { kind } => {
            let mut pre = match kind.size() {
                1 => Ok(vec![]),
                // insert an instruction that doubles the top-of-stack count
                2 => {
                    let top = get_reg(sm.get(0))?;
                    Ok(vec![ tenyr_insn!( top <- top + top )? ])
                },
                _ => Err("impossible size"),
            }?;
            let descriptor = "(I)Ljava.lang.Object;";
            let proc = "alloc";
            let name = make_builtin_name(proc, descriptor)?;
            let (addr, v, dest) = make_call(sm, &name, descriptor)?;
            pre.extend(v);
            Ok((addr, pre, dest))
        },
        Compare { kind, nans } => {
            let mut v = sm.reserve(1);
            let name = "cmp";
            let ch : char = kind.try_into()?;

            let gc = get_reg(sm.get(0))?;
            let n = match nans {
                Some(NanComparisons::Greater) => 1,
                Some(NanComparisons::Less) => -1,
                _ => 0,
            };
            v.push(tenyr_insn!( gc <- (n) )?);

            let desc = format!("({}{}I)I", ch, ch);
            let (addr, insns, dests) = make_call(sm, &make_builtin_name(name, &desc)?, &desc)?;
            v.extend(insns);
            Ok((addr, v, dests))
        },
        Conversion { from : JType::Int, to : JType::Byte } => {
            let top = get_reg(sm.get(0))?;
            let left  = tenyr_insn!( top <- top << 24 )?;
            let right = tenyr_insn!( top <- top >> 24 )?; // arithmetic shift, result is signed
            Ok((*addr, vec![ left, right ], default_dest ))
        },
        Conversion { from : JType::Int, to : JType::Short } => {
            let top = get_reg(sm.get(0))?;
            let left  = tenyr_insn!( top <- top << 16 )?;
            let right = tenyr_insn!( top <- top >> 16 )?; // arithmetic shift, result is signed
            Ok((*addr, vec![ left, right ], default_dest ))
        },
        Conversion { from : JType::Int, to : JType::Char } => {
            let top = get_reg(sm.get(0))?;
            let left  = tenyr_insn!( top <- top <<  16 )?;
            let right = tenyr_insn!( top <- top >>> 16 )?; // logical shift, result is positive
            Ok((*addr, vec![ left, right ], default_dest ))
        },
        Conversion { from, to } => {
            let ch_from : char = from.try_into()?;
            let ch_to   : char = to  .try_into()?;
            let name = format!("into_{}", ch_to); // TODO improve naming
            let desc = format!("({}){}", ch_from, ch_to);
            make_call(sm, &make_builtin_name(&name, &desc)?, &desc)
        },
        VarAction  { op, kind, index } => {
            use classfile_parser::constant_info::ConstantInfo::*;

            if let FieldRef(fr) = gc.get_constant(index) {
                use exprtree::Atom;
                use tenyr::MemoryOpType::*;

                let fr = gc.contextualize(fr);
                let mut insns = Vec::new();

                let len = util::field_type(&fr)?.size();

                let make_off = |base, i| {
                    use exprtree::Operation::Add;
                    use std::rc::Rc;
                    use tenyr::Immediate;
                    use tenyr::TwentyBit;

                    let e = exprtree::Expr { a : base, b : Atom::Immediate(i), op : Add };
                    Immediate::Expr(Atom::Expression(Rc::new(e))) as Immediate<TwentyBit>
                };

                let op_depth = match op { VarOp::Get => 0, VarOp::Put => 1 };

                let format = |suff|
                    Ok(format!("@{}", mangle(&[ &fr, &suff ])?))
                        as GeneralResult<std::string::String>;

                let (reg, base) = match kind {
                    VarKind::Static =>
                        (   Register::P,
                            make_target(&format("static")?)?
                        ),
                    VarKind::Field =>
                        (   get_reg(sm.get((op_depth * len).into()))?,
                            Atom::Variable(format("field_offset")?)
                        ),
                };

                let mut range = 0i32..len.into();
                let mut reversed = range.clone().rev();
                let (prior, post, memop, iter) : (_, _, _, &mut Iterator<Item=_>) = match op {
                    VarOp::Get => (1, 0, LoadRight , &mut range   ),
                    VarOp::Put => (0, 1, StoreRight, &mut reversed),
                };

                for it in iter {
                    let imm = make_off(base.clone(), it);
                    insns.extend(sm.reserve(prior));
                    let top = get_reg(sm.get(0))?;
                    insns.push(Instruction { dd : memop, ..tenyr_insn!( top <- [reg + (imm)] )? });
                    insns.extend(sm.release(post));
                }

                // Drop object reference
                if kind == VarKind::Field {
                    insns.extend(sm.release(1));
                }

                Ok((*addr, insns, default_dest))
            } else {
                Err("invalid ConstantInfo kind".into())
            }
        },
        Allocation { index } => {
            let class = gc.get_constant(index);
            if let ConstantInfo::Class(cc) = class {
                let name = get_string(gc, cc.name_index).ok_or("no class name")?;
                let desc = format!("()L{};", name);
                let call = mangle(&[ &&*name, &"new" ])?;
                make_call(sm, &call, &desc)
            } else {
                Err("invalid ConstantInfo kind".into())
            }
        },

        Invocation { .. } |
        StackOp    { .. } |
        Unhandled  ( .. ) =>
            Err(format!("unhandled operation {:?}", op).into()),
    }
}

#[test]
fn test_make_instruction() -> GeneralResult<()> {
    use Instruction;
    use Register::A;
    use classfile_parser::constant_info::ConstantInfo;
    use classfile_parser::constant_info::ConstantInfo::Unusable;
    use tenyr::InstructionType::Type3;
    use tenyr::MemoryOpType::NoLoad;
    struct Useless;
    impl<'a> ContextConstantGetter<'a> for Useless {
        fn get_constant(&self, _ : u16) -> &'a ConstantInfo {
            &Unusable
        }
    }
    impl<'a> Contextualizer<'a> for Useless {
        fn contextualize<U>(&self, _ : U) -> Context<'a, U> {
            unreachable!("this code is for testing only")
        }
    }

    let mut sm = stack::Manager::new(5, STACK_PTR, STACK_REGS.to_owned());
    let op = Operation::Constant { kind : JType::Int, value : 5 };
    let namer = |x : &dyn fmt::Display| Ok(format!("{}:{}", "test", x.to_string()));
    let insn = make_instructions(&mut sm, (&0, &op), &namer, &Useless)?;
    let imm = 5_u8.into();
    assert_eq!(insn.1, vec![ Instruction { kind : Type3(imm), z : STACK_REGS[0], x : A, dd : NoLoad } ]);
    assert_eq!(insn.1[0].to_string(), " B  <-  5");

    Ok(())
}

pub type GeneralResult<T> = std::result::Result<T, Box<dyn Error>>;

fn generic_error(e : impl Error) -> Box<dyn Error> {
    format!("unknown error: {}", e).into()
}

#[cfg(test)]
fn parse_class(path : &Path) -> GeneralResult<classfile_parser::ClassFile> {
    let p = path.with_extension("");
    let p = p.to_str().ok_or("bad path")?;
    classfile_parser::parse_class(p).map_err(Into::into)
}

type RangeMap<T> = (Vec<Range<usize>>, BTreeMap<usize, T>);

fn derive_ranges<T>(body : Vec<(usize, T)>, table : &[StackMapFrame])
    -> GeneralResult<RangeMap<T>>
{
    use classfile_parser::attribute_info::StackMapFrame::*;
    let get_delta = |f : &StackMapFrame| match *f {
        SameFrame                           { frame_type }       => frame_type.into(),

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
    let max = body.last().ok_or("body unexpectedly empty")?.0 + 1;

    #[allow(clippy::len_zero)] // is_empty is ambiguous for Range at the time of this writing
    let ranges =
        std::iter::once(0)
            .chain(before.cloned())
            .chain(std::iter::once(0)).chain(after.map(|&n| n + 1))
            .scan(0, |state, x| { *state += x; Some(usize::from(*state)) })
            .chain(std::iter::once(max))
            .collect::<Vec<_>>()
            .windows(2)
            .map(|x| x[0]..x[1])
            .filter(|x| x.len() > 0)
            .collect::<Vec<_>>();

    let tree = body.into_iter().collect();
    Ok((ranges, tree))
}

fn get_method_code(method : &MethodInfo) -> GeneralResult<CodeAttribute> {
    use classfile_parser::attribute_info::code_attribute_parser;
    Ok(code_attribute_parser(&method.attributes[0].info).map_err(generic_error)?.1)
}

mod util {
    use classfile_parser::ClassFile;
    use classfile_parser::constant_info::ConstantInfo;
    use classfile_parser::constant_info::FieldRefConstant;
    use std::convert::TryFrom;
    use std::rc::Rc;
    use super::GeneralResult;
    use super::jvmtypes::JType;
    use super::mangling;
    use super::stack;
    use super::tenyr::Instruction;
    use super::tenyr::MemoryOpType;
    use super::tenyr::Register;
    use super::{Described, Named};

    pub type ConstantGetter<'a> = dyn Fn(u16) -> &'a ConstantInfo + 'a;

    pub trait Contextualizer<'a> {
        fn contextualize<U>(&self, nested : U) -> Context<'a, U>;
    }

    pub trait ContextConstantGetter<'a> {
        fn get_constant(&self, index : u16) -> &'a ConstantInfo;
    }

    pub struct Context<'a, T> {
        get_constant : Rc<ConstantGetter<'a>>,
        nested : Rc<T>,
    }

    impl<'a, T> Clone for Context<'a, T> {
        fn clone(&self) -> Self {
            Context {
                get_constant : Rc::clone(&self.get_constant),
                nested : Rc::clone(&self.nested),
            }
        }
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

    pub fn get_constant_getter<'a>(nested : &'a ClassFile) -> Context<'a, &ClassFile> {
        let gc = move |n| &nested.const_pool[usize::from(n) - 1];
        Context { get_constant : Rc::new(gc), nested : Rc::new(nested) }
    }

    pub fn get_string(g : &dyn ContextConstantGetter, i : u16) -> Option<String> {
        if let classfile_parser::constant_info::ConstantInfo::Utf8(u) = g.get_constant(i) {
            Some(u.utf8_string.to_string())
        } else {
            None
        }
    }

    pub trait Manglable {
        fn pieces(&self) -> GeneralResult<Vec<String>>;
        fn stringify(&self) -> GeneralResult<String> {
            Ok(self.pieces()?.join(":"))
        }
        fn mangle(&self) -> GeneralResult<String> {
            mangling::mangle(self.stringify()?.bytes())
        }
    }

    impl<T> Manglable for Context<'_, &T>
        where T : Named + Described
    {
        fn pieces(&self) -> GeneralResult<Vec<String>> {
            Ok(vec![
                get_string(self, self.as_ref().name_index()      ).ok_or("no name")?,
                get_string(self, self.as_ref().descriptor_index()).ok_or("no desc")?,
            ])
        }
    }

    impl Manglable for Context<'_, &dyn Named> {
        fn pieces(&self) -> GeneralResult<Vec<String>> {
            let r : GeneralResult<String> = get_string(self, self.as_ref().name_index()).ok_or_else(|| "no name".into());
            Ok(vec![ r? ])
        }
    }

    impl Manglable for &[&dyn Manglable] {
        fn pieces(&self) -> GeneralResult<Vec<String>> {
            self.iter().map(|x| x.stringify()).collect() // TODO flatten
        }
    }

    pub fn field_type(fr : &Context<'_, &FieldRefConstant>) -> GeneralResult<JType> {
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

    pub fn index_local(sm : &stack::Manager, reg : Register, idx : i32) -> Instruction {
        use super::tenyr::InstructionType::Type3;
        let x = sm.get_stack_ptr();
        Instruction { dd : MemoryOpType::NoLoad, z : reg, x, kind : Type3(sm.get_frame_offset(idx)) }
    }
    pub fn load_local(sm : &stack::Manager, reg : Register, idx : i32) -> Instruction {
        Instruction { dd : MemoryOpType::LoadRight, ..index_local(sm, reg, idx) }
    }
    pub fn store_local(sm : &stack::Manager, reg : Register, idx : i32) -> Instruction {
        Instruction { dd : MemoryOpType::StoreRight, ..index_local(sm, reg, idx) }
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
    let ops = map.into_iter().map(decode_insn).collect::<BTreeMap<_,_>>();
    Ok((ranges, ops))
}

fn get_method_parts(g : &dyn ContextConstantGetter, pool_index : u16) -> GeneralResult<[String ; 3]> {
    use classfile_parser::constant_info::ConstantInfo::*;

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
    let joined = parts.join(":");
    mangling::mangle(joined.bytes())
}

fn mangle(list : &[&dyn Manglable]) -> GeneralResult<String> {
    list.mangle()
}

fn get_class<'a, T>(ctx : util::Context<'a, T>, index : u16) -> GeneralResult<Context<'a, &'a ClassConstant>> {
    match ctx.get_constant(index) {
        classfile_parser::constant_info::ConstantInfo::Class(cl) => Ok(ctx.contextualize(cl)),
        _ => Err("not a class".into()),
    }
}

fn make_label(class : &Context<'_, &ClassConstant>, method : &Context<'_, &MethodInfo>, suffix : &dyn Display) -> GeneralResult<String> {
    Ok(format!(".L{}", mangle(&[ class, method, &&*format!("__{}", &suffix) ])?))
}

fn make_basic_block(
        class : &Context<'_, &ClassConstant>,
        method : &Context<'_, &MethodInfo>,
        list : impl IntoIterator<Item=InsnTriple>,
        range : &Range<usize>
    ) -> GeneralResult<(tenyr::BasicBlock, BTreeSet<usize>)>
{
    use Destination::*;
    use tenyr::BasicBlock;

    let mut insns = Vec::with_capacity(range.len() * 2); // heuristic
    let mut exits = BTreeSet::new();

    let inside = |addr| addr >= range.start && addr < range.end;

    let mut includes_successor = false;
    for insn in list {
        let does_branch = |&e| if let Address(n) = e { Some(n) } else { None };

        let (_, ins, exs) = insn;

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

// The incoming stack::Manager represents a "prototype" stack::Manager which should be empty, and which
// will be cloned each time a new BasicBlock is seen.
fn make_blocks_for_method<'a, 'b>(class : &'a Context<'b, &'b ClassConstant>, method : &'a Context<'b, &'b MethodInfo>, sm : &stack::Manager)
    -> GeneralResult<Vec<tenyr::BasicBlock>>
{
    use std::iter::FromIterator;

    struct Params<'a, 'b> {
        class : &'a Context<'b, &'b ClassConstant>,
        method : &'a Context<'b, &'b MethodInfo>,
        rangemap : &'a BTreeMap<usize, Range<usize>>,
        ops : &'a BTreeMap<usize, Operation>,
    }

    fn make_blocks(params : &Params, seen : &mut HashSet<usize>, mut sm : stack::Manager, which : &Range<usize>) -> GeneralResult<Vec<tenyr::BasicBlock>> {
        let (class, method, rangemap, ops) =
            (params.class, params.method, params.rangemap, params.ops);
        if seen.contains(&which.start) {
            return Ok(vec![]);
        }
        seen.insert(which.start);

        let block : GeneralResult<Vec<_>> =
            ops .range(which.clone())
                .map(|x| make_instructions(&mut sm, x, &|y| make_label(class, method, y), class))
                .collect();
        let (bb, ee) = make_basic_block(class, method, block?, which)?;
        let mut out = Vec::new();
        out.push(bb);

        for exit in &ee {
            out.extend(make_blocks(params, seen, sm.clone(), &rangemap[exit])?); // intentional clone of stack::Manager
        }

        Ok(out)
    }

    let (ranges, ops) = get_ranges_for_method(method)?;
    let rangemap = &BTreeMap::from_iter(ranges.into_iter().map(|r| (r.start, r)));
    let ops = &ops;

    let params = Params { class, method, rangemap, ops };

    let mut seen = HashSet::new();

    make_blocks(&params, &mut seen, sm.clone(), &rangemap[&0]) // intentional clone of stack::Manager
}

#[cfg(test)]
fn test_stack_map_table(path : &Path) -> GeneralResult<()> {
    let class = parse_class(path)?;
    for method in class.methods.iter().filter(|m| !m.access_flags.contains(MethodAccessFlags::NATIVE)) {
        let sm = stack::Manager::new(5, STACK_PTR, STACK_REGS.to_owned());
        let get_constant = get_constant_getter(&class);
        let class = get_class(get_constant, class.this_class)?;
        let method = class.contextualize(method);
        let bbs = make_blocks_for_method(&class, &method, &sm)?;
        for bb in &bbs {
            eprintln!("{}", bb);
        }
    }

    Ok(())
}

#[test]
fn test_parse_classes() -> GeneralResult<()>
{
    let is_dir_or_class = |e : &walkdir::DirEntry| {
        e.metadata().map(|e| e.is_dir()).unwrap_or(false) ||
            e.file_name().to_str().map(|s| s.ends_with(".class")).unwrap_or(false)
    };
    for class in WalkDir::new(env!("OUT_DIR")).into_iter().filter_entry(is_dir_or_class) {
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
    use std::convert::TryFrom;
    use super::GeneralResult;
    use super::JType;

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

        if s.is_empty() { return Ok(0); }
        let ch = s.chars().next().ok_or("impossible empty string")?; // cannot fail since s is not empty
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

pub fn translate_method<'a, 'b>(class : &'a Context<'b, &'b ClassConstant>, method : &'a Context<'b, &'b MethodInfo>) -> GeneralResult<Method> {
    let total_locals = get_method_code(method.as_ref())?.max_locals;
    let descriptor = get_string(class, method.as_ref().descriptor_index).ok_or("method descriptor missing")?;
    let num_returns = count_returns(&descriptor)?.into();
    // Pretend we have at least as many locals as we have return-slots, so we have somewhere to
    // store our results when we Yield.
    let max_locals = total_locals.max(num_returns);

    let sm = &stack::Manager::new(max_locals, STACK_PTR, STACK_REGS.to_owned());
    let sp = sm.get_stack_ptr();
    let max_locals = i32::from(max_locals);

    let prologue = {
        let name = "prologue";
        let off = -(max_locals - i32::from(count_params(&descriptor)?) + i32::from(SAVE_SLOTS));
        let down = sm.get_frame_offset(max_locals);
        let rp = sm.get_regs()[0];
        let insns = {
            // save return address in save-slot, one past the maximum number of locals
            tenyr_insn_list!(
                sp <-  sp + (off)   ;
                rp -> [sp + (down)] ;
            ).collect()
        };
        let label = make_label(class, method, &name)?;
        tenyr::BasicBlock { label, insns }
    };

    let epilogue = {
        let name = "epilogue";
        let off = i32::from(stack::SAVE_SLOTS) + i32::from(total_locals) - i32::from(num_returns);
        let down = i32::from(num_returns) - max_locals;
        let rp = Register::P;
        let insns = {
            tenyr_insn_list!(
                sp <-  sp + (off)   ;
                rp <- [sp + (down)] ;
            ).collect()
        };
        let label = make_label(class, method, &name)?;
        tenyr::BasicBlock { label, insns }
    };

    let blocks = make_blocks_for_method(class, method, sm)?;
    let name = mangle(&[ class, method ])?;
    Ok(Method { name, prologue, blocks, epilogue })
}

fn get_width<'a, T>(
        class : &Context<'_, &ClassConstant>,
        list : impl IntoIterator<Item=&'a T>,
        suff : Option<&str>,
    ) -> usize
    where T : Named + Described + 'a
{
    let get_len = |m| {
        let context = class.contextualize(m);
        let mut v : Vec<&dyn Manglable> = vec![ class, &context ];
        {
            #![allow(clippy::needless_borrow)]
            if let Some(ref suff) = suff { v.push(suff) }
        }
        mangle(&v).unwrap_or_default().len()
    };
    list.into_iter().fold(0, |c, m| c.max(get_len(m)))
}

fn write_method_table(class : &Context<'_, &ClassConstant>, methods : &[MethodInfo], outfile : &mut dyn Write) -> GeneralResult<()> {
    let label = ".Lmethod_table";
    writeln!(outfile, "{}:", label)?;
    let width = get_width(class, methods, None);
    for method in methods {
        let flags = method.access_flags;
        let mangled_name = mangle(&[ class, &class.contextualize(method) ])?;

        writeln!(outfile, "    .word @{:width$} - {}, {:#06x}", mangled_name, label, flags.bits(), width=width)?;
    }

    writeln!(outfile, "{}_end:", label)?;
    writeln!(outfile, "    .zero 0")?;
    writeln!(outfile)?;
    Ok(())
}

fn write_vslot_list(class : &Context<'_, &ClassConstant>, methods : &[MethodInfo], outfile : &mut dyn Write) -> GeneralResult<()> {
    let non_virtual = MethodAccessFlags::STATIC | MethodAccessFlags::PRIVATE;
    let suff = "vslot";

    let virtuals : Vec<_> = methods.iter().filter(|m| (m.access_flags & non_virtual).is_empty()).collect();
    let width = get_width(class, virtuals.clone().into_iter(), Some(suff)); // TODO obviate clone

    for (index, &method) in virtuals.iter().enumerate() {
        let mangled_name = mangle(&[ class, &class.contextualize(method), &suff ])?;
        writeln!(outfile, "    .global {}", mangled_name)?;
        writeln!(outfile, "    .set    {:width$}, {}", mangled_name, index, width=width)?;
    }
    if ! virtuals.is_empty() {
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
    let fields : Vec<_> = fields.iter().filter(selector).collect();
    let width = get_width(class, fields.clone().into_iter(), Some(suff));
    let get_size = |i| {
        let s = get_string(class, i).expect("missing descriptor");
        let desc = s.chars().next().expect("empty descriptor");
        args::field_size(desc).expect("getting field size failed")
    };
    let offsets = fields.iter().scan(0, |off, &f| {
        let old = *off;
        *off += get_size(f.descriptor_index);
        Some(old)
    });

    for (&field, offset) in fields.iter().zip(offsets) {
        let mangled_name = mangle(&[ class, &class.contextualize(field), &suff ])?;
        let size = get_size(field.descriptor_index);
        generator(outfile, &mangled_name, offset.into(), size.into(), width)?;
    }
    if ! fields.is_empty() {
        writeln!(outfile)?;
    }

    Ok(())
}

pub fn translate_class(class : classfile_parser::ClassFile, outfile : &mut dyn Write) -> GeneralResult<()> {
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

