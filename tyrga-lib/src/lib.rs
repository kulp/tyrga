#![deny(clippy::option_unwrap_used)]
#![deny(clippy::result_unwrap_used)]

mod exprtree;
mod jvmtypes;
pub mod mangling;
mod stack;
#[macro_use] mod tenyr;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::ops::Range;

use classfile_parser::ClassFile;
use classfile_parser::attribute_info::CodeAttribute;
use classfile_parser::attribute_info::StackMapFrame;
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

fn expand_immediate_load(sm : &mut StackManager, insn : Instruction, imm : i32)
    -> GeneralResult<Vec<Instruction>>
{
    use tenyr::InsnGeneral;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::*;
    use SmallestImmediate::*;

    fn make_imm(temp_reg : Register, imm : SmallestImmediate) -> GeneralResult<Vec<Instruction>> {
        use tenyr::Register::*;

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

            let adder  = InsnGeneral { y : Register::A, op : Add , imm : 0u8.into() };
            let reserve = sm.reserve(1).into_iter();
            let temp = sm.get(0).ok_or("stack unexpectedly empty")?;
            let pack = make_imm(temp, imm)?.into_iter();
            let (op, a, b, c) = match kind {
                Type3(_) => (BitwiseOr, insn.x, Register::A, temp), // should never be reached, but provides generality
                Type0(g) => (g.op, insn.x, g.y, temp),
                Type1(g) => (g.op, insn.x, temp, g.y),
                Type2(g) => (g.op, temp, insn.x, g.y),
            };
            let operate = once(Instruction { kind : Type0(InsnGeneral { op, y : b, imm : 0u8.into() }), x : a, dd : NoLoad, z : insn.z });
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
    let mut sm = StackManager::new(5, O, v.clone());

    {
        let imm = 8675390i32;
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
        let imm = 8675309i32;
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
            assert_eq!(g.imm, 123u8.into());
        } else {
            return Err("wrong type".into());
        }
    }

    Ok(())
}

type Namer = dyn Fn(usize) -> GeneralResult<String>;
type MakeInsnResult = GeneralResult<(usize, Vec<Instruction>, Vec<Destination>)>;

fn make_target(target : u16, target_namer : &Namer) -> GeneralResult<exprtree::Atom> {
    use exprtree::Atom::*;
    use exprtree::Expr;
    use exprtree::Operation::*;
    use std::rc::Rc;

    let tn = target_namer(target.into())?;
    let a = Variable(tn);
    let b = Expression(Rc::new(Expr { a : Variable(".".to_owned()), op : Add, b : Immediate(1) }));
    Ok(Expression(Rc::new(Expr { a, op : Sub, b })))
}

type BranchComp = dyn FnMut(&mut StackManager) -> GeneralResult<(tenyr::Register, Vec<Instruction>)>;

fn make_int_branch(sm : &mut StackManager, addr : usize, invert : bool, target : u16, target_namer : &Namer, comp : &mut BranchComp) -> MakeInsnResult {
    use tenyr::*;
    use tenyr::InstructionType::*;
    use tenyr::Register::*;

    let o = make_target(target, target_namer)?;

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

fn make_instructions(sm : &mut StackManager, (addr, op) : (&usize, &Operation), target_namer : &Namer, get_constant : &ConstantGetter)
    -> MakeInsnResult
{
    use Operation::*;
    use jvmtypes::SwitchParams::*;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::*;

    // We need to track destinations and return them so that the caller can track stack state
    // through the chain of control flow, possibly cloning the StackManager state along the way to
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

    let make_mov  = |to, from| Instruction { dd : NoLoad, kind : Type3(0u8.into()), z : to, x : from };
    let make_load = |to, from| Instruction { dd : LoadRight , ..make_mov(to, from) };

    let make_jump = |target| {
        let result : GeneralResult<Instruction> = Ok(Instruction {
            kind : Type3(tenyr::Immediate::Expr(make_target(target, target_namer)?)),
            ..make_mov(tenyr::Register::P, tenyr::Register::P)
        });
        result
    };

    let translate_way = |way|
        match way {
            jvmtypes::Comparison::Eq => (tenyr::Opcode::CompareEq, false, false),
            jvmtypes::Comparison::Ne => (tenyr::Opcode::CompareEq, false, true ),
            jvmtypes::Comparison::Lt => (tenyr::Opcode::CompareLt, false, false),
            jvmtypes::Comparison::Ge => (tenyr::Opcode::CompareGe, false, false),
            jvmtypes::Comparison::Gt => (tenyr::Opcode::CompareLt, true , false),
            jvmtypes::Comparison::Le => (tenyr::Opcode::CompareGe, true , false),
        };

    let make_call = |sm : &mut StackManager, target, descriptor| {
        let mut insns = Vec::new();
        insns.extend(sm.freeze());

        // Save return address into bottom of register-based stack
        let bottom = sm.get_regs()[0];
        insns.push(Instruction {
            kind : Type3(1u8.into()),
            ..make_mov(bottom, tenyr::Register::P)
        });

        let far = format!("@+{}", target);
        insns.push(Instruction {
            kind : Type3(tenyr::Immediate::Expr(exprtree::Atom::Variable(far))),
            ..make_mov(tenyr::Register::P, tenyr::Register::P)
        });

        // adjust stack for returned values
        let takes = count_args(descriptor)?;
        let rets = count_returns(descriptor)?;
        sm.release_frozen(takes.into());
        sm.reserve_frozen(rets.into());
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
        let ch : GeneralResult<_> =
            kind.get_char()
                .ok_or("no char for kind")
                .map_err(Into::into);
        let ch : char = ch?;

        use ArithmeticOperation::*;
        let nargs = match op {
            Add | Sub | Mul | Div | Rem | Shl | Shr | Ushr | And | Or | Xor => 2,
            Neg => 1,
        };
        let result = format!("({}){}", std::iter::repeat(ch).take(nargs).collect::<String>(), ch);
        Ok(result) as GeneralResult<String>
    };
    let make_arithmetic_name = |kind, op| {
        let descriptor = make_arithmetic_descriptor(kind, op)?;
        let k = kind.get_char().ok_or("no char for kind")?;
        let proc = format!("{}{}", name_op(op).to_lowercase(), k);
        mangling::mangle(join_name_parts("tyrga/Builtin", &proc, &descriptor).bytes())
    };

    let make_int_constant = |sm : &mut StackManager, value : i32| {
        let mut v = Vec::new();
        v.extend(sm.reserve(1));
        let insn = make_mov(get_reg(sm.get(0))?, Register::A);
        v.extend(expand_immediate_load(sm, insn, value)?);
        Ok((*addr, v, default_dest.clone()))
    };

    match op.clone() { // TODO obviate clone
        Constant { kind : JType::Int, value } =>
            make_int_constant(sm, value),
        Yield { kind } => {
            let ret = Instruction { kind : Type3(1u8.into()), ..make_load(Register::P, sm.get_stack_ptr()) };
            use JType::*;
            let mut v = match kind {
                Void =>
                    vec![ ret ],
                Int | Float | Object | Short | Char | Byte =>
                    vec![
                        store_local(sm, get_reg(sm.get(0))?, 0),
                        ret,
                    ],
                Double | Long =>
                    vec![
                        store_local(sm, get_reg(sm.get(1))?, 0),
                        store_local(sm, get_reg(sm.get(0))?, 1),
                        ret
                    ],
            };
            v.extend(sm.empty());
            Ok((*addr, v, vec![])) // leaving the method is not a Destination we care about
        },

        Arithmetic { kind : JType::Int, op : ArithmeticOperation::Neg }
            => {
                use tenyr::*;
                let y = get_reg(sm.get(0))?;
                let x = Register::A;
                let z = x; // update same location on stack
                let op = Opcode::Subtract;
                let dd = MemoryOpType::NoLoad;
                let imm = 0u8.into();
                let v = vec![ Instruction { kind : Type0(InsnGeneral { y, op, imm }), x, z, dd } ];
                Ok((*addr, v, default_dest))
            },
        Arithmetic { kind : JType::Int, op } if translate_arithmetic_op(op).is_some()
            => {
                use tenyr::*;
                let y = get_reg(sm.get(0))?;
                let x = get_reg(sm.get(1))?;
                let z = x;
                let op = translate_arithmetic_op(op).ok_or("no op for this opcode")?;
                let dd = MemoryOpType::NoLoad;
                let imm = 0u8.into();
                let mut v = Vec::new();
                v.push(Instruction { kind : Type0(InsnGeneral { y, op, imm }), x, z, dd });
                v.extend(sm.release(1));
                Ok((*addr, v, default_dest))
            },
        Arithmetic { kind, op }
            => make_call(sm, &make_arithmetic_name(kind, op)?, &make_arithmetic_descriptor(kind, op)?),
        LoadLocal { kind : JType::Int, index } |
        LoadLocal { kind : JType::Object, index }
            => {
                let mut v = Vec::new();
                v.extend(sm.reserve(1));
                v.push(load_local(sm, get_reg(sm.get(0))?, index.into()));
                Ok((*addr, v, default_dest))
            },
        StoreLocal { kind : JType::Int, index } |
        StoreLocal { kind : JType::Object, index }
            => {
                let mut v = Vec::new();
                v.push(store_local(sm, get_reg(sm.get(0))?, index.into()));
                v.extend(sm.release(1));
                Ok((*addr, v, default_dest))
            },
        Increment { index, value } => {
            use tenyr::*;
            let index = index.into();
            let imm = value.into();
            // This reserving of a stack slot may exceed the "maximum depth" statistic on the
            // method, but we should try to avoid dedicated temporary registers.
            let mut v = Vec::new();
            v.extend(sm.reserve(1));
            let temp_reg = get_reg(sm.get(0))?;
            v.extend(vec![
                load_local(sm, temp_reg, index),
                Instruction { kind : Type3(imm), ..make_mov(temp_reg, temp_reg) },
                store_local(sm, temp_reg, index),
            ]);
            v.extend(sm.release(1));

            Ok((*addr, v, default_dest))
        },
        Branch { kind : JType::Int, ops : OperandCount::_1, way, target } => {
            use tenyr::*;
            use tenyr::InstructionType::*;

            let (op, _, _) = translate_way(way);

            let mut op1 = move |sm : &mut StackManager| {
                let top = get_reg(sm.get(0))?;
                let temp_reg = top;
                let mut v = sm.release(1);
                v.push(Instruction {
                    kind : Type1(
                        InsnGeneral {
                           y : Register::A,
                           op,
                           imm : 0u8.into(),
                        }),
                    z : temp_reg,
                    x : top,
                    dd : MemoryOpType::NoLoad,
                });
                Ok((temp_reg, v))
            };

            make_int_branch(sm, *addr, false, target, target_namer, &mut op1)
        },
        Branch { kind : JType::Int, ops : OperandCount::_2, way, target } => {
            use tenyr::*;
            use tenyr::InstructionType::*;

            let (op, swap, invert) = translate_way(way);

            let mut op2 = move |sm : &mut StackManager| {
                let rhs = get_reg(sm.get(0))?;
                let lhs = get_reg(sm.get(1))?;
                let (rhs, lhs) = if swap { (lhs, rhs) } else { (rhs, lhs) };
                let temp_reg = lhs;
                let mut v = sm.release(2);
                v.push(Instruction {
                    kind : Type0(
                        InsnGeneral {
                           y : rhs,
                           op,
                           imm : 0u8.into(),
                        }),
                    z : temp_reg,
                    x : lhs,
                    dd : MemoryOpType::NoLoad,
                });
                Ok((temp_reg, v))
            };

            make_int_branch(sm, *addr, invert, target, target_namer, &mut op2)
        },
        Switch(Lookup { default, pairs }) => {
            use tenyr::*;
            use tenyr::InstructionType::*;

            let here = *addr as i32;
            let there = (default + here) as u16;

            let mut dests = Vec::new();
            let top = get_reg(sm.get(0))?;
            let mut insns = sm.reserve(1); // need a persistent temporary
            let temp_reg = get_reg(sm.get(0))?;

            let maker = |imm : i32| {
                move |sm : &mut StackManager| {
                    let insn = tenyr_insn!( temp_reg <- top == 0 );
                    let insns = expand_immediate_load(sm, insn?, imm)?;
                    Ok((temp_reg, insns))
                }
            };

            let brancher = |(compare, target)| {
                let result = make_int_branch(sm, *addr, false, (target + here) as u16, target_namer, &mut maker(compare));
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

            insns.push(make_jump(there)?);
            dests.push(Destination::Address(there.into()));

            Ok((*addr, insns, dests))
        },
        Switch(Table { default, low, high, offsets }) => {
            use tenyr::*;
            use tenyr::InstructionType::*;

            let here = *addr as i32;
            let there = (default + here) as u16;

            type InsnType = dyn Fn(InsnGeneral) -> InstructionType;

            let mut dests = Vec::new();
            let top = get_reg(sm.get(0))?;
            let mut insns = sm.reserve(1); // need a persistent temporary
            let temp_reg = get_reg(sm.get(0))?;

            let maker = |kind : &'static InsnType, imm : i32| {
                move |sm : &mut StackManager| {
                    let insn = Instruction {
                        kind : kind(
                            InsnGeneral {
                               y : Register::A,
                               op : Opcode::CompareLt,
                               imm : 0u8.into(), // placeholder
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
                make_int_branch(sm, *addr, false, there, target_namer, &mut maker(&Type1, low))?;
            let (_hi_addr, hi_insns, hi_dests) =
                make_int_branch(sm, *addr, false, there, target_namer, &mut maker(&Type2, high))?;

            let addr = lo_addr;

            insns.extend(lo_insns);
            insns.extend(hi_insns);

            use tenyr::Register::*;
            let insn = tenyr_insn!( P <- top - 0 + P );
            insns.extend(expand_immediate_load(sm, insn?, low)?);

            let make_pairs = |n| {
                let result : GeneralResult<(_,_)> =
                    Ok((make_jump((n + here) as u16)?, Destination::Address((n + here) as usize)));
                result
            };
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
            let mut v = Vec::new();
            let array_params = |sm : &mut StackManager, v : &mut Vec<Instruction>| {
                let idx = get_reg(sm.get(0))?;
                let arr = get_reg(sm.get(1))?;
                v.extend(sm.release(2));
                Ok((idx, arr)) as GeneralResult<(Register, Register)>
            };
            use tenyr::*;
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
                1 => Ok((Opcode::BitwiseOr, 0u8)),
                2 => Ok((Opcode::ShiftLeft, 1u8)),
                _ => Err("bad kind size"),
            }?;
            let imm = imm.into();
            let kind = Type1(InsnGeneral { y, op, imm });
            let insn = Instruction { kind, z, x, dd };
            v.push(insn);
            Ok((*addr, v, default_dest))
        },
        Noop => Ok((*addr, vec![ make_mov(Register::A, Register::A) ], default_dest)),
        Length => {
            // TODO document layout of arrays
            // This implementation assumes a reference to an array points to its first element, and
            // that one word below that element is a word containing the number of elements.
            let top = get_reg(sm.get(0))?;
            let insn = Instruction { kind : Type3((-1i8).into()), ..make_load(top, top) };
            Ok((*addr, vec![ insn ], default_dest))
        },
        // TODO fully handle Special (this is dumb partial handling)
        Invocation { kind : InvokeKind::Special, index } |
            Invocation { kind : InvokeKind::Static, index } =>
            make_call(sm, &make_callable_name(get_constant, index)?, &get_method_parts(get_constant, index)?.2),
        StackOp { op : StackOperation::Pop, size } => {
            let size : u8 = size.into();
            let v = sm.release(size.into());
            Ok((*addr, v, default_dest))
        },

        _ => Err(format!("unhandled operation {:?}", op).into()),
    }
}

#[test]
fn test_make_instruction() -> GeneralResult<()> {
    use tenyr::MemoryOpType::*;
    use Register::*;
    use Instruction;
    use tenyr::InstructionType::*;
    let mut sm = StackManager::new(5, STACK_PTR, STACK_REGS.to_owned());
    let op = Operation::Constant { kind : JType::Int, value : 5 };
    let namer = |x| Ok(format!("{}:{}", "test", x));
    use classfile_parser::constant_info::ConstantInfo::Unusable;
    let insn = make_instructions(&mut sm, (&0, &op), &namer, &|_| &Unusable)?;
    let imm = 5u8.into();
    assert_eq!(insn.1, vec![ Instruction { kind : Type3(imm), z : STACK_REGS[0], x : A, dd : NoLoad } ]);
    assert_eq!(insn.1[0].to_string(), " B  <-  5");

    Ok(())
}

pub type GeneralResult<T> = std::result::Result<T, Box<dyn Error>>;

fn generic_error<E>(e : E) -> Box<dyn Error>
    where E : std::error::Error
{
    format!("unknown error: {}", e).into()
}

#[cfg(test)]
fn parse_class(stem : &str) -> GeneralResult<ClassFile> {
    let mut name = String::from(concat!(env!("OUT_DIR"), "/"));
    name.push_str(stem);
    classfile_parser::parse_class(&name).map_err(Into::into)
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

    use std::iter::once;
    #[allow(clippy::len_zero)] // is_empty is ambiguous at the time of this writing
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

    pub type ConstantGetter<'a> = dyn Fn(u16) -> &'a ConstantInfo + 'a;

    pub fn get_constant_getter<'a>(class : &'a ClassFile) -> impl Fn(u16) -> &'a ConstantInfo + 'a {
        move |n| &class.const_pool[usize::from(n) - 1]
    }

    pub fn get_string(get_constant : &ConstantGetter, i : u16) -> Option<String>
    {
        use classfile_parser::constant_info::ConstantInfo::Utf8;
        match get_constant(i) {
            Utf8(u) => Some(u.utf8_string.to_string()),
            _ => None,
        }
    }

    use super::StackManager;
    use super::tenyr::Instruction;
    use super::tenyr::MemoryOpType::*;
    use super::tenyr::Register;

    pub fn index_local(sm : &StackManager, reg : Register, idx : i32) -> Instruction {
        Instruction { dd : NoLoad, z : reg, ..sm.get_frame_offset(idx) }
    }
    pub fn load_local(sm : &StackManager, reg : Register, idx : i32) -> Instruction {
        Instruction { dd : LoadRight, ..index_local(sm, reg, idx) }
    }
    pub fn store_local(sm : &StackManager, reg : Register, idx : i32) -> Instruction {
        Instruction { dd : StoreRight, ..index_local(sm, reg, idx) }
    }
}

fn get_ranges_for_method(class : &ClassFile, method : &MethodInfo)
    -> GeneralResult<RangeMap<Operation>>
{
    use classfile_parser::attribute_info::AttributeInfo;
    use classfile_parser::attribute_info::stack_map_table_attribute_parser;
    use classfile_parser::constant_info::ConstantInfo::Utf8;

    let get_constant = get_constant_getter(class);
    let named = |a : &AttributeInfo|
        match get_constant(a.attribute_name_index) {
            Utf8(u) => Ok((a.info.clone(), &u.utf8_string)),
            _ => Err("not a name"),
        };

    let code = get_method_code(method)?;
    let names : Result<Vec<_>,_> = code.attributes.iter().map(named).collect();
    let info = names?.into_iter().find(|(_, name)| name == &"StackMapTable").map(|t| t.0);
    let keep;
    let table = match info {
        Some(info) => {
            keep = stack_map_table_attribute_parser(&info).map_err(generic_error)?;
            &keep.1.entries
        },
        _ => &[] as &[StackMapFrame],
    };

    use classfile_parser::code_attribute::code_parser;
    let vec = code_parser(&code.code).map_err(generic_error)?.1;
    let (ranges, map) = derive_ranges(vec, table)?;
    let ops = map.into_iter().map(decode_insn).collect::<BTreeMap<_,_>>();
    Ok((ranges, ops))
}

fn join_name_parts(class : &str, name : &str, desc : &str) -> String {
    vec![ class, name, desc ].join(":")
}

type MethodNameParts = (String, String, String);

fn get_method_parts(get_constant : &ConstantGetter, pool_index : u16) -> GeneralResult<MethodNameParts> {
    use classfile_parser::constant_info::ConstantInfo::*;

    let get_string = |n| get_string(get_constant, n);

    if let MethodRef(mr) = get_constant(pool_index) {
        if let Class(cl) = get_constant(mr.class_index) {
            if let NameAndType(nt) = get_constant(mr.name_and_type_index) {
                return Ok((
                        get_string(cl.name_index).ok_or("bad class name")?,
                        get_string(nt.name_index).ok_or("bad method name")?,
                        get_string(nt.descriptor_index).ok_or("bad method descriptor")?,
                    ));
            }
        }
    }

    Err("error during constant pool lookup".into())
}

fn make_callable_name(get_constant : &ConstantGetter, pool_index : u16) -> GeneralResult<String> {
    let parts = get_method_parts(get_constant, pool_index)?;
    let joined = join_name_parts(&parts.0, &parts.1, &parts.2);
    mangling::mangle(joined.bytes())
}

fn make_unique_method_name(class : &ClassFile, method : &MethodInfo) -> GeneralResult<String> {
    use classfile_parser::constant_info::ConstantInfo::*;

    let get_constant = get_constant_getter(class);
    let get_string = |n| get_string(&get_constant, n);

    let cl = match get_constant(class.this_class) { Class(c) => Ok(c), _ => Err("not a class") }?;
    let name = join_name_parts(
        get_string(cl.name_index).ok_or("bad class name")?.as_ref(),
        get_string(method.name_index).ok_or("bad method name")?.as_ref(),
        get_string(method.descriptor_index).ok_or("bad method descriptor")?.as_ref()
    );
    Ok(name)
}

fn make_mangled_method_name(class : &ClassFile, method : &MethodInfo) -> GeneralResult<String> {
    let name = make_unique_method_name(class, method)?;
    mangling::mangle(name.bytes())
}

fn make_label(class : &ClassFile, method : &MethodInfo, suffix : &str) -> GeneralResult<String> {
    Ok(format!(".L{}{}",
        make_mangled_method_name(class, method)?,
        mangling::mangle(format!(":__{}", suffix).bytes())?))
}

fn make_basic_block<T>(class : &ClassFile, method : &MethodInfo, list : T, range : &Range<usize>)
    -> GeneralResult<(tenyr::BasicBlock, BTreeSet<usize>)>
    where T : IntoIterator<Item=MakeInsnResult>
{
    use tenyr::BasicBlock;

    let mut insns = Vec::with_capacity(range.len() * 2); // heuristic
    let mut exits = BTreeSet::new();

    let inside = |addr| addr >= range.start && addr < range.end;

    use Destination::*;

    let mut includes_successor = false;
    for insn in list {
        let does_branch = |&e| if let Address(n) = e { Some(n) } else { None };

        let (_, ins, exs) = insn?;

        // update the state of includes_successor each time so that the last instruction's behavior
        // is captured
        includes_successor = exs.iter().any(|e| if let Successor = e { true } else { false });

        exits.extend(exs.iter().filter_map(does_branch).filter(|&e| !inside(e)));
        insns.extend(ins);
    }
    let label = make_label(class, method, &range.start.to_string())?;

    if includes_successor {
        exits.insert(range.end);
    }

    insns.shrink_to_fit();

    Ok((BasicBlock { label, insns }, exits))
}

// The incoming StackManager represents a "prototype" StackManager which should be empty, and which
// will be cloned each time a new BasicBlock is seen.
fn make_blocks_for_method(class : &ClassFile, method : &MethodInfo, sm : &StackManager)
    -> GeneralResult<Vec<tenyr::BasicBlock>>
{
    let (ranges, ops) = get_ranges_for_method(&class, &method)?;
    use std::iter::FromIterator;
    let rangemap = &BTreeMap::from_iter(ranges.into_iter().map(|r| (r.start, r)));
    let ops = &ops;

    struct Params<'a> {
        class : &'a ClassFile,
        method : &'a MethodInfo,
        rangemap : &'a BTreeMap<usize, Range<usize>>,
        ops : &'a BTreeMap<usize, Operation>,
    }

    let params = Params { class, method, rangemap, ops };

    let mut seen = HashSet::new();

    fn make_blocks(params : &Params, seen : &mut HashSet<usize>, mut sm : StackManager, which : &Range<usize>) -> GeneralResult<Vec<tenyr::BasicBlock>> {
        let (class, method, rangemap, ops) = (params.class, params.method, params.rangemap, params.ops);
        if seen.contains(&which.start) {
            return Ok(vec![]);
        }
        seen.insert(which.start);

        let namer = {
            // TODO obviate clones
            let class = class.clone();
            let method = method.clone();
            move |x : usize| make_label(&class, &method, &x.to_string())
        };

        let get_constant = get_constant_getter(&class);

        let block : Vec<_> = ops.range(which.clone()).map(|x| make_instructions(&mut sm, x, &namer, &get_constant)).collect();
        let (bb, ee) = make_basic_block(&class, &method, block, which)?;
        let mut out = Vec::new();
        out.push(bb);

        for exit in &ee {
            out.extend(make_blocks(params, seen, sm.clone(), &rangemap[&exit])?); // intentional clone of StackManager
        }

        Ok(out)
    }

    make_blocks(&params, &mut seen, sm.clone(), &rangemap[&0]) // intentional clone of StackManager
}

#[cfg(test)]
fn test_stack_map_table(stem : &str) -> GeneralResult<()> {
    let class = parse_class(stem)?;
    for method in &class.methods {
        let sm = StackManager::new(5, STACK_PTR, STACK_REGS.to_owned());
        let bbs = make_blocks_for_method(&class, method, &sm)?;
        for bb in &bbs {
            eprintln!("{}", bb);
        }
    }

    Ok(())
}

#[cfg(test)]
const CLASS_LIST : &[&str] = &[
    "Except",
    "Expr",
    "GCD",
    "Nest",
    "Sieve",
    "Switch",
    "Tiny",
];

#[test]
fn test_parse_classes() -> GeneralResult<()>
{
    for name in CLASS_LIST {
        test_stack_map_table(name)?;
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Method {
    name : String,
    preamble : tenyr::BasicBlock,
    blocks : Vec<tenyr::BasicBlock>,
}

impl fmt::Display for Method {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, ".global {}", self.name)?;
        writeln!(f, "{}:", self.name)?;
        write!(f, "{}", self.preamble)?;
        for bb in &self.blocks {
            write!(f, "{}", bb)?
        }
        Ok(())
    }
}

mod args {
    use super::GeneralResult;

    fn count_internal(s : &str) -> GeneralResult<u8> {
        fn eat(s : &str) -> GeneralResult<usize> {
            let ch = s.chars().nth(0).ok_or("string ended too soon")?;
            match ch {
                'B' | 'C' | 'F' | 'I' | 'S' | 'Z' | 'D' | 'J' | 'V' => Ok(1),
                'L' => Ok(1 + s.find(';').ok_or("string ended too soon")?),
                '[' => Ok(1 + eat(&s[1..])?),
                _ => Err(format!("unexpected character {}", ch).into()),
            }
        }

        if s.is_empty() { return Ok(0); }
        let ch = s.chars().nth(0).ok_or("impossible empty string")?; // cannot fail since s is not empty
        let mine = match ch {
            'B' | 'C' | 'F' | 'I' | 'S' | 'Z' => Ok(1),
            'D' | 'J' => Ok(2),
            'L' => Ok(1),
            '[' => Ok(1),
            'V' => Ok(0),
            _ => Err(format!("unexpected character {}", ch)),
        };
        Ok(mine? + count_internal(&s[eat(s)?..])?)
    }

    // JVM limitations restrict the count of method parameters to 255 at most
    pub fn count_args(descriptor : &str) -> GeneralResult<u8> {
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
fn test_count_args() -> GeneralResult<()> {
    assert_eq!(3, count_args("(III)V")?);
    assert_eq!(4, count_args("(JD)I")?);
    assert_eq!(2, count_args("(Lmetasyntactic;Lvariable;)I")?);
    assert_eq!(1, count_args("([[[I)I")?);
    assert_eq!(0, count_args("()Lplaceholder;")?);
    assert_eq!(0, count_args("()D")?);
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

pub fn translate_method(class : &ClassFile, method : &MethodInfo) -> GeneralResult<Method> {
    use tenyr::*;
    use tenyr::MemoryOpType::*;
    use tenyr::InstructionType::*;

    let max_locals = get_method_code(method)?.max_locals;
    let sm = StackManager::new(max_locals, STACK_PTR, STACK_REGS.to_owned());
    let sm = &sm;

    let insns = {
        let descriptor = get_string(&get_constant_getter(class), method.descriptor_index).ok_or("method descriptor missing")?;

        let max_locals = i32::from(max_locals);
        let net = max_locals - i32::from(count_args(&descriptor)?);

        let z = sm.get_stack_ptr();
        let x = z;
        let kind = Type3((-(net + i32::from(SAVE_SLOTS))).try_into()?);
        let bottom = sm.get_regs()[0];
        vec![
            // update stack pointer
            Instruction { dd : NoLoad, kind, z, x },
            // save return address in save-slot, one past the maximum number of locals
            store_local(sm, bottom, max_locals),
        ]
    };
    let label = make_label(class, method, "preamble")?;
    let preamble = tenyr::BasicBlock { label, insns };

    let blocks = make_blocks_for_method(class, method, sm)?;
    let name = make_mangled_method_name(class, method)?;
    Ok(Method { name, preamble, blocks })
}
