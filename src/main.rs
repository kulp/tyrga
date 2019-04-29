mod exprtree;
mod jvmtypes;
mod mangling;
mod stack;
mod tenyr;

use std::collections::BTreeSet;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::ops::Range;

use jvmtypes::*;

use classfile_parser::ClassFile;

use tenyr::Register;

use stack::*;

const STACK_PTR : Register = Register::O;
const STACK_REGS : &[Register] = { use Register::*; &[ B, C, D, E, F, G, H, I, J, K, L, M, N ] };

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Destination {
    Successor,
    Address(usize),
    Return,
}

type Namer = Fn(usize) -> String;
type MakeInsnResult = (usize, Vec<tenyr::Instruction>, Vec<Destination>);

fn make_target(target : u16, target_namer : &Namer) -> exprtree::Atom {
    use exprtree::Atom::*;
    use exprtree::Expr;
    use exprtree::Operation::*;
    use std::rc::Rc;

    let tn = target_namer(target.into());
    let a = Variable(tn);
    let b = Expression(Rc::new(Expr { a : Variable(".".to_owned()), op : Add, b : Immediate(1) }));
    Expression(Rc::new(Expr { a, op : Sub, b }))
}

type BranchComp = FnMut(&mut StackManager) -> (tenyr::Register, Vec<tenyr::Instruction>);

fn make_int_branch(sm : &mut StackManager, addr : usize, invert : bool, target : u16, target_namer : &Namer, comp : &mut BranchComp) -> MakeInsnResult {
    use tenyr::*;
    use tenyr::InstructionType::*;

    let mut dest = Vec::new();
    dest.push(Destination::Successor);
    dest.push(Destination::Address(target.into()));
    let o = make_target(target, target_namer);

    let (temp_reg, sequence) = comp(sm);
    let branch = Instruction {
        kind : Type2(
            InsnGeneral {
               y : Register::P,
               op : if invert { Opcode::BitwiseAndn } else { Opcode::BitwiseAnd },
               imm : tenyr::Immediate::Expr(o),
            }),
        z : Register::P,
        x : temp_reg,
        dd : MemoryOpType::NoLoad,
    };
    let mut v = sequence;
    v.push(branch);
    (addr, v, dest)
}

fn make_instructions(sm : &mut StackManager, (addr, op) : (&usize, &Operation), target_namer : &Namer, get_constant : &ConstantGetter)
    -> MakeInsnResult
{
    use Operation::*;
    use jvmtypes::SwitchParams::*;
    use tenyr::Immediate12;
    use tenyr::Immediate20;
    use tenyr::Instruction;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::*;

    // We need to track destinations and return them so that the caller can track stack state
    // through the chain of control flow, possibly cloning the StackManager state along the way to
    // follow multiple destinations. Each basic block needs to be visited only once, however, since
    // the JVM guarantees that every instance of every instruction within a method always sees the
    // same depth of the operand stack every time that instance is executed.
    let default_dest = vec![Destination::Successor];

    let get_reg = |t : Option<_>| t.expect("asked but did not receive");

    let make_imm12 = |n| Immediate12::new(n).unwrap();

    let pos1_20 = Immediate20::new( 1).unwrap(); // will not fail
    let neg1_20 = Immediate20::new(-1).unwrap(); // will not fail

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

    let make_mov  = |to, from| Instruction { dd : NoLoad, kind : Type3(Immediate20::ZERO), z : to, x : from };
    let make_load = |to, from| Instruction { dd : LoadRight , ..make_mov(to, from) };

    let make_jump = |target| {
        Instruction {
            kind : Type3(tenyr::Immediate::Expr(make_target(target, target_namer))),
            ..make_mov(tenyr::Register::P, tenyr::Register::P)
        }
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

    match op.clone() { // TODO obviate clone
        Constant { kind : JType::Int, value } => {
            let kind = Type3(Immediate20::new(value).expect("immediate too large"));
            let mut v = Vec::with_capacity(10);
            v.extend(sm.reserve(1));
            let z = get_reg(sm.get(0));
            v.push(Instruction { kind, z, x : Register::A, dd : NoLoad });
            (*addr, v, default_dest)
        },
        Yield { kind } => {
            let ret = Instruction { kind : Type3(pos1_20), ..make_load(Register::P, sm.get_stack_ptr()) };
            // TODO how to correctly place stack pointer ?
            // StackManager will somehow have to help us manipulate it because we do not
            // here have enough context otherwise.

            use JType::*;
            let mut v = match kind {
                Void => {
                    vec![ ret ]
                },
                Int | Float | Object | Short | Char | Byte => {
                    let top = get_reg(sm.get(0));
                    vec![
                        store_local(sm, top, 0),
                        ret
                    ]
                },
                Double | Long => {
                    let top = get_reg(sm.get(0));
                    let sec = get_reg(sm.get(1));
                    vec![
                        store_local(sm, sec, 0),
                        store_local(sm, top, 1),
                        ret
                    ]
                },
            };
            v.extend(sm.empty());
            (*addr, v, vec![ Destination::Return ])
        },

        Arithmetic { kind : JType::Int, op : ArithmeticOperation::Neg }
            => {
                use tenyr::*;
                let y = get_reg(sm.get(0));
                let x = Register::A;
                let z = x; // update same location on stack
                let op = Opcode::Subtract;
                let dd = MemoryOpType::NoLoad;
                let imm = Immediate12::ZERO;
                let v = vec![ Instruction { kind : Type0(InsnGeneral { y, op, imm }), x, z, dd } ];
                (*addr, v, default_dest)
            },
        Arithmetic { kind : JType::Int, op } if translate_arithmetic_op(op).is_some()
            => {
                use tenyr::*;
                let y = get_reg(sm.get(0));
                let x = get_reg(sm.get(1));
                let z = x;
                let op = translate_arithmetic_op(op).unwrap();
                let dd = MemoryOpType::NoLoad;
                let imm = Immediate12::ZERO;
                let mut v = Vec::new();
                v.push(Instruction { kind : Type0(InsnGeneral { y, op, imm }), x, z, dd });
                v.extend(sm.release(1));
                (*addr, v, default_dest)
            },
        LoadLocal { kind, index } if kind == JType::Int || kind == JType::Object
            => {
                let mut v = Vec::with_capacity(10);
                v.extend(sm.reserve(1));
                v.push(load_local(sm, get_reg(sm.get(0)), i32::from(index)));
                (*addr, v, default_dest)
            },
        StoreLocal { kind, index } if kind == JType::Int || kind == JType::Object
            => {
                let mut v = Vec::with_capacity(10);
                v.push(store_local(sm, get_reg(sm.get(0)), i32::from(index)));
                v.extend(sm.release(1));
                (*addr, v, default_dest)
            },
        Increment { index, value } => {
            use tenyr::*;
            let index = i32::from(index);
            let imm = make_imm12(value);
            let y = Register::A;
            let op = Opcode::Add;
            // This reserving of a stack slot may exceed the "maximum depth" statistic on the
            // method, but we should try to avoid dedicated temporary registers.
            let mut v = Vec::with_capacity(10);
            v.extend(sm.reserve(1));
            let temp_reg = get_reg(sm.get(0));
            v.extend(vec![
                load_local(sm, temp_reg, index),
                Instruction { kind : Type1(InsnGeneral { y, op, imm }), ..make_mov(temp_reg, temp_reg) },
                store_local(sm, temp_reg, index),
            ]);
            v.extend(sm.release(1));

            (*addr, v, default_dest)
        },
        Branch { kind : JType::Int, ops : OperandCount::_1, way, target } => {
            use tenyr::*;
            use tenyr::InstructionType::*;

            let (op, _, _) = translate_way(way);

            let mut op1 = move |sm : &mut StackManager| {
                let top = get_reg(sm.get(0));
                let temp_reg = top;
                let mut v = sm.release(1);
                v.push(Instruction {
                    kind : Type1(
                        InsnGeneral {
                           y : Register::A,
                           op,
                           imm : Immediate12::ZERO,
                        }),
                    z : temp_reg,
                    x : top,
                    dd : MemoryOpType::NoLoad,
                });
                (temp_reg, v)
            };

            make_int_branch(sm, *addr, false, target, target_namer, &mut op1)
        },
        Branch { kind : JType::Int, ops : OperandCount::_2, way, target } => {
            use tenyr::*;
            use tenyr::InstructionType::*;

            let (op, swap, invert) = translate_way(way);

            let mut op2 = move |sm : &mut StackManager| {
                let rhs = get_reg(sm.get(0));
                let lhs = get_reg(sm.get(1));
                let (rhs, lhs) = if swap { (lhs, rhs) } else { (rhs, lhs) };
                let temp_reg = lhs;
                let mut v = sm.release(2);
                v.push(Instruction {
                    kind : Type0(
                        InsnGeneral {
                           y : rhs,
                           op,
                           imm : Immediate12::ZERO,
                        }),
                    z : temp_reg,
                    x : lhs,
                    dd : MemoryOpType::NoLoad,
                });
                (temp_reg, v)
            };

            make_int_branch(sm, *addr, invert, target, target_namer, &mut op2)
        },
        Switch(Lookup { default, pairs }) => {
            use tenyr::*;
            use tenyr::InstructionType::*;

            let here = *addr as i32;
            let there = (default + here) as u16;

            let mut dests = Vec::new();
            let top = get_reg(sm.get(0));
            let mut insns = sm.reserve(1); // need a persistent temporary
            let temp_reg = get_reg(sm.get(0));

            let maker = |imm| {
                move |_sm : &mut StackManager| {
                    let insns = vec![ Instruction {
                        kind : Type1(
                            InsnGeneral {
                               y : Register::A,
                               op : Opcode::CompareEq,
                               imm : Immediate12::new(imm).unwrap(),
                            }),
                        z : temp_reg,
                        x : top,
                        dd : MemoryOpType::NoLoad,
                    } ];
                    (temp_reg, insns)
                }
            };

            let (i, d) : (Vec<_>, Vec<_>) = pairs.iter().map(|&(compare, target)| {
                let (_, insns, dests) =
                    make_int_branch(sm, *addr, false, (target + here) as u16, target_namer, &mut maker(compare));
                (insns, dests)
            }).unzip();

            let i = i.concat();
            let d = d.concat();

            insns.extend(i);
            dests.extend(d);

            insns.push(make_jump(there));
            dests.push(Destination::Address(usize::from(there)));

            (*addr, insns, dests)
        },
        Switch(Table { default, low, high, offsets }) => {
            use tenyr::*;
            use tenyr::InstructionType::*;

            let here = *addr as i32;
            let there = (default + here) as u16;

            type InsnType = dyn Fn(InsnGeneral) -> InstructionType;

            let mut dests = Vec::new();
            let top = get_reg(sm.get(0));
            let mut insns = sm.reserve(1); // need a persistent temporary
            let temp_reg = get_reg(sm.get(0));

            let maker = |kind : &'static InsnType, imm| {
                move |_sm : &mut StackManager| {
                    let insns = vec![ Instruction {
                        kind : kind(
                            InsnGeneral {
                               y : Register::A,
                               op : Opcode::CompareLt,
                               imm : Immediate12::new(imm).unwrap(),
                            }),
                        z : temp_reg,
                        x : top,
                        dd : MemoryOpType::NoLoad,
                    } ];
                    (temp_reg, insns)
                }
            };

            let (lo_addr, lo_insns, lo_dests) =
                make_int_branch(sm, *addr, false, there, target_namer, &mut maker(&Type1, low));
            let (_hi_addr, hi_insns, hi_dests) =
                make_int_branch(sm, *addr, false, there, target_namer, &mut maker(&Type2, high));

            let addr = lo_addr;

            insns.extend(lo_insns);
            insns.extend(hi_insns);

            let kind = Type1(InsnGeneral { y : Register::P, op : Opcode::Subtract, imm : Immediate12::new(low).unwrap() });
            insns.push(Instruction { kind, z : Register::P, x : top, dd : NoLoad });

            let (i, d) : (Vec<_>, Vec<_>) =
                offsets
                    .into_iter()
                    .map(|n| (make_jump((n + here) as u16), Destination::Address((n + here) as usize)))
                    .unzip();

            insns.extend(i);
            dests.extend(d);

            dests.extend(lo_dests);
            dests.extend(hi_dests);

            insns.extend(sm.release(1)); // release temporary

            (addr, insns, dests)
        },
        Jump { target } => (*addr, vec![ make_jump(target) ], vec![ Destination::Address(target as usize) ]),
        LoadArray(kind) | StoreArray(kind) => {
            let mut v = Vec::with_capacity(10);
            let array_params = |sm : &mut StackManager, v : &mut Vec<Instruction>| {
                let idx = get_reg(sm.get(0));
                let arr = get_reg(sm.get(1));
                v.extend(sm.release(2));
                (idx, arr)
            };
            use JType::*;
            use tenyr::*;
            let (x, y, z, dd) = match *op {
                LoadArray(_) => {
                    let (idx, arr) = array_params(sm, &mut v);
                    v.extend(sm.reserve(1));
                    let res = get_reg(sm.get(0));
                    (idx, arr, res, LoadRight)
                },
                StoreArray(_) => {
                    let val = get_reg(sm.get(0));
                    v.extend(sm.release(1));
                    let (idx, arr) = array_params(sm, &mut v);
                    (idx, arr, val, StoreRight)
                },
                _ => unreachable!(),
            };
            // For now, all arrays of int or smaller are stored unpacked (i.e. one bool/short/char
            // per 32-bit tenyr word)
            let imm = make_imm12(match kind { Double | Long => 2, _ => 1 });
            let kind = Type1(InsnGeneral { y, op : Opcode::Multiply, imm });
            let insn = Instruction { kind, z, x, dd };
            v.push(insn);
            (*addr, v, default_dest)
        },
        Noop => (*addr, vec![ make_mov(Register::A, Register::A) ], default_dest),
        Length => {
            // TODO document layout of arrays
            // This implementation assumes a reference to an array points to its first element, and
            // that one word below that element is a word containing the number of elements.
            let top = get_reg(sm.get(0));
            let insn = Instruction { kind : Type3(neg1_20), ..make_load(top, top) };
            (*addr, vec![ insn ], default_dest)
        },
        // TODO fully handle Special (this is dumb partial handling)
        Invocation { kind : InvokeKind::Special, index } |
            Invocation { kind : InvokeKind::Static, index } =>
        {
            let mut insns = Vec::new();
            insns.extend(sm.freeze());

            // Save return address into bottom of register-based stack
            let bottom = Register::B; // TODO get bottom of StackManager instead
            insns.push(Instruction {
                kind : Type3(pos1_20),
                ..make_mov(bottom, tenyr::Register::P)
            });

            let far = format!("@+{}", make_callable_name(get_constant, index));
            insns.push(Instruction {
                kind : Type3(tenyr::Immediate::Expr(exprtree::Atom::Variable(far))),
                ..make_mov(tenyr::Register::P, tenyr::Register::P)
            });

            // adjust stack for returned values
            let parts = get_method_parts(get_constant, index);
            let takes = count_args(&parts.2);
            let takes = takes.expect("failed to compute arguments size");
            let rets = count_returns(&parts.2);
            let rets = rets.expect("failed to compute return size");
            sm.release_frozen(u16::from(takes));
            sm.reserve_frozen(u16::from(rets));
            insns.extend(sm.thaw());
            (*addr, insns, default_dest)
        },

        _ => panic!("unhandled operation {:?}", op),
    }
}

#[test]
fn test_make_instruction() {
    use tenyr::MemoryOpType::*;
    use Register::*;
    use tenyr::Instruction;
    use tenyr::InstructionType::*;
    use tenyr::Immediate20;
    let mut sm = StackManager::new(5, STACK_PTR, STACK_REGS.to_owned());
    let op = Operation::Constant { kind : JType::Int, value : 5 };
    let namer = |x| format!("{}:{}", "test", x);
    use classfile_parser::constant_info::ConstantInfo::Unusable;
    let insn = make_instructions(&mut sm, (&0, &op), &namer, &|_| Unusable);
    let imm = Immediate20::new(5).unwrap();
    assert_eq!(insn.1, vec![ Instruction { kind: Type3(imm), z: STACK_REGS[0], x: A, dd: NoLoad } ]);
    assert_eq!(insn.1[0].to_string(), " B  <-  5");
}

#[derive(Clone, Debug)]
pub struct TranslationError(String);

impl TranslationError {
    fn new(msg: &str) -> TranslationError {
        TranslationError(msg.to_string())
    }
}

impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for TranslationError {
    fn description(&self) -> &str { &self.0 }
}

pub type Result<T> = std::result::Result<T, Box<Error>>;

fn generic_error<E>(e : E) -> Box<TranslationError>
    where E : std::error::Error
{
    Box::new(TranslationError::new(&format!("unknown error: {}", e)))
}

#[cfg(test)]
fn parse_class(stem : &str) -> ClassFile {
    let mut name = String::from(concat!(env!("OUT_DIR"), "/"));
    name.push_str(stem);
    classfile_parser::parse_class(&name).expect("failed to parse class")
}

use classfile_parser::attribute_info::StackMapFrame;
use std::collections::BTreeMap;
fn derive_ranges<'a, T>(body : &[(usize, &'a T)], table : &[StackMapFrame])
    -> Result<(Vec<Range<usize>>, BTreeMap<usize, &'a T>)>
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
    let err = Box::new(TranslationError::new("body unexpectedly empty"));
    let max = body.last().ok_or(err)?.0 + 1;
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
    Ok((ranges, tree))
}

use classfile_parser::attribute_info::CodeAttribute;
use classfile_parser::method_info::MethodInfo;

fn get_method_code(method : &MethodInfo) -> Result<CodeAttribute> {
    use classfile_parser::attribute_info::code_attribute_parser;
    Ok(code_attribute_parser(&method.attributes[0].info).map_err(generic_error)?.1)
}

mod util {
    use classfile_parser::ClassFile;
    use classfile_parser::constant_info::ConstantInfo;

    pub type ConstantGetter = Fn(u16) -> ConstantInfo;

    // TODO make this return a reference once we can appease the borrow checker
    pub fn get_constant_getter(class : &ClassFile) -> impl Fn(u16) -> ConstantInfo {
        let pool = class.const_pool.clone();
        move |n| pool[usize::from(n) - 1].clone()
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
use util::*;

fn get_ranges_for_method(class : &ClassFile, method : &MethodInfo)
    -> Result<(Vec<Range<usize>>, BTreeMap<usize, Operation>)>
{
    use classfile_parser::attribute_info::AttributeInfo;
    use classfile_parser::attribute_info::stack_map_table_attribute_parser;
    use classfile_parser::constant_info::ConstantInfo::Utf8;

    let get_constant = get_constant_getter(class);
    let name_of = |a : &AttributeInfo|
        match get_constant(a.attribute_name_index) {
            Utf8(u) => u.utf8_string.to_string(),
            _ => panic!("not a name")
        };

    let code = get_method_code(method)?;
    let attr = &code.attributes.iter().find(|a| name_of(a) == "StackMapTable");
    let keep;
    let table = match attr {
        Some(attr) => {
            keep = stack_map_table_attribute_parser(&attr.info).map_err(generic_error)?;
            &keep.1.entries
        },
        _ => &[] as &[StackMapFrame],
    };

    use classfile_parser::code_attribute::code_parser;
    let vec = code_parser(&code.code).map_err(generic_error)?.1;
    let refed = vec.iter().map(|(s, x)| (*s, x)).collect::<Vec<_>>();
    let (ranges, map) = derive_ranges(&refed, table)?;
    let ops = map.into_iter().map(decode_insn).collect::<BTreeMap<_,_>>();
    Ok((ranges, ops))
}

fn join_name_parts(class : &str, name : &str, desc : &str) -> String {
    vec![ class, name, desc ].join(":")
}

type MethodNameParts = (String, String, String);

fn get_method_parts(get_constant : &ConstantGetter, pool_index : u16) -> MethodNameParts {
    use classfile_parser::constant_info::ConstantInfo::*;

    let get_string = |n| match get_constant(n) {
        Utf8(u) => Some(u.utf8_string.to_string()),
        _ => None,
    };

    if let MethodRef(mr) = get_constant(pool_index) {
        if let Class(cl) = get_constant(mr.class_index) {
            if let NameAndType(nt) = get_constant(mr.name_and_type_index) {
                return (
                        get_string(cl.name_index).expect("bad class name"),
                        get_string(nt.name_index).expect("bad method name"),
                        get_string(nt.descriptor_index).expect("bad method descriptor"),
                    );
            }
        }
    }

    panic!("error during constant pool lookup");
}

fn make_callable_name(get_constant : &ConstantGetter, pool_index : u16) -> String {
    let parts = get_method_parts(get_constant, pool_index);
    let joined = join_name_parts(&parts.0, &parts.1, &parts.2);
    mangling::mangle(joined.bytes()).expect("failed to mangle")
}

fn make_unique_method_name(class : &ClassFile, method : &MethodInfo) -> String {
    use classfile_parser::constant_info::ConstantInfo::*;

    let get_constant = get_constant_getter(class);
    let get_string = |n| get_string(&get_constant, n);

    let cl = match get_constant(class.this_class) { Class(c) => c, _ => panic!("not a class") };
    join_name_parts(
        get_string(cl.name_index).expect("bad class name").as_ref(),
        get_string(method.name_index).expect("bad method name").as_ref(),
        get_string(method.descriptor_index).expect("bad method descriptor").as_ref()
    )
}

fn make_mangled_method_name(class : &ClassFile, method : &MethodInfo) -> String {
    mangling::mangle(make_unique_method_name(class, method).bytes()).expect("failed to mangle")
}

fn make_label(class : &ClassFile, method : &MethodInfo, suffix : &str) -> String {
    format!(".L{}{}",
        make_mangled_method_name(class, method),
        mangling::mangle(format!(":__{}", suffix).bytes()).expect("failed to mangle"))
}

fn make_basic_block<T>(class : &ClassFile, method : &MethodInfo, list : T, range : &Range<usize>) -> (tenyr::BasicBlock, BTreeSet<usize>)
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

        let (_, ins, exs) = insn;

        // update the state of includes_successor each time so that the last instruction's behavior
        // is captured
        includes_successor = exs.iter().any(|e| if let Successor = e { true } else { false });

        exits.extend(exs.iter().filter_map(does_branch).filter(|&e| !inside(e)));
        insns.extend(ins);
    }
    let label = make_label(class, method, &range.start.to_string());

    if includes_successor {
        exits.insert(range.end);
    }

    insns.shrink_to_fit();

    (BasicBlock { label, insns }, exits)
}

// The incoming StackManager represents a "prototype" StackManager which should be empty, and which
// will be cloned each time a new BasicBlock is seen.
fn make_blocks_for_method(class : &ClassFile, method : &MethodInfo, sm : &StackManager) -> Vec<tenyr::BasicBlock> {
    let (ranges, ops) = get_ranges_for_method(&class, &method).expect("failed to get ranges for map");
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

    fn make_blocks(params : &Params, seen : &mut HashSet<usize>, mut sm : StackManager, which : &Range<usize>) -> Vec<tenyr::BasicBlock> {
        let (class, method, rangemap, ops) = (params.class, params.method, params.rangemap, params.ops);
        if seen.contains(&which.start) {
            return vec![];
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
        let (bb, ee) = make_basic_block(&class, &method, block, which);
        let mut out = Vec::new();
        out.push(bb);

        for exit in &ee {
            out.extend(make_blocks(params, seen, sm.clone(), &rangemap[&exit])); // intentional clone of StackManager
        }

        out
    }

    make_blocks(&params, &mut seen, sm.clone(), &rangemap[&0]) // intentional clone of StackManager
}

#[cfg(test)]
fn test_stack_map_table(stem : &str) {
    let class = parse_class(stem);
    for method in &class.methods {
        let sm = StackManager::new(5, STACK_PTR, STACK_REGS.to_owned());
        let bbs = make_blocks_for_method(&class, method, &sm);
        for bb in &bbs {
            eprintln!("{}", bb);
        }
    }
}

#[cfg(test)]
const CLASS_LIST : &[&str] = &[
    //"Except",
    "Expr",
    "GCD",
    "Nest",
    //"Sieve",
    "Switch",
    //"Tiny",
];

#[test]
fn test_parse_classes()
{
    for name in CLASS_LIST {
        test_stack_map_table(name);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Method {
    name : String,
    preamble : tenyr::BasicBlock,
    blocks : Vec<tenyr::BasicBlock>,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    use super::TranslationError;
    use super::Result;

    fn count_internal(s : &str) -> Result<u8> {
        fn eat(s : &str) -> Result<usize> {
            let ch = s.chars().nth(0).ok_or_else(|| TranslationError::new("string ended too soon"))?;
            match ch {
                'B' | 'C' | 'F' | 'I' | 'S' | 'Z' | 'D' | 'J' | 'V' => Ok(1),
                'L' => Ok(1 + s.find(';').ok_or_else(|| TranslationError::new("string ended too soon"))?),
                '[' => Ok(1 + eat(&s[1..])?),
                _ => Err(TranslationError(format!("unexpected character {}", ch)).into()),
            }
        }

        if s.is_empty() { return Ok(0); }
        let ch = s.chars().nth(0).unwrap(); // cannot fail since s is not empty
        let mine = match ch {
            'B' | 'C' | 'F' | 'I' | 'S' | 'Z' => Ok(1),
            'D' | 'J' => Ok(2),
            'L' => Ok(1),
            '[' => Ok(1),
            'V' => Ok(0),
            _ => Err(TranslationError(format!("unexpected character {}", ch))),
        };
        Ok(mine? + count_internal(&s[eat(s)?..])?)
    }

    // JVM limitations restrict the count of method parameters to 255 at most
    pub fn count_args(descriptor : &str) -> Result<u8> {
        let open = 1; // byte index of open parenthesis is 0
        let close = descriptor.rfind(')').ok_or_else(|| TranslationError::new("descriptor missing closing parenthesis"))?;
        count_internal(&descriptor[open..close])
    }

    // JVM limitations restrict the count of return values to 1 at most, of size 2 at most
    pub fn count_returns(descriptor : &str) -> Result<u8> {
        let close = descriptor.rfind(')').ok_or_else(|| TranslationError::new("descriptor missing closing parenthesis"))?;
        count_internal(&descriptor[close+1..])
    }
}
use args::*;

#[test]
fn test_count_args() -> Result<()> {
    assert_eq!(3, count_args("(III)V")?);
    assert_eq!(4, count_args("(JD)I")?);
    assert_eq!(2, count_args("(Lmetasyntactic;Lvariable;)I")?);
    assert_eq!(1, count_args("([[[I)I")?);
    assert_eq!(0, count_args("()Lplaceholder;")?);
    assert_eq!(0, count_args("()D")?);
    Ok(())
}

#[test]
fn test_count_returns() -> Result<()> {
    assert_eq!(0, count_returns("(III)V")?);
    assert_eq!(1, count_returns("(JD)I")?);
    assert_eq!(1, count_returns("(Lmetasyntactic;Lvariable;)I")?);
    assert_eq!(1, count_returns("([[[I)I")?);
    assert_eq!(1, count_returns("()Lplaceholder;")?);
    assert_eq!(2, count_returns("()D")?);
    Ok(())
}

fn translate_method(class : &ClassFile, method : &MethodInfo, sm : &StackManager) -> Result<Method> {
    use tenyr::*;
    use tenyr::MemoryOpType::*;
    use tenyr::InstructionType::*;

    let sm = &mut sm.clone();

    let insns = {
        let err = || TranslationError::new("method descriptor missing");
        let descriptor = get_string(&get_constant_getter(class), method.descriptor_index).ok_or_else(err)?;

        let max_locals = i32::from(get_method_code(method)?.max_locals);
        let net = max_locals - i32::from(count_args(&descriptor)?);

        let bad_imm = || TranslationError::new("failed to create immediate");
        let z = sm.get_stack_ptr();
        let x = z;
        let kind = Type3(Immediate20::new(-(net + i32::from(SAVE_SLOTS))).ok_or_else(bad_imm)?);
        let bottom = Register::B; // TODO get bottom of StackManager instead
        vec![
            // update stack pointer
            Instruction { dd : NoLoad, kind, z, x },
            // save return address in save-slot, one past the maximum number of locals
            store_local(sm, bottom, max_locals),
        ]
    };
    let label = make_label(class, method, "preamble");
    let preamble = tenyr::BasicBlock { label, insns };

    let blocks = make_blocks_for_method(class, method, sm);
    let name = make_mangled_method_name(class, method);
    Ok(Method { name, preamble, blocks })
}

fn main() -> std::result::Result<(), Box<Error>> {
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    use clap::*;

    let m =
        app_from_crate!()
            .subcommand(
                SubCommand::with_name("translate")
                    .about("Translates JVM .class files into tenyr .tas assembly files")
                    .arg(Arg::with_name("classes")
                            .help("Names .class files as input")
                            .multiple(true)
                            .required(true)
                        )
                )
            .subcommand(
                SubCommand::with_name("mangle")
                    .about("Mangles strings of bytes into valid tenyr symbols")
                    .arg(Arg::with_name("strings")
                            .help("Provides string inputs for mangling")
                            .multiple(true)
                            .required(true)
                        )
                )
            .subcommand(
                SubCommand::with_name("demangle")
                    .about("Decodes mangled tenyr symbols into strings")
                    .arg(Arg::with_name("strings")
                            .help("Provides string inputs for demangling")
                            .multiple(true)
                            .required(true)
                        )
                )
            .get_matches();

    if let Some(m) = m.subcommand_matches("translate") {
        for file in m.values_of("classes").expect("expected at least one input file") {
            let stem = Path::new(&file).with_extension("");
            let out = stem.with_extension("tas");
            let out = out.file_name().expect("failed to format name for output file");
            let stem = stem.to_str().expect("expected Unicode filename");
            let class = classfile_parser::parse_class(&stem).expect("failed to parse class");

            println!("Creating {} from {} ...", out.to_str().expect("expected Unicode filename"), file);
            let mut file = File::create(out)?;
            for method in &class.methods {
                let code = get_method_code(method)?;
                let sm = StackManager::new(code.max_locals, STACK_PTR, STACK_REGS.to_owned());
                let mm = translate_method(&class, method, &sm)?;
                writeln!(file, "{}", mm)?;
            }
        }
    } else if let Some(m) = m.subcommand_matches("mangle") {
        for string in m.values_of("strings").expect("expected at least one string to mangle") {
            println!("{}", mangling::mangle(string.bytes()).expect("failed to mangle"));
        }
    } else if let Some(m) = m.subcommand_matches("demangle") {
        for string in m.values_of("strings").expect("expected at least one string to mangle") {
            let de = mangling::demangle(&string).expect("failed to demangle");
            let st = String::from_utf8(de).expect("expected UTF-8 result from demangle");
            println!("{}", st);
        }
    }

    Ok(())
}

