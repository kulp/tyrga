mod deadcode;
mod exprtree;
mod jvmtypes;
mod mangling;
mod tenyr;

use std::collections::BTreeSet;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::ops::Range;

use jvmtypes::*;

use classfile_parser::ClassFile;

use tenyr::Register;

#[derive(Clone, Debug)]
pub struct StackManager {
    ptr : Register,
    stack : Vec<Register>,
    count : usize,
    frozen : usize,
}

// Someday we will be able to find operands in a stack frame
#[derive(Copy,Clone,Debug,PartialEq,Eq)]
pub enum OperandLocation {
    Register(tenyr::Register),
    Stacked(usize),
}

impl From<Register> for OperandLocation {
    fn from(r : Register) -> OperandLocation { OperandLocation::Register(r) }
}

type StackActions = Vec<tenyr::Instruction>;

// This simple StackManager implementation does not do spilling to nor reloading from memory.
// For now, it panics if we run out of free registers.
impl StackManager {
    pub fn new(sp : Register, r : Vec<Register>) -> StackManager {
        StackManager { count : 0, frozen : 0, ptr : sp, stack : r }
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn reserve(&mut self, n : usize) -> StackActions {
        assert!(self.count + n <= self.stack.len(), "operand stack overflow");
        self.count += n;
        vec![] // TODO support spilling
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn release(&mut self, n : usize) -> StackActions {
        assert!(self.count >= n, "operand stack underflow");
        self.count -= n;
        vec![] // TODO support reloading
    }

    pub fn depth(&self) -> usize { self.count }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn empty(&mut self) -> StackActions {
        let mut v = self.release(self.count);
        v.extend(self.thaw()); // capture instruction moving stack pointer, if any
        v
    }

    fn get_reg(&self, which : usize) -> Register {
        assert!(which <= self.count, "attempt to access nonexistent depth");
        // indexing is relative to top of stack, counting backward
        self.stack[(self.count - which - 1) % self.stack.len()]
    }

    pub fn get(&self, which : usize) -> OperandLocation {
        // TODO handle Stacked
        assert!(which <= self.stack.len(), "attempt to access register deeper than register depth");
        // indexing is relative to top of stack, counting backward
        self.get_reg(which).into()
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    fn set_watermark(&mut self, level : usize) -> StackActions {
        // TODO remove casts

        // `self.frozen` counts the number of spilled registers (from bottom of operand stack) and
        // takes values in the interval [ 0, +infinity )
        // `level` requests a number of free registers (from top of operand stack) and takes values
        // in the interval [ 0, min(self.stack.len(), self.count) )
        // `unfrozen` derives the current watermark in the same units as `level`
        let level = std::cmp::min(self.count, level) as i32; // cannot freeze more than we have
        let count = self.count as i32;
        let frozen = self.frozen as i32;
        let unfrozen = count - frozen;

        use tenyr::*;
        use tenyr::InstructionType::*;
        use tenyr::MemoryOpType::*;
        let stack_ptr = self.ptr;

        let stack_movement = -(unfrozen as i32 - level as i32) as i32; // TODO check overflow issues here
        if stack_movement == 0 {
            return vec![];
        }

        let new_frozen = frozen as i32 - stack_movement;
        self.frozen = new_frozen as usize;

        let make_insn = |reg, offset| Instruction { dd : NoLoad, kind : Type3(Immediate20::new(offset).unwrap()), z : reg, x : stack_ptr };
        let make_move = |i, offset| make_insn(self.get_reg(i as usize), i + offset + 1);
        // Only one of { `freezing`, `thawing` } will have any elements in it
        let freezing = (level..unfrozen).map(|i| Instruction { dd : StoreRight, ..make_move(i, 0) });
        let thawing  = (unfrozen..level).map(|i| Instruction { dd : LoadRight , ..make_move(i, -stack_movement) });
        let update = make_insn(stack_ptr, stack_movement);

        std::iter::once(update)
            .chain(freezing)
            .chain(thawing)
            .collect()
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn freeze(&mut self) -> StackActions {
        self.set_watermark(0)
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn thaw(&mut self) -> StackActions {
        self.set_watermark(self.stack.len())
    }
}

#[test]
fn test_get_reg() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(O, v.clone());
    let _ = sm.reserve(v.len());
    assert_eq!(&v[0], &sm.get_reg(v.len() - 1));
}


#[test]
#[should_panic(expected="underflow")]
fn test_underflow() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(O, v);
    let _ = sm.reserve(3);
    let _ = sm.release(4);
}

#[test]
#[should_panic(expected="overflow")]
fn test_overflow() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let len = v.len();
    let mut sm = StackManager::new(O, v);
    let _ = sm.reserve(len + 1);
}

#[test]
fn test_normal_stack() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let t = v.clone();
    let mut sm = StackManager::new(O, v);
    let off = 3;
    let _ = sm.reserve(off);
    assert_eq!(sm.get(0), t[off - 1].into());
}

#[test]
fn test_watermark() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(O, v);
    let _ = sm.reserve(4);

    let insns = sm.set_watermark(0);
    assert!(insns.len() == 5);
    let insns = sm.set_watermark(0);
    assert!(insns.len() == 0);

    let insns = sm.set_watermark(3);
    assert!(insns.len() == 4);
    let insns = sm.set_watermark(3);
    assert!(insns.len() == 0);

    let insns = sm.set_watermark(1);
    assert!(insns.len() == 3);
    let insns = sm.set_watermark(1);
    assert!(insns.len() == 0);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Destination {
    Successor,
    Address(usize),
    Return,
}

type Namer = Fn(usize) -> String;
type Caller = Fn(u16) -> String;
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
    (addr.clone(), v, dest)
}

fn make_instructions(sm : &mut StackManager, (addr, op) : (&usize, &Operation), target_namer : &Namer, method_namer : &Caller)
    -> MakeInsnResult
{
    use Operation::*;
    use jvmtypes::SwitchParams::*;
    use tenyr::Immediate12;
    use tenyr::Immediate20;
    use tenyr::Instruction;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::*;

    let stack_ptr = Register::O;
    let frame_ptr = Register::N;

    // We need to track destinations and return them so that the caller can track stack state
    // through the chain of control flow, possibly cloning the StackManager state along the way to
    // follow multiple destinations. Each basic block needs to be visited only once, however, since
    // the JVM guarantees that every instance of every instruction within a method always sees the
    // same depth of the operand stack every time that instance is executed.
    let default_dest = vec![Destination::Successor];

    let get_reg = |t| match t {
        OperandLocation::Register(r) => r,
        _ => panic!("unsupported location {:?}", t),
    };

    let make_imm20 = |n| Immediate20::new(n).unwrap();
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

    let make_mov   = |to, from| Instruction { dd : NoLoad, kind : Type3(Immediate20::ZERO), z : to, x : from };
    let make_load  = |to, from| Instruction { dd : LoadRight , ..make_mov(to, from) };
    let make_store = |lhs, rhs| Instruction { dd : StoreRight, ..make_mov(lhs, rhs) };

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
            (addr.clone(), v, default_dest)
        },
        Yield { kind } => {
            let ret = Instruction { kind : Type3(pos1_20), ..make_load(Register::P, stack_ptr) };
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
                        make_store(top, frame_ptr),
                        ret
                    ]
                },
                Double | Long => {
                    let top = get_reg(sm.get(0));
                    let sec = get_reg(sm.get(1));
                    vec![
                        make_store(sec, frame_ptr),
                        Instruction { kind : Type3(neg1_20), ..make_store(top, frame_ptr) },
                        ret
                    ]
                },
            };
            v.extend(sm.empty());
            (addr.clone(), v, vec![ Destination::Return ])
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
                (addr.clone(), v, default_dest)
            },
        LoadLocal { kind, index } | StoreLocal { kind, index }
            if kind == JType::Int || kind == JType::Object
            => {
                let mut v = Vec::with_capacity(10);
                let index = i32::from(index);
                if let LoadLocal { .. } = *op { v.extend(sm.reserve(1)); }
                {
                    use tenyr::*;
                    let x = frame_ptr;
                    let z = get_reg(sm.get(0));
                    let dd = match *op {
                        LoadLocal  { .. } => MemoryOpType::LoadRight,
                        StoreLocal { .. } => MemoryOpType::StoreRight,
                        _ => unreachable!(),
                    };
                    let imm = Immediate20::new(-index).unwrap();
                    v.push(Instruction { kind : Type3(imm), x, z, dd });
                }
                if let StoreLocal { .. } = *op { v.extend(sm.release(1)); }
                (addr.clone(), v, default_dest)
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
                Instruction { kind : Type3(make_imm20(-index)), ..make_load(temp_reg, frame_ptr) },
                Instruction { kind : Type1(InsnGeneral { y, op, imm }), ..make_mov(temp_reg, temp_reg) },
                Instruction { kind : Type3(make_imm20(-index)), ..make_store(temp_reg, frame_ptr) },
            ]);
            v.extend(sm.release(1));

            (addr.clone(), v, default_dest)
        },
        Branch { kind : JType::Int, ops : OperandCount::_1, way, target } => {
            use tenyr::*;
            use tenyr::Immediate12;
            use tenyr::Instruction;
            use tenyr::InstructionType::*;
            use tenyr::MemoryOpType;

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

            make_int_branch(sm, addr.clone(), false, target, target_namer, &mut op1)
        },
        Branch { kind : JType::Int, ops : OperandCount::_2, way, target } => {
            use tenyr::*;
            use tenyr::Immediate12;
            use tenyr::Instruction;
            use tenyr::InstructionType::*;
            use tenyr::MemoryOpType;

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

            make_int_branch(sm, addr.clone(), invert, target, target_namer, &mut op2)
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
                    make_int_branch(sm, addr.clone(), false, (target + here) as u16, target_namer, &mut maker(compare));
                (insns, dests)
            }).unzip();

            let i = i.concat();
            let d = d.concat();

            insns.extend(i);
            dests.extend(d);

            insns.push(make_jump(there));
            dests.push(Destination::Address(usize::from(there)));

            (addr.clone(), insns, dests)
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
                make_int_branch(sm, addr.clone(), false, there, target_namer, &mut maker(&Type1, low));
            let (_hi_addr, hi_insns, hi_dests) =
                make_int_branch(sm, addr.clone(), false, there, target_namer, &mut maker(&Type2, high));

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
        Jump { target } => (addr.clone(), vec![ make_jump(target) ], vec![ Destination::Address(target as usize) ]),
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
            (addr.clone(), v, default_dest)
        },
        Noop => (addr.clone(), vec![ make_mov(Register::A, Register::A) ], default_dest),
        Length => {
            // TODO document layout of arrays
            // This implementation assumes a reference to an array points to its first element, and
            // that one word below that element is a word containing the number of elements.
            let top = get_reg(sm.get(0));
            let insn = Instruction { kind : Type3(neg1_20), ..make_load(top, top) };
            (addr.clone(), vec![ insn ], default_dest)
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
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(O, v);
    let op = Operation::Constant { kind : JType::Int, value : 5 };
    let namer = |x| format!("{}:{}", "test", x);
    let caller = |x| format!("{}_{}", "test", x);
    let insn = make_instructions(&mut sm, (&0, &op), &namer, &caller);
    let imm = Immediate20::new(5).unwrap();
    assert_eq!(insn.1, vec![ Instruction { kind: Type3(imm), z: C, x: A, dd: NoLoad } ]);
    assert_eq!(insn.1[0].to_string(), " C  <-  5");
}

#[derive(Debug)]
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

use classfile_parser::method_info::MethodInfo;
fn get_ranges_for_method(class : &ClassFile, method : &MethodInfo)
    -> Result<(Vec<Range<usize>>, BTreeMap<usize, Operation>)>
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

    let code = code_attribute_parser(&method.attributes[0].info).map_err(generic_error)?.1;
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

fn make_callable_name(class : &ClassFile, pool_index : u16) -> String {
    use classfile_parser::constant_info::ConstantInfo::*;

    let get_constant = |n| &class.const_pool[usize::from(n) - 1];
    let get_string = |i|
        match get_constant(i) {
            Utf8(u) => Some(u.utf8_string.to_string()),
            _ => None,
        };

    if let MethodRef(mr) = get_constant(pool_index) {
        if let Class(cl) = get_constant(mr.class_index) {
            if let NameAndType(nt) = get_constant(mr.name_and_type_index) {
                return mangling::mangle(
                    vec![
                        get_string(cl.name_index).expect("bad class name"),
                        get_string(nt.name_index).expect("bad method name"),
                        get_string(nt.descriptor_index).expect("bad method descriptor"),
                    ].join(":").bytes()
                ).expect("failed to mangle");
            }
        }
    }

    panic!("error during constant pool lookup");
}

fn make_unique_method_name(class : &ClassFile, method : &MethodInfo) -> String {
    use classfile_parser::constant_info::ConstantInfo::*;

    let get_constant = |n| &class.const_pool[usize::from(n) - 1];
    let get_string = |i|
        match get_constant(i) {
            Utf8(u) => Some(u.utf8_string.to_string()),
            _ => None,
        };

    let cl = match get_constant(class.this_class) { Class(c) => c, _ => panic!("not a class") };
    vec![
        get_string(cl.name_index).expect("bad class name"),
        get_string(method.name_index).expect("bad method name"),
        get_string(method.descriptor_index).expect("bad method descriptor"),
    ].join(":")
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
            // TODO obivate clones
            let class = class.clone();
            let method = method.clone();
            move |x : usize| make_label(&class, &method, &x.to_string())
        };

        let caller = {
            // TODO obivate clones
            let class = class.clone();
            move |x| make_callable_name(&class, x)
        };

        let block : Vec<_> = ops.range(which.clone()).map(|x| make_instructions(&mut sm, x, &namer, &caller)).collect();
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
    for method in class.methods.iter().filter(|m| !make_unique_method_name(&class, m).contains(":<")) {
        let v = { use Register::*; vec![ C, D, E, F, G, H, I, J, K, L, M ] }; // TODO get range working
        let sm = StackManager::new(Register::O, v);
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

fn main() -> std::result::Result<(), Box<Error>> {
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    use clap::*;

    let m =
        app_from_crate!()
            .arg(Arg::with_name("classes")
                    .help("Names .class files as input")
                    .multiple(true)
                    .required(true))
            .get_matches();

    for file in m.values_of("classes").expect("expected at least one input file") {
        let stem = Path::new(&file).with_extension("");
        let out = stem.with_extension("tas");
        let out = out.file_name().expect("failed to format name for output file");
        let stem = stem.to_str().expect("expected Unicode filename");
        let class = classfile_parser::parse_class(&stem).expect("failed to parse class");

        println!("Creating {} from {} ...", out.to_str().expect("expected Unicode filename"), file);
        let mut file = File::create(out)?;
        for method in class.methods.iter().filter(|m| !make_unique_method_name(&class, m).contains(":<")) {
            let v = { use Register::*; vec![ C, D, E, F, G, H, I, J, K, L, M ] }; // TODO get range working
            let sm = StackManager::new(Register::O, v);
            let bbs = make_blocks_for_method(&class, method, &sm);
            for bb in &bbs {
                write!(file, "{}", bb)?;
            }

            write!(file, "\n")?;
        }
    }

    Ok(())
}

