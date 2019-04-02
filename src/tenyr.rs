use enum_primitive::*;

use std::fmt;

enum_from_primitive! {
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Register {
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P,
}
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

enum_from_primitive! {
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Opcode {
    BitwiseOr       = 0x0, BitwiseOrn       = 0x8,
    BitwiseAnd      = 0x1, BitwiseAndn      = 0x9,
    BitwiseXor      = 0x2, Pack             = 0xa,
    ShiftRightArith = 0x3, ShiftRightLogic  = 0xb,

    Add             = 0x4, Subtract         = 0xc,
    Multiply        = 0x5, ShiftLeft        = 0xd,
    CompareEq       = 0x6, TestBit          = 0xe,
    CompareLt       = 0x7, CompareGe        = 0xf,
}
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Opcode::*;
        let s = match self {
            BitwiseOr       => "|" , BitwiseOrn       => "|~" ,
            BitwiseAnd      => "&" , BitwiseAndn      => "&~" ,
            BitwiseXor      => "^" , Pack             => "^^" ,
            ShiftRightArith => ">>", ShiftRightLogic  => ">>>",

            Add             => "+" , Subtract         => "-"  ,
            Multiply        => "*" , ShiftLeft        => "<<" ,
            CompareEq       => "==", TestBit          => "@"  ,
            CompareLt       => "<" , CompareGe        => ">=" ,
        };
        write!(f, "{}", s)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryOpType {
    NoLoad,     //  Z  <-  ...
    StoreRight, //  Z  -> [...]
    StoreLeft,  // [Z] <-  ...
    LoadRight,  //  Z  <- [...]
}

type Immediate = i32;

#[derive(Clone)]
pub struct TwelveBit;
#[derive(Clone)]
pub struct TwentyBit;

pub trait BitWidth {
    fn size() -> usize;
}

impl BitWidth for TwelveBit { fn size() -> usize { 12 } }
impl BitWidth for TwentyBit { fn size() -> usize { 20 } }

use std::marker::PhantomData;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SizedImmediate<T>(i32, PhantomData<T>);

impl<T> SizedImmediate<T>
    where T: BitWidth
{
    fn new(val : i32) -> Option<SizedImmediate<T>> {
        if val >= -(1 << (T::size() - 1)) && val < (1 << (T::size() - 1)) {
            Some(SizedImmediate(val, PhantomData))
        } else {
            None
        }
    }
}

type Immediate12 = SizedImmediate<TwelveBit>;
type Immediate20 = SizedImmediate<TwentyBit>;

#[derive(Clone)]
pub struct InsnGeneral {
    y   : Register,
    op  : Opcode,
}

#[derive(Clone)]
pub enum InstructionType {
    Type0(InsnGeneral), // [Z] <- [X f Y + I]
    Type1(InsnGeneral), // [Z] <- [X f I + Y]
    Type2(InsnGeneral), // [Z] <- [I f X + Y]
    Type3,              // [Z] <- [X     + I]
}

#[derive(Clone)]
pub struct Instruction {
    kind : InstructionType,
    z    : Register,
    x    : Register,
    dd   : MemoryOpType,
    imm  : Immediate,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use InstructionType::*;
        use InsnGeneral as Gen;
        let (a, b, c) = match self.kind {
            Type0(Gen { y, .. }) => (self.x  .to_string() ,      y   .to_string(), self.imm.to_string()),
            Type1(Gen { y, .. }) => (self.x  .to_string() , self.imm .to_string(),      y  .to_string()),
            Type2(Gen { y, .. }) => (self.imm.to_string() , self.x   .to_string(),      y  .to_string()),
            Type3                => (self.x  .to_string() , "unused" .to_string(), self.imm.to_string()),
        };
        let rhs = match self.kind {
            Type3 if self.imm == 0
                => format!("{a}", a=a),
            Type3
                => format!("{a} + {c}", a=a, c=c),
            Type0(Gen { op, .. }) if self.imm == 0
                => format!("{a} {op:^3} {b}", a=a, b=b, op=op.to_string()),
            Type1(Gen { op, y }) | Type2(Gen { op, y }) if y == Register::A
                => format!("{a} {op:^3} {b}", a=a, b=b, op=op.to_string()),
            Type0(Gen { op, .. }) |
            Type1(Gen { op, .. }) |
            Type2(Gen { op, .. })
                => format!("{a} {op:^3} {b} + {c}", a=a, b=b, c=c, op=op.to_string()),
        };

        use MemoryOpType::*;
        match self.dd {
            NoLoad     => write!(f, " {z}  <-  {rhs}" , z=self.z, rhs=rhs),
            StoreRight => write!(f, " {z}  -> [{rhs}]", z=self.z, rhs=rhs),
            StoreLeft  => write!(f, "[{z}] <-  {rhs}" , z=self.z, rhs=rhs),
            LoadRight  => write!(f, " {z}  <- [{rhs}]", z=self.z, rhs=rhs),
        }
    }
}

#[cfg(test)]
fn instruction_test_cases() -> &'static [(&'static str, Instruction)] {
    use InstructionType::*;
    use MemoryOpType::*;
    use Opcode::*;
    use Register::*;

    use Instruction as Insn;
    use InsnGeneral as Gen;
    return &[
        (" B  <-  C >>  D + -3" , Insn { dd : NoLoad    , z : B, x : C, imm : -3, kind : Type0(Gen { y : D, op : ShiftRightArith }) }),
        (" B  <-  C >>  D"      , Insn { dd : NoLoad    , z : B, x : C, imm :  0, kind : Type0(Gen { y : D, op : ShiftRightArith }) }),
        (" B  <-  C  |  D + -3" , Insn { dd : NoLoad    , z : B, x : C, imm : -3, kind : Type0(Gen { y : D, op : BitwiseOr       }) }),
        (" B  <-  C  |  D"      , Insn { dd : NoLoad    , z : B, x : C, imm :  0, kind : Type0(Gen { y : D, op : BitwiseOr       }) }),
        (" B  <-  C  |  -3 + D" , Insn { dd : NoLoad    , z : B, x : C, imm : -3, kind : Type1(Gen { y : D, op : BitwiseOr       }) }),
        (" B  <-  C  |  0 + D"  , Insn { dd : NoLoad    , z : B, x : C, imm :  0, kind : Type1(Gen { y : D, op : BitwiseOr       }) }),
        (" B  <-  -3  |  C + D" , Insn { dd : NoLoad    , z : B, x : C, imm : -3, kind : Type2(Gen { y : D, op : BitwiseOr       }) }),
        (" B  <-  0  |  C + D"  , Insn { dd : NoLoad    , z : B, x : C, imm :  0, kind : Type2(Gen { y : D, op : BitwiseOr       }) }),
        (" B  <-  C  |  A + -3" , Insn { dd : NoLoad    , z : B, x : C, imm : -3, kind : Type0(Gen { y : A, op : BitwiseOr       }) }),
        (" B  <-  C  |  A"      , Insn { dd : NoLoad    , z : B, x : C, imm :  0, kind : Type0(Gen { y : A, op : BitwiseOr       }) }),
        (" P  <-  C + -4"       , Insn { dd : NoLoad    , z : P, x : C, imm : -4, kind : Type3                                      }),
        (" P  <-  C"            , Insn { dd : NoLoad    , z : P, x : C, imm :  0, kind : Type3                                      }),
        (" P  -> [C]"           , Insn { dd : StoreRight, z : P, x : C, imm :  0, kind : Type3                                      }),
        (" P  <- [C]"           , Insn { dd : LoadRight , z : P, x : C, imm :  0, kind : Type3                                      }),
        ("[P] <-  C"            , Insn { dd : StoreLeft , z : P, x : C, imm :  0, kind : Type3                                      }),
    ];
}

#[test]
fn test_instruction_display() {
    for (string, instruction) in instruction_test_cases() {
        assert_eq!(string, &instruction.to_string());
    }
}

pub struct BasicBlock {
    label : String,
    insns : Vec<Instruction>,
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for insn in &self.insns {
            writeln!(f, "    {}", insn.to_string())?;
        }
        Ok(())
    }
}

#[test]
fn test_basicblock_display() {
    let (_, insns) : (Vec<_>, Vec<_>) = instruction_test_cases().iter().cloned().unzip();
    let label = "testbb".to_string();
    let bb = BasicBlock { label, insns };
    let ss = bb.to_string();
    let first_line = ss.lines().nth(0).unwrap();
    assert_eq!(':', first_line.chars().last().unwrap());
    assert_eq!(bb.label, first_line[..first_line.len()-1]);
    assert_eq!(bb.insns.len() + 1, ss.lines().count());
}

