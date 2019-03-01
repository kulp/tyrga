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

enum_from_primitive! {
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum InstructionType {
    Type0, // [Z] <- [X f Y + I]
    Type1, // [Z] <- [X f I + Y]
    Type2, // [Z] <- [I f X + Y]
    Type3, // [Z] <- [X     + I]
}
}

enum_from_primitive! {
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryOpType {
    NoLoad,     //  Z  <-  ...
    StoreRight, //  Z  -> [...]
    StoreLeft,  // [Z] <-  ...
    LoadRight,  //  Z  <- [...]
}
}

type Immediate = i32;

#[derive(Copy, Clone)]
pub struct Instruction {
    p   : InstructionType,
    dd  : MemoryOpType,
    z   : Register,
    x   : Register,
    y   : Register,
    op  : Opcode,
    imm : Immediate,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use InstructionType::*;
        use Opcode::*;
        use Register::*;
        let mut rhs = match self.p {
            Type3 => self.x.to_string(),
            _     => match self.op {
                BitwiseOr if self.y == A =>
                    format!("{x}", x=self.x),
                _ =>
                    format!("{x} {op:^3} {y}", x=self.x, y=self.y, op=self.op.to_string()),
            },
        };
        if self.imm != 0 {
            rhs.push_str(&format!(" + {imm}", imm=self.imm));
        }

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

    return &[
        (" B  <-  C >>  D + -3" , Instruction { p : Type0, dd : NoLoad, z : B, x : C, y : D, op : ShiftRightArith , imm : -3 }),
        (" B  <-  C >>  D"      , Instruction { p : Type0, dd : NoLoad, z : B, x : C, y : D, op : ShiftRightArith , imm :  0 }),
        (" B  <-  C  |  D + -3" , Instruction { p : Type0, dd : NoLoad, z : B, x : C, y : D, op : BitwiseOr       , imm : -3 }),
        (" B  <-  C  |  D"      , Instruction { p : Type0, dd : NoLoad, z : B, x : C, y : D, op : BitwiseOr       , imm :  0 }),
        (" B  <-  C + -3"       , Instruction { p : Type0, dd : NoLoad, z : B, x : C, y : A, op : BitwiseOr       , imm : -3 }),
        (" B  <-  C"            , Instruction { p : Type0, dd : NoLoad, z : B, x : C, y : A, op : BitwiseOr       , imm :  0 }),
        (" P  <-  C + -4"       , Instruction { p : Type3, dd : NoLoad, z : P, x : C, y : A, op : Add             , imm : -4 }),
        (" P  <-  C"            , Instruction { p : Type3, dd : NoLoad, z : P, x : C, y : A, op : Add             , imm :  0 }),
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
        write!(f, "{}:\n", self.label)?;
        for insn in &self.insns {
            write!(f, "    {}\n", insn.to_string())?;
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

