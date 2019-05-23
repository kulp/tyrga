use crate::exprtree;

use std::convert::Infallible;
use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::fmt;
use std::marker::PhantomData;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Register {
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P,
}

impl fmt::Display for Register {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        (self as &fmt::Debug).fmt(f)
    }
}

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

impl fmt::Display for Opcode {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MemoryOpType {
    NoLoad,     //  Z  <-  ...
    StoreRight, //  Z  -> [...]
    #[allow(dead_code)] // StoreLeft does not get used very often
    StoreLeft,  // [Z] <-  ...
    LoadRight,  //  Z  <- [...]
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TwelveBit;
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TwentyBit;

pub trait BitWidth : Clone + PartialEq + Eq {
    const BITS : u8;
    const UMAX : i32 =  (1 << (Self::BITS    ));
    const IMAX : i32 =  (1 << (Self::BITS - 1)) - 1;
    const IMIN : i32 = -(1 << (Self::BITS - 1));
}

impl BitWidth for TwelveBit { const BITS : u8 = 12; }
impl BitWidth for TwentyBit { const BITS : u8 = 20; }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SizedImmediate<T : BitWidth>(i32, PhantomData<T>);

impl<T : BitWidth> From<i8> for SizedImmediate<T> {
    fn from(val : i8) -> Self { Self(val.into(), PhantomData) }
}

impl<T : BitWidth> From<u8> for SizedImmediate<T> {
    fn from(val : u8) -> Self { Self(val.into(), PhantomData) }
}

impl<T : BitWidth, U> From<U> for Immediate<T>
    where U : Into<SizedImmediate<T>>
{
    fn from(val : U) -> Self { Immediate::Fixed(val.into()) }
}

// Sometimes we need to convert 12-bit unsigned numbers to 12-bit signed immediates
impl Immediate12 {
    const BITS : u8  = TwelveBit::BITS;
    const UMAX : i32 = TwelveBit::UMAX;

    pub fn try_from_bits(val : u16) -> Result<Immediate12, String> {
        if i32::from(val) < Self::UMAX {
            // Convert u16 into an i32 with the same bottom 12 bits
            let mask = if (val & 0x800) != 0 { -1i32 << 12 } else { 0 };
            let val = i32::from(val) | mask;
            Immediate12::try_from(val)
        } else {
            Err(format!("number {} is too big for a {}-bit immediate", val, Self::BITS))
        }
    }
}

impl<T : BitWidth> TryFrom<i32> for SizedImmediate<T> {
    type Error = String;
    fn try_from(val : i32) -> Result<SizedImmediate<T>, Self::Error> {
        if val >= T::IMIN && val <= T::IMAX {
            Ok(SizedImmediate(val, PhantomData))
        } else {
            Err(format!("number {} is too big for a {}-bit immediate", val, T::BITS))
        }
    }
}

impl<T : BitWidth> TryFrom<i32> for Immediate<T> {
    type Error = <SizedImmediate<T> as TryFrom<i32>>::Error;
    fn try_from(val : i32) -> Result<Immediate<T>, Self::Error> {
        SizedImmediate::try_from(val).map(Immediate::Fixed)
    }
}

impl From<i16> for Immediate20 {
    fn from(val : i16) -> Self { Immediate::Fixed(SizedImmediate(val.into(), PhantomData)) }
}

impl From<u16> for Immediate20 {
    fn from(val : u16) -> Self { Immediate::Fixed(SizedImmediate(val.into(), PhantomData)) }
}

impl<T : BitWidth> From<SizedImmediate<T>> for i32 {
    fn from(what : SizedImmediate<T>) -> i32 { what.0 }
}

impl<T : BitWidth> TryFrom<Immediate<T>> for i32 {
    type Error = String;

    fn try_from(what : Immediate<T>) -> Result<i32, Self::Error> {
        match what {
            Immediate::Fixed(s) => Ok(s.into()),
            _ => Err("cannot evaluate non-Fixed Immediate".to_owned()),
        }
    }
}

impl From<Immediate12> for Immediate20 {
    fn from(imm : Immediate12) -> Self {
        use Immediate::*;
        match imm {
            Fixed(imm) => Fixed(SizedImmediate(imm.0, PhantomData)),
            Expr(imm) => Expr(imm),
        }
    }
}

impl<T : BitWidth> Display for SizedImmediate<T> {
    fn fmt(&self, f : &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.0.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Immediate<T : BitWidth> {
    Fixed(SizedImmediate<T>),
    Expr(exprtree::Atom),
}

impl<T : BitWidth> fmt::Display for Immediate<T> {
    fn fmt(&self, f : &mut Formatter) -> Result<(), fmt::Error> {
        match self {
            Immediate::Fixed(x) => write!(f, "{}", x.to_string()),
            Immediate::Expr(x)  => write!(f, "{}", x.to_string()),
        }
    }
}

pub type Immediate12 = Immediate<TwelveBit>;
pub type Immediate20 = Immediate<TwentyBit>;

#[test]
fn test_immediates() {
    assert!(Immediate12::try_from(-(1 << 11) - 1).is_err());
    assert!(Immediate12::try_from(-(1 << 11) - 0).is_ok());
    assert!(Immediate12::try_from( (1 << 11) - 1).is_ok());
    assert!(Immediate12::try_from( (1 << 11) - 0).is_err());

    assert!(Immediate20::try_from(-(1 << 19) - 1).is_err());
    assert!(Immediate20::try_from(-(1 << 19) - 0).is_ok());
    assert!(Immediate20::try_from( (1 << 19) - 1).is_ok());
    assert!(Immediate20::try_from( (1 << 19) - 0).is_err());
}

pub type Immediate32 = i32;

#[derive(Clone, PartialEq, Eq)]
pub enum SmallestImmediate {
    Imm12(Immediate12),
    Imm20(Immediate20),
    Imm32(Immediate32),
}

impl TryFrom<i32> for SmallestImmediate {
    type Error = Infallible;

    fn try_from(n : i32) -> Result<Self, Self::Error> {
        use SmallestImmediate::*;

        Err(0)
            .or_else(|_| Immediate12::try_from(n).map(Imm12))
            .or_else(|_| Immediate20::try_from(n).map(Imm20))
            .or_else(|_| Immediate32::try_from(n).map(Imm32))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InsnGeneral {
    pub y   : Register,
    pub op  : Opcode,
    pub imm : Immediate12,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstructionType {
    Type0(InsnGeneral), // [Z] <- [X f Y + I]
    Type1(InsnGeneral), // [Z] <- [X f I + Y]
    Type2(InsnGeneral), // [Z] <- [I f X + Y]
    Type3(Immediate20), // [Z] <- [X     + I]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub kind : InstructionType,
    pub z    : Register,
    pub x    : Register,
    pub dd   : MemoryOpType,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        use InstructionType::*;
        use InsnGeneral as Gen;
        let (a, b, c) = match &self.kind {
            Type0(Gen { y, imm, .. }) => (self.x  .to_string() ,      y   .to_string(), imm.to_string()),
            Type1(Gen { y, imm, .. }) => (self.x  .to_string() ,      imm .to_string(), y  .to_string()),
            Type2(Gen { y, imm, .. }) => (     imm.to_string() , self.x   .to_string(), y  .to_string()),
            Type3(imm)                => (self.x  .to_string() , "unused" .to_string(), imm.to_string()),
        };
        let rhs = match self.kind {
            Type3(..) if self.x == Register::A
                => c,
            Type3(Immediate::Fixed(ref imm)) if *imm == 0u8.into()
                => a,
            Type3(..)
                => format!("{a} + {c}", a=a, c=c),
            Type0(Gen { op, imm : Immediate::Fixed(ref imm), .. }) if *imm == 0u8.into()
                => format!("{a} {op:^3} {b}", a=a, b=b, op=op.to_string()),
            Type1(Gen { op, y, .. }) | Type2(Gen { op, y, .. }) if y == Register::A
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
fn instruction_test_cases() -> Vec<(&'static str, Instruction)> {
    use InstructionType::*;
    use MemoryOpType::*;
    use Opcode::*;
    use Register::*;

    use Instruction as Insn;
    use InsnGeneral as Gen;

    let zero_20 = Immediate20::from( 0i8);
    let zero_12 = Immediate12::from( 0i8);
    let neg3_12 = Immediate12::from(-3i8);
    let neg4_20 = Immediate20::from(-4i8);

    vec![
        (" B  <-  C >>  D + -3" , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : neg3_12.clone(), y : D, op : ShiftRightArith }) }),
        (" B  <-  C >>  D"      , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : zero_12.clone(), y : D, op : ShiftRightArith }) }),
        (" B  <-  C  |  D + -3" , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : neg3_12.clone(), y : D, op : BitwiseOr       }) }),
        (" B  <-  C  |  D"      , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : zero_12.clone(), y : D, op : BitwiseOr       }) }),
        (" B  <-  C  |  -3 + D" , Insn { dd : NoLoad    , z : B, x : C, kind : Type1(Gen { imm : neg3_12.clone(), y : D, op : BitwiseOr       }) }),
        (" B  <-  C  |  0 + D"  , Insn { dd : NoLoad    , z : B, x : C, kind : Type1(Gen { imm : zero_12.clone(), y : D, op : BitwiseOr       }) }),
        (" B  <-  -3  |  C + D" , Insn { dd : NoLoad    , z : B, x : C, kind : Type2(Gen { imm : neg3_12.clone(), y : D, op : BitwiseOr       }) }),
        (" B  <-  0  |  C + D"  , Insn { dd : NoLoad    , z : B, x : C, kind : Type2(Gen { imm : zero_12.clone(), y : D, op : BitwiseOr       }) }),
        (" B  <-  C  |  A + -3" , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : neg3_12.clone(), y : A, op : BitwiseOr       }) }),
        (" B  <-  C  |  A"      , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : zero_12.clone(), y : A, op : BitwiseOr       }) }),
        (" P  <-  C + -4"       , Insn { dd : NoLoad    , z : P, x : C, kind : Type3(neg4_20.clone()) }),
        (" P  <-  C"            , Insn { dd : NoLoad    , z : P, x : C, kind : Type3(zero_20.clone()) }),
        (" P  -> [C]"           , Insn { dd : StoreRight, z : P, x : C, kind : Type3(zero_20.clone()) }),
        (" P  <- [C]"           , Insn { dd : LoadRight , z : P, x : C, kind : Type3(zero_20.clone()) }),
        ("[P] <-  C"            , Insn { dd : StoreLeft , z : P, x : C, kind : Type3(zero_20.clone()) }),
        (" P  <-  0"            , Insn { dd : NoLoad    , z : P, x : A, kind : Type3(zero_20.clone()) }),
    ]
}

#[test]
fn test_instruction_display() {
    for (string, instruction) in instruction_test_cases() {
        assert_eq!(string, &instruction.to_string());
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasicBlock {
    pub label : String,
    pub insns : Vec<Instruction>,
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for insn in &self.insns {
            writeln!(f, "    {}", insn.to_string())?;
        }
        Ok(())
    }
}

#[test]
fn test_basicblock_display() -> Result<(), Box<std::error::Error>> {
    let (_, insns) : (Vec<_>, Vec<_>) = instruction_test_cases().iter().cloned().unzip();
    let label = "testbb".to_string();
    let bb = BasicBlock { label, insns };
    let ss = bb.to_string();
    let first_line = ss.lines().nth(0).ok_or("no lines in input")?;
    assert_eq!(':', first_line.chars().last().ok_or("no characters in line")?);
    assert_eq!(bb.label, first_line[..first_line.len()-1]);
    assert_eq!(bb.insns.len() + 1, ss.lines().count());

    Ok(())
}

