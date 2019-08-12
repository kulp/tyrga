use std::convert::TryFrom;

use classfile_parser::code_attribute::Instruction;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum JType {
    Int,
    Long,
    Float,
    Double,
    Object,
    Byte,
    Char,
    Short,
    Void,
}

impl JType {
    pub fn size(self) -> u8 {
        use JType::*;
        match self {
            Long | Double => 2,
            Int | Float | Object | Byte | Char | Short => 1,
            Void => 0,
        }
    }
}

impl TryFrom<JType> for char {
    type Error = &'static str;
    fn try_from(t : JType) -> Result<Self, Self::Error> {
        use JType::*;
        match t {
            Int    => Ok('I'),
            Long   => Ok('J'),
            Float  => Ok('F'),
            Double => Ok('D'),
            Byte   => Ok('Z'),
            Char   => Ok('C'),
            Short  => Ok('S'),
            Void   => Ok('V'),
            _  => Err("no such mapping"),
        }
    }
}

impl TryFrom<char> for JType {
    type Error = &'static str;
    fn try_from(ch : char) -> Result<Self, Self::Error> {
        use JType::*;
        match ch {
            'I' => Ok(Int),
            'J' => Ok(Long),
            'F' => Ok(Float),
            'D' => Ok(Double),
            'L' => Ok(Object),
            '[' => Ok(Object),
            'Z' => Ok(Byte),
            'C' => Ok(Char),
            'S' => Ok(Short),
            'V' => Ok(Void),
            _ => Err("no such mapping"),
        }
    }
}

impl TryFrom<u8> for JType {
    type Error = &'static str;
    fn try_from(a : u8) -> Result<Self, Self::Error> {
        match a {
            4  /* Boolean */ => Ok(JType::Byte), // arbitrary mapping for Boolean
            5  /* Char    */ => Ok(JType::Char),
            6  /* Float   */ => Ok(JType::Float),
            7  /* Double  */ => Ok(JType::Double),
            8  /* Byte    */ => Ok(JType::Byte),
            9  /* Short   */ => Ok(JType::Short),
            10 /* Int     */ => Ok(JType::Int),
            11 /* Long    */ => Ok(JType::Long),
            _  => Err("no such mapping"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Comparison {
    Eq,
    Ne,
    Lt,
    Ge,
    Gt,
    Le,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ArithmeticOperation {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Shl,
    Shr,
    Ushr,
    And,
    Or,
    Xor,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum StackOperation {
    Pop,
    Dup,
    DupX1,
    DupX2,
    Swap,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NanComparisons {
    Greater,
    Less,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum OperandCount {
    _1 = 1,
    _2 = 2,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VarOp {
    Get,
    Put,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VarKind {
    Static,
    Field,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InvokeKind {
    Dynamic,
    Interface(u8),
    Special,
    Static,
    Virtual,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SwitchParams {
    Lookup { default : i32, pairs : Vec<(i32, i32)> },
    Table  { default : i32, low : i32, high : i32, offsets : Vec<i32> },
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Indirection<T> {
    Explicit(T),
    Indirect(u16),
}

pub type ArrayKind = Indirection<JType>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExplicitConstant {
    pub kind : JType,
    pub value : i16,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AllocationKind {
    Array { kind : ArrayKind, dims : u8 },
    Element { index : u16 },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    Allocation  (AllocationKind),
    Arithmetic  { kind : JType, op : ArithmeticOperation },
    Branch      { kind : JType, ops : OperandCount, way : Comparison, target : u16 },
    Compare     { kind : JType, nans : Option<NanComparisons> },
    Constant    (Indirection<ExplicitConstant>),
    Conversion  { from : JType, to : JType },
    Increment   { index : u16, value : i16 },
    Invocation  { kind : InvokeKind, index : u16 },
    Jump        { target : u16 },
    Length,     /* i.e. arraylength */
    LoadArray   (JType),
    LoadLocal   { kind : JType, index : u16 },
    Noop,
    StackOp     { size : OperandCount, op : StackOperation },
    StoreArray  (JType),
    StoreLocal  { kind : JType, index : u16 },
    Switch      (SwitchParams),
    VarAction   { op : VarOp, kind : VarKind, index : u16 },
    Yield       { kind : JType }, /* i.e. return */

    Unhandled   (Instruction),
}

trait OperandType {
    fn get_operand_type(&self) -> Option<JType>;
}

impl OperandType for Instruction {
    fn get_operand_type(&self) -> Option<JType> {
        use Instruction::*;
        use JType::*;
        match self {
            Aconstnull
                | Aload(_) | AloadWide(_)
                | Aload0 | Aload1 | Aload2 | Aload3
                | Astore(_) | AstoreWide(_)
                | Astore0 | Astore1 | Astore2 | Astore3
                | Areturn
                | IfAcmpeq(_) | IfAcmpne(_) | Ifnull(_) | Ifnonnull(_)
                => Some(Object),
            Iconstm1 | Iconst0 | Iconst1 | Iconst2 | Iconst3 | Iconst4 | Iconst5
                | Iload(_) | IloadWide(_)
                | Iload0 | Iload1 | Iload2 | Iload3
                | Istore(_) | IstoreWide(_)
                | Istore0 | Istore1 | Istore2 | Istore3
                | Iadd | Isub | Imul | Idiv | Irem | Ineg
                | Ishl | Ishr | Iushr | Iand | Ior | Ixor
                | Ireturn
                | Ifeq(_) | Ifne(_) | Iflt(_) | Ifge(_) | Ifgt(_) | Ifle(_)
                | IfIcmpeq(_) | IfIcmpne(_) | IfIcmplt(_) | IfIcmpge(_) | IfIcmpgt(_) | IfIcmple(_)
                => Some(Int),
            Lconst0 | Lconst1
                | Lload(_) | LloadWide(_)
                | Lload0 | Lload1 | Lload2 | Lload3
                | Lstore(_) | LstoreWide(_)
                | Lstore0 | Lstore1 | Lstore2 | Lstore3
                | Ladd | Lsub | Lmul | Ldiv | Lrem | Lneg
                | Lshl | Lshr | Lushr | Land | Lor | Lxor
                | Lreturn
                => Some(Long),
            Fconst0 | Fconst1 | Fconst2
                | Fload(_) | FloadWide(_)
                | Fload0 | Fload1 | Fload2 | Fload3
                | Fstore(_) | FstoreWide(_)
                | Fstore0 | Fstore1 | Fstore2 | Fstore3
                | Fadd | Fsub | Fmul | Fdiv | Frem | Fneg
                | Freturn
                => Some(Float),
            Dconst0 | Dconst1
                | Dload(_) | DloadWide(_)
                | Dload0 | Dload1 | Dload2 | Dload3
                | Dstore(_) | DstoreWide(_)
                | Dstore0 | Dstore1 | Dstore2 | Dstore3
                | Dadd | Dsub | Dmul | Ddiv | Drem | Dneg
                | Dreturn
                => Some(Double),
            Return
                => Some(Void),

                  I2l | I2f | I2d | I2b | I2c | I2s => Some(Int),
            L2i |       L2f | L2d                   => Some(Long),
            F2i | F2l |       F2d                   => Some(Float),
            D2i | D2l | D2f                         => Some(Double),

            _ => None,
        }
    }
}

// returns any Operation parsed and the number of bytes consumed
pub fn decode_insn(insn : (usize, Instruction)) -> (usize, Operation) {
    use AllocationKind::*;
    use JType::*;
    use Indirection::*;
    use Instruction::*;
    use Operation::*;
    use SwitchParams::*;

    let (addr, insn) = insn;

    let make_constant = |kind, value|
        Constant(Explicit(ExplicitConstant { kind, value }));

    let kind_of = |insn : &Instruction|
        insn.get_operand_type().unwrap_or_else(|| unreachable!("kind must be valid but is not"));

    let kind = || kind_of(&insn);

    let op = match insn {
        Nop => Noop,

        Iconstm1   => make_constant(kind(), -1),

        Aconstnull | Iconst0 | Lconst0 | Fconst0 | Dconst0
            => make_constant(kind(),  0),
        Iconst1 | Lconst1 | Fconst1 | Dconst1
            => make_constant(kind(),  1),
        Iconst2 | Fconst2
            => make_constant(kind(),  2),

        Iconst3 => make_constant(kind(),  3),
        Iconst4 => make_constant(kind(),  4),
        Iconst5 => make_constant(kind(),  5),

        Bipush(v) => make_constant(Int, v.into()),
        Sipush(v) => make_constant(Int, v),

        Iload(index) | Lload(index) | Fload(index) | Dload(index) | Aload(index)
            => LoadLocal { kind : kind(), index : index.into() },

        IloadWide(index) | LloadWide(index) | FloadWide(index) | DloadWide(index) | AloadWide(index)
            => LoadLocal { kind : kind(), index },

        Dload0 | Fload0 | Iload0 | Lload0 | Aload0 => LoadLocal { kind : kind(), index : 0 },
        Dload1 | Fload1 | Iload1 | Lload1 | Aload1 => LoadLocal { kind : kind(), index : 1 },
        Dload2 | Fload2 | Iload2 | Lload2 | Aload2 => LoadLocal { kind : kind(), index : 2 },
        Dload3 | Fload3 | Iload3 | Lload3 | Aload3 => LoadLocal { kind : kind(), index : 3 },

        Iaload => LoadArray(Int),
        Laload => LoadArray(Long),
        Faload => LoadArray(Float),
        Daload => LoadArray(Double),
        Aaload => LoadArray(Object),
        Baload => LoadArray(Byte),
        Caload => LoadArray(Char),
        Saload => LoadArray(Short),

        Istore(index) | Lstore(index) | Fstore(index) | Dstore(index) | Astore(index)
            => StoreLocal { kind : kind(), index : index.into() },

        IstoreWide(index) | LstoreWide(index) | FstoreWide(index) | DstoreWide(index) | AstoreWide(index)
            => StoreLocal { kind : kind(), index },

        Dstore0 | Fstore0 | Istore0 | Lstore0 | Astore0 => StoreLocal { kind : kind(), index : 0 },
        Dstore1 | Fstore1 | Istore1 | Lstore1 | Astore1 => StoreLocal { kind : kind(), index : 1 },
        Dstore2 | Fstore2 | Istore2 | Lstore2 | Astore2 => StoreLocal { kind : kind(), index : 2 },
        Dstore3 | Fstore3 | Istore3 | Lstore3 | Astore3 => StoreLocal { kind : kind(), index : 3 },

        Iastore => StoreArray(Int),
        Lastore => StoreArray(Long),
        Fastore => StoreArray(Float),
        Dastore => StoreArray(Double),
        Aastore => StoreArray(Object),
        Bastore => StoreArray(Byte),
        Castore => StoreArray(Char),
        Sastore => StoreArray(Short),

        Pop     => StackOp { op : StackOperation::Pop  , size : OperandCount::_1 },
        Pop2    => StackOp { op : StackOperation::Pop  , size : OperandCount::_2 },
        Dup     => StackOp { op : StackOperation::Dup  , size : OperandCount::_1 },
        Dupx1   => StackOp { op : StackOperation::DupX1, size : OperandCount::_1 },
        Dupx2   => StackOp { op : StackOperation::DupX2, size : OperandCount::_1 },
        Dup2    => StackOp { op : StackOperation::Dup  , size : OperandCount::_2 },
        Dup2x1  => StackOp { op : StackOperation::DupX1, size : OperandCount::_2 },
        Dup2x2  => StackOp { op : StackOperation::DupX2, size : OperandCount::_2 },
        Swap    => StackOp { op : StackOperation::Swap , size : OperandCount::_2 },

        Iadd | Ladd | Fadd | Dadd
            | Isub | Lsub | Fsub | Dsub
            | Imul | Lmul | Fmul | Dmul
            | Idiv | Ldiv | Fdiv | Ddiv
            | Irem | Lrem | Frem | Drem
            | Ineg | Lneg | Fneg | Dneg
            | Ishl | Lshl
            | Ishr | Lshr
            | Iushr| Lushr
            | Iand | Land
            | Ior  | Lor
            | Ixor | Lxor
            => {
                let op = match insn {
                    Iadd | Ladd | Fadd | Dadd => ArithmeticOperation::Add,
                    Isub | Lsub | Fsub | Dsub => ArithmeticOperation::Sub,
                    Imul | Lmul | Fmul | Dmul => ArithmeticOperation::Mul,
                    Idiv | Ldiv | Fdiv | Ddiv => ArithmeticOperation::Div,
                    Irem | Lrem | Frem | Drem => ArithmeticOperation::Rem,
                    Ineg | Lneg | Fneg | Dneg => ArithmeticOperation::Neg,
                    Ishl | Lshl               => ArithmeticOperation::Shl,
                    Ishr | Lshr               => ArithmeticOperation::Shr,
                    Iushr| Lushr              => ArithmeticOperation::Ushr,
                    Iand | Land               => ArithmeticOperation::And,
                    Ior  | Lor                => ArithmeticOperation::Or,
                    Ixor | Lxor               => ArithmeticOperation::Xor,
                    _ => unreachable!(),
                };
                Arithmetic { kind : kind(), op }
            },

        Iinc     { index, value } => Increment { index : index.into(), value : value.into() },
        IincWide { index, value } => Increment { index, value },

        I2l | I2f | I2d | L2i | L2f | L2d | F2i | F2l | F2d | D2i | D2l | D2f | I2b | I2c | I2s
            => {
                let to = match insn {
                    I2b                   => Byte,
                    I2c                   => Char,
                    I2s                   => Short,

                          L2i | F2i | D2i => Int,
                    I2l |       F2l | D2l => Long,
                    I2f | L2f |       D2f => Float,
                    I2d | L2d | F2d       => Double,

                    _ => unreachable!(),
                };

                Conversion { from : kind(), to }
            },

        Lcmp    => Compare { kind : Long  , nans : None                          },
        Fcmpl   => Compare { kind : Float , nans : Some(NanComparisons::Less   ) },
        Fcmpg   => Compare { kind : Float , nans : Some(NanComparisons::Greater) },
        Dcmpl   => Compare { kind : Double, nans : Some(NanComparisons::Less   ) },
        Dcmpg   => Compare { kind : Double, nans : Some(NanComparisons::Greater) },

        Ifeq(off) | Ifne(off) | Iflt(off) | Ifge(off) | Ifgt(off) | Ifle(off)
            | IfIcmpeq(off) | IfIcmpne(off)
            | IfIcmplt(off) | IfIcmpge(off) | IfIcmpgt(off) | IfIcmple(off)
            | IfAcmpeq(off) | IfAcmpne(off)
            | Ifnull(off) | Ifnonnull(off)
            => {
                let target = (addr as isize + off as isize) as u16;
                let way = match insn {
                    Ifeq(_) | IfIcmpeq(_) | IfAcmpeq(_) | Ifnull(_)    => Comparison::Eq,
                    Ifne(_) | IfIcmpne(_) | IfAcmpne(_) | Ifnonnull(_) => Comparison::Ne,
                    Iflt(_) | IfIcmplt(_)                              => Comparison::Lt,
                    Ifge(_) | IfIcmpge(_)                              => Comparison::Ge,
                    Ifgt(_) | IfIcmpgt(_)                              => Comparison::Gt,
                    Ifle(_) | IfIcmple(_)                              => Comparison::Le,

                    _ => unreachable!(),
                };
                let ops = match insn {
                    Ifeq(_) | Ifne(_) | Iflt(_) | Ifge(_) | Ifgt(_) | Ifle(_)
                        | Ifnull(_) | Ifnonnull(_)
                        => OperandCount::_1,
                    IfIcmpeq(_) | IfIcmpne(_)
                        | IfIcmplt(_) | IfIcmpge(_) | IfIcmpgt(_) | IfIcmple(_)
                        | IfAcmpeq(_) | IfAcmpne(_)
                        => OperandCount::_2,
                    _ => unreachable!(),
                };

                Branch { kind : kind(), way, ops, target }
            },

        Goto(off) => Jump { target : (addr as isize + off as isize) as u16 }, // TODO remove casts
        GotoW(off) => Jump { target : (addr as isize + off as isize) as u16 }, // TODO remove casts

        Ireturn | Lreturn | Freturn | Dreturn | Areturn | Return
            => Yield { kind : kind() },

        Getstatic(index) => VarAction { op : VarOp::Get, kind : VarKind::Static, index },
        Putstatic(index) => VarAction { op : VarOp::Put, kind : VarKind::Static, index },
        Getfield(index)  => VarAction { op : VarOp::Get, kind : VarKind::Field , index },
        Putfield(index)  => VarAction { op : VarOp::Put, kind : VarKind::Field , index },

        Invokevirtual(index) => Invocation { kind : InvokeKind::Virtual, index },
        Invokespecial(index) => Invocation { kind : InvokeKind::Special, index },
        Invokestatic(index)  => Invocation { kind : InvokeKind::Static , index },
        Invokedynamic(index) => Invocation { kind : InvokeKind::Dynamic, index },
        Invokeinterface { index, count } => Invocation { kind : InvokeKind::Interface(count), index },

        New(index) => Allocation(Element { index }),
        Newarray(kind) =>
            JType::try_from(kind)
                .map(|k| Allocation(Array { kind : Explicit(k), dims : 1 }))
                .unwrap_or(Unhandled(insn)),
        Anewarray(_) =>
            Allocation(Array { kind : Explicit(Object), dims : 1 }),
        Multianewarray { index, dimensions : dims } =>
            Allocation(Array { kind : Indirect(index), dims }),

        Arraylength => Length,

        Tableswitch { default, low, high, offsets } => Switch(Table { default, low, high, offsets }),
        Lookupswitch { default, pairs } => Switch(Lookup { default, pairs }),

        Ldc(index) => Constant(Indirect(index.into())),
        LdcW(index) | Ldc2W(index) => Constant(Indirect(index)),

        // We do not intend ever to handle Jsr and Ret
        Jsr(_) | JsrW(_) | Ret(_) | RetWide(_) => Unhandled(insn),

        Athrow
            | Checkcast(_) | Instanceof(_)
            | Monitorenter | Monitorexit
            => Unhandled(insn),
    };

    (addr, op)
}

