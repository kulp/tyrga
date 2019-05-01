use enum_primitive::*;

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

pub fn char_for_primitive_type(t : JType) -> Option<char> {
    use JType::*;
    match t {
        Int    => Some('I'),
        Long   => Some('J'),
        Float  => Some('F'),
        Double => Some('D'),
        Byte   => Some('Z'),
        Char   => Some('C'),
        Short  => Some('S'),
        _ => None,
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
    _1,
    _2,
}

impl From<OperandCount> for u8 {
    fn from(oc : OperandCount) -> Self {
        match oc {
            OperandCount::_1 => 1,
            OperandCount::_2 => 2,
        }
    }
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

enum_from_primitive! {
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ArrayKind {
    Boolean = 4,
    Char    = 5,
    Float   = 6,
    Double  = 7,
    Byte    = 8,
    Short   = 9,
    Int     = 10,
    Long    = 11,
}
}

#[derive(Clone, Debug, PartialEq)]
pub enum SwitchParams {
    Lookup { default : i32, pairs : Vec<(i32, i32)> },
    Table  { default : i32, low : i32, high : i32, offsets : Vec<i32> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    Allocation  { index : u16 },
    Arithmetic  { kind : JType, op : ArithmeticOperation },
    ArrayAlloc  { kind : ArrayKind },
    Branch      { kind : JType, ops : OperandCount, way : Comparison, target : u16 },
    Compare     { kind : JType, nans : Option<NanComparisons> },
    Constant    { kind : JType, value : i32 },
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

// returns any Operation parsed and the number of bytes consumed
pub fn decode_insn(insn : (usize, &Instruction)) -> (usize, Operation) {
    use JType::*;
    use Instruction::*;
    use Operation::*;
    use SwitchParams::*;

    let (addr, insn) = insn;
    let insn = insn.clone(); // TODO obviate clone

    let op = match insn {
        Nop => Noop,

        Aconstnull => Constant { kind : Object, value :  0 },
        Iconstm1   => Constant { kind : Int   , value : -1 },
        Iconst0    => Constant { kind : Int   , value :  0 },
        Iconst1    => Constant { kind : Int   , value :  1 },
        Iconst2    => Constant { kind : Int   , value :  2 },
        Iconst3    => Constant { kind : Int   , value :  3 },
        Iconst4    => Constant { kind : Int   , value :  4 },
        Iconst5    => Constant { kind : Int   , value :  5 },
        Lconst0    => Constant { kind : Long  , value :  0 },
        Lconst1    => Constant { kind : Long  , value :  1 },
        Fconst0    => Constant { kind : Float , value :  0 },
        Fconst1    => Constant { kind : Float , value :  1 },
        Fconst2    => Constant { kind : Float , value :  1 },
        Dconst0    => Constant { kind : Double, value :  0 },
        Dconst1    => Constant { kind : Double, value :  1 },

        Bipush(v) => Constant { kind : Int, value : i32::from(v) },
        Sipush(v) => Constant { kind : Int, value : i32::from(v) },

        Iload(index) => LoadLocal { kind : Int   , index : index.into() },
        Lload(index) => LoadLocal { kind : Long  , index : index.into() },
        Fload(index) => LoadLocal { kind : Float , index : index.into() },
        Dload(index) => LoadLocal { kind : Double, index : index.into() },
        Aload(index) => LoadLocal { kind : Object, index : index.into() },

        IloadWide(index) => LoadLocal { kind : Int   , index },
        LloadWide(index) => LoadLocal { kind : Long  , index },
        FloadWide(index) => LoadLocal { kind : Float , index },
        DloadWide(index) => LoadLocal { kind : Double, index },
        AloadWide(index) => LoadLocal { kind : Object, index },

        Iload0 => LoadLocal { kind : Int   , index : 0 },
        Iload1 => LoadLocal { kind : Int   , index : 1 },
        Iload2 => LoadLocal { kind : Int   , index : 2 },
        Iload3 => LoadLocal { kind : Int   , index : 3 },
        Lload0 => LoadLocal { kind : Long  , index : 0 },
        Lload1 => LoadLocal { kind : Long  , index : 1 },
        Lload2 => LoadLocal { kind : Long  , index : 2 },
        Lload3 => LoadLocal { kind : Long  , index : 3 },
        Fload0 => LoadLocal { kind : Float , index : 0 },
        Fload1 => LoadLocal { kind : Float , index : 1 },
        Fload2 => LoadLocal { kind : Float , index : 2 },
        Fload3 => LoadLocal { kind : Float , index : 3 },
        Dload0 => LoadLocal { kind : Double, index : 0 },
        Dload1 => LoadLocal { kind : Double, index : 1 },
        Dload2 => LoadLocal { kind : Double, index : 2 },
        Dload3 => LoadLocal { kind : Double, index : 3 },
        Aload0 => LoadLocal { kind : Object, index : 0 },
        Aload1 => LoadLocal { kind : Object, index : 1 },
        Aload2 => LoadLocal { kind : Object, index : 2 },
        Aload3 => LoadLocal { kind : Object, index : 3 },

        Iaload => LoadArray(Int),
        Laload => LoadArray(Long),
        Faload => LoadArray(Float),
        Daload => LoadArray(Double),
        Aaload => LoadArray(Object),
        Baload => LoadArray(Byte),
        Caload => LoadArray(Char),
        Saload => LoadArray(Short),

        Istore(index) => StoreLocal { kind : Int   , index : index.into() },
        Lstore(index) => StoreLocal { kind : Long  , index : index.into() },
        Fstore(index) => StoreLocal { kind : Float , index : index.into() },
        Dstore(index) => StoreLocal { kind : Double, index : index.into() },
        Astore(index) => StoreLocal { kind : Object, index : index.into() },

        IstoreWide(index) => StoreLocal { kind : Int   , index },
        LstoreWide(index) => StoreLocal { kind : Long  , index },
        FstoreWide(index) => StoreLocal { kind : Float , index },
        DstoreWide(index) => StoreLocal { kind : Double, index },
        AstoreWide(index) => StoreLocal { kind : Object, index },

        Istore0 => StoreLocal { kind : Int   , index : 0 },
        Istore1 => StoreLocal { kind : Int   , index : 1 },
        Istore2 => StoreLocal { kind : Int   , index : 2 },
        Istore3 => StoreLocal { kind : Int   , index : 3 },
        Lstore0 => StoreLocal { kind : Long  , index : 0 },
        Lstore1 => StoreLocal { kind : Long  , index : 1 },
        Lstore2 => StoreLocal { kind : Long  , index : 2 },
        Lstore3 => StoreLocal { kind : Long  , index : 3 },
        Fstore0 => StoreLocal { kind : Float , index : 0 },
        Fstore1 => StoreLocal { kind : Float , index : 1 },
        Fstore2 => StoreLocal { kind : Float , index : 2 },
        Fstore3 => StoreLocal { kind : Float , index : 3 },
        Dstore0 => StoreLocal { kind : Double, index : 0 },
        Dstore1 => StoreLocal { kind : Double, index : 1 },
        Dstore2 => StoreLocal { kind : Double, index : 2 },
        Dstore3 => StoreLocal { kind : Double, index : 3 },
        Astore0 => StoreLocal { kind : Object, index : 0 },
        Astore1 => StoreLocal { kind : Object, index : 1 },
        Astore2 => StoreLocal { kind : Object, index : 2 },
        Astore3 => StoreLocal { kind : Object, index : 3 },

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
                let kind = match insn {
                    Iadd | Isub | Imul | Idiv | Irem | Ineg
                        | Ishl | Ishr | Iushr | Iand | Ior | Ixor
                        => JType::Int,
                    Ladd | Lsub | Lmul | Ldiv | Lrem | Lneg
                        | Lshl | Lshr | Lushr | Land | Lor | Lxor
                        => JType::Long,
                    Fadd | Fsub | Fmul | Fdiv | Frem | Fneg
                        => JType::Float,
                    Dadd | Dsub | Dmul | Ddiv | Drem | Dneg
                        => JType::Double,

                    _ => unreachable!(),
                };
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
                Arithmetic { kind, op }
            },

        Iinc     { index, value } => Increment { index : index.into(), value : value.into() },
        IincWide { index, value } => Increment { index, value },

        I2l | I2f | I2d | L2i | L2f | L2d | F2i | F2l | F2d | D2i | D2l | D2f | I2b | I2c | I2s
            => {
                let from = match insn {
                          I2l | I2f | I2d | I2b | I2c | I2s => Int,
                    L2i |       L2f | L2d                   => Long,
                    F2i | F2l |       F2d                   => Float,
                    D2i | D2l | D2f                         => Double,

                    _ => unreachable!(),
                };
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

                Conversion { from, to }
            },

        Lcmp    => Compare { kind : Long  , nans : None                          },
        Fcmpl   => Compare { kind : Float , nans : Some(NanComparisons::Less   ) },
        Fcmpg   => Compare { kind : Float , nans : Some(NanComparisons::Greater) },
        Dcmpl   => Compare { kind : Double, nans : Some(NanComparisons::Less   ) },
        Dcmpg   => Compare { kind : Double, nans : Some(NanComparisons::Greater) },

        Ifeq(off) | Ifne(off) | Iflt(off) | Ifge(off) | Ifgt(off) | Ifle(off)
            | IfIcmpeq(off) | IfIcmpne(off) | IfIcmplt(off) | IfIcmpge(off) | IfIcmpgt(off) | IfIcmple(off) | IfAcmpeq(off) | IfAcmpne(off)
            | Ifnull(off) | Ifnonnull(off)
            => {
                let kind = match insn {
                    IfAcmpeq(_) | IfAcmpne(_) | Ifnull(_) | Ifnonnull(_) => Object,
                    _ => Int,
                };
                let target = (addr as isize + off as isize) as u16;
                let way = {
                    match insn {
                        Ifeq(_) | IfIcmpeq(_) | IfAcmpeq(_) | Ifnull(_)    => Comparison::Eq,
                        Ifne(_) | IfIcmpne(_) | IfAcmpne(_) | Ifnonnull(_) => Comparison::Ne,
                        Iflt(_) | IfIcmplt(_)                              => Comparison::Lt,
                        Ifge(_) | IfIcmpge(_)                              => Comparison::Ge,
                        Ifgt(_) | IfIcmpgt(_)                              => Comparison::Gt,
                        Ifle(_) | IfIcmple(_)                              => Comparison::Le,

                        _ => unreachable!(),
                    }
                };
                let ops = {
                    match insn {
                        Ifeq(_) | Ifne(_) | Iflt(_) | Ifge(_) | Ifgt(_) | Ifle(_)
                            | Ifnull(_) | Ifnonnull(_)
                            => OperandCount::_1,
                        IfIcmpeq(_) | IfIcmpne(_) | IfIcmplt(_) | IfIcmpge(_) | IfIcmpgt(_) | IfIcmple(_) | IfAcmpeq(_) | IfAcmpne(_)
                            => OperandCount::_2,
                        _ => unreachable!(),
                    }
                };

                Branch { kind, way, ops, target }
            },

        Goto(off) => Jump { target : (addr as isize + off as isize) as u16 }, // TODO remove casts
        GotoW(off) => Jump { target : (addr as isize + off as isize) as u16 }, // TODO remove casts

        Ireturn | Lreturn | Freturn | Dreturn | Areturn | Return
            => {
                let kind = match insn {
                    Ireturn => Int,
                    Lreturn => Long,
                    Freturn => Float,
                    Dreturn => Double,
                    Areturn => Object,
                    Return  => Void,

                    _ => unreachable!(),
                };
                Yield { kind }
            },

        Getstatic(index) => VarAction { op : VarOp::Get, kind : VarKind::Static, index },
        Putstatic(index) => VarAction { op : VarOp::Put, kind : VarKind::Static, index },
        Getfield(index)  => VarAction { op : VarOp::Get, kind : VarKind::Field , index },
        Putfield(index)  => VarAction { op : VarOp::Put, kind : VarKind::Field , index },

        Invokevirtual(index) => Invocation { kind : InvokeKind::Virtual, index },
        Invokespecial(index) => Invocation { kind : InvokeKind::Special, index },
        Invokestatic(index)  => Invocation { kind : InvokeKind::Static , index },
        Invokedynamic(index) => Invocation { kind : InvokeKind::Dynamic, index },
        Invokeinterface { index, count } => Invocation { kind : InvokeKind::Interface(count), index },

        New(index) => Allocation { index },
        Newarray(kind) => ArrayAlloc { kind : ArrayKind::from_u8(kind).unwrap() },
        Multianewarray { .. } | Anewarray(_) => Unhandled(insn),

        Arraylength => Length,

        Tableswitch { default, low, high, offsets } => Switch(Table { default, low, high, offsets }),
        Lookupswitch { default, pairs } => Switch(Lookup { default, pairs }),

        // We do not intend ever to handle Jsr and Ret
        Jsr(_) | JsrW(_) | Ret(_) | RetWide(_) => Unhandled(insn),

        Athrow
            | Checkcast(_) | Instanceof(_)
            | Monitorenter | Monitorexit
            | Ldc(_) | LdcW(_) | Ldc2W(_)
            => Unhandled(insn),
    };

    (addr, op)
}

