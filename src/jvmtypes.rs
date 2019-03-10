#![allow(dead_code)]

use enum_primitive::*;

enum_from_primitive! {
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
enum JvmOps {
    Nop             = 0x00,
    AconstNull      = 0x01,
    IconstM1        = 0x02,
    Iconst0         = 0x03,
    Iconst1         = 0x04,
    Iconst2         = 0x05,
    Iconst3         = 0x06,
    Iconst4         = 0x07,
    Iconst5         = 0x08,
    Lconst0         = 0x09,
    Lconst1         = 0x0a,
    Fconst0         = 0x0b,
    Fconst1         = 0x0c,
    Fconst2         = 0x0d,
    Dconst0         = 0x0e,
    Dconst1         = 0x0f,
    Bipush          = 0x10,
    Sipush          = 0x11,
    Ldc             = 0x12,
    LdcW            = 0x13,
    Ldc2W           = 0x14,
    Iload           = 0x15,
    Lload           = 0x16,
    Fload           = 0x17,
    Dload           = 0x18,
    Aload           = 0x19,
    Iload0          = 0x1a,
    Iload1          = 0x1b,
    Iload2          = 0x1c,
    Iload3          = 0x1d,
    Lload0          = 0x1e,
    Lload1          = 0x1f,
    Lload2          = 0x20,
    Lload3          = 0x21,
    Fload0          = 0x22,
    Fload1          = 0x23,
    Fload2          = 0x24,
    Fload3          = 0x25,
    Dload0          = 0x26,
    Dload1          = 0x27,
    Dload2          = 0x28,
    Dload3          = 0x29,
    Aload0          = 0x2a,
    Aload1          = 0x2b,
    Aload2          = 0x2c,
    Aload3          = 0x2d,
    Iaload          = 0x2e,
    Laload          = 0x2f,
    Faload          = 0x30,
    Daload          = 0x31,
    Aaload          = 0x32,
    Baload          = 0x33,
    Caload          = 0x34,
    Saload          = 0x35,
    Istore          = 0x36,
    Lstore          = 0x37,
    Fstore          = 0x38,
    Dstore          = 0x39,
    Astore          = 0x3a,
    Istore0         = 0x3b,
    Istore1         = 0x3c,
    Istore2         = 0x3d,
    Istore3         = 0x3e,
    Lstore0         = 0x3f,
    Lstore1         = 0x40,
    Lstore2         = 0x41,
    Lstore3         = 0x42,
    Fstore0         = 0x43,
    Fstore1         = 0x44,
    Fstore2         = 0x45,
    Fstore3         = 0x46,
    Dstore0         = 0x47,
    Dstore1         = 0x48,
    Dstore2         = 0x49,
    Dstore3         = 0x4a,
    Astore0         = 0x4b,
    Astore1         = 0x4c,
    Astore2         = 0x4d,
    Astore3         = 0x4e,
    Iastore         = 0x4f,
    Lastore         = 0x50,
    Fastore         = 0x51,
    Dastore         = 0x52,
    Aastore         = 0x53,
    Bastore         = 0x54,
    Castore         = 0x55,
    Sastore         = 0x56,
    Pop             = 0x57,
    Pop2            = 0x58,
    Dup             = 0x59,
    DupX1           = 0x5a,
    DupX2           = 0x5b,
    Dup2            = 0x5c,
    Dup2X1          = 0x5d,
    Dup2X2          = 0x5e,
    Swap            = 0x5f,
    Iadd            = 0x60,
    Ladd            = 0x61,
    Fadd            = 0x62,
    Dadd            = 0x63,
    Isub            = 0x64,
    Lsub            = 0x65,
    Fsub            = 0x66,
    Dsub            = 0x67,
    Imul            = 0x68,
    Lmul            = 0x69,
    Fmul            = 0x6a,
    Dmul            = 0x6b,
    Idiv            = 0x6c,
    Ldiv            = 0x6d,
    Fdiv            = 0x6e,
    Ddiv            = 0x6f,
    Irem            = 0x70,
    Lrem            = 0x71,
    Frem            = 0x72,
    Drem            = 0x73,
    Ineg            = 0x74,
    Lneg            = 0x75,
    Fneg            = 0x76,
    Dneg            = 0x77,
    Ishl            = 0x78,
    Lshl            = 0x79,
    Ishr            = 0x7a,
    Lshr            = 0x7b,
    Iushr           = 0x7c,
    Lushr           = 0x7d,
    Iand            = 0x7e,
    Land            = 0x7f,
    Ior             = 0x80,
    Lor             = 0x81,
    Ixor            = 0x82,
    Lxor            = 0x83,
    Iinc            = 0x84,
    I2l             = 0x85,
    I2f             = 0x86,
    I2d             = 0x87,
    L2i             = 0x88,
    L2f             = 0x89,
    L2d             = 0x8a,
    F2i             = 0x8b,
    F2l             = 0x8c,
    F2d             = 0x8d,
    D2i             = 0x8e,
    D2l             = 0x8f,
    D2f             = 0x90,
    I2b             = 0x91,
    I2c             = 0x92,
    I2s             = 0x93,
    Lcmp            = 0x94,
    Fcmpl           = 0x95,
    Fcmpg           = 0x96,
    Dcmpl           = 0x97,
    Dcmpg           = 0x98,
    Ifeq            = 0x99,
    Ifne            = 0x9a,
    Iflt            = 0x9b,
    Ifge            = 0x9c,
    Ifgt            = 0x9d,
    Ifle            = 0x9e,
    IfIcmpeq        = 0x9f,
    IfIcmpne        = 0xa0,
    IfIcmplt        = 0xa1,
    IfIcmpge        = 0xa2,
    IfIcmpgt        = 0xa3,
    IfIcmple        = 0xa4,
    IfAcmpeq        = 0xa5,
    IfAcmpne        = 0xa6,
    Goto            = 0xa7,
    Jsr             = 0xa8,
    Ret             = 0xa9,
    Tableswitch     = 0xaa,
    Lookupswitch    = 0xab,
    Ireturn         = 0xac,
    Lreturn         = 0xad,
    Freturn         = 0xae,
    Dreturn         = 0xaf,
    Areturn         = 0xb0,
    Return          = 0xb1,
    Getstatic       = 0xb2,
    Putstatic       = 0xb3,
    Getfield        = 0xb4,
    Putfield        = 0xb5,
    Invokevirtual   = 0xb6,
    Invokespecial   = 0xb7,
    Invokestatic    = 0xb8,
    Invokeinterface = 0xb9,
    Invokedynamic   = 0xba,
    New             = 0xbb,
    Newarray        = 0xbc,
    Anewarray       = 0xbd,
    Arraylength     = 0xbe,
    Athrow          = 0xbf,
    Checkcast       = 0xc0,
    Instanceof      = 0xc1,
    Monitorenter    = 0xc2,
    Monitorexit     = 0xc3,
    Wide            = 0xc4,
    Multianewarray  = 0xc5,
    Ifnull          = 0xc6,
    Ifnonnull       = 0xc7,
    GotoW           = 0xc8,
    JsrW            = 0xc9,
    Breakpoint      = 0xca,
    /* cb - fd reserved */
    Impdep1         = 0xfe,
    Impdep2         = 0xff,
}
}

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
    Inc,
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
    Interface,
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Operation {
    Allocation  { index : u16 },
    Arithmetic  { kind : JType, op : ArithmeticOperation },
    ArrayAlloc  { kind : ArrayKind },
    Branch      { kind : JType, ops : OperandCount, way : Comparison, target : u16 },
    Compare     { kind : JType, nans : Option<NanComparisons> },
    Constant    { kind : JType, value : i32 },
    Conversion  { from : JType, to : JType },
    Invocation  { kind : InvokeKind, index : u16, count : u8 },
    Jump        { target : u16 },
    Leave,      /* i.e. void return */
    Length,     /* i.e. arraylength */
    LoadArray   (JType),
    LoadLocal   { kind : JType, index : u8 },
    Noop,
    StackOp     { size : u8, op : StackOperation },
    StoreArray  (JType),
    StoreLocal  { kind : JType, index : u8 },
    Subtract    { kind : JType },
    VarAction   { op : VarOp, kind : VarKind },
    Yield       { kind : JType }, /* i.e. return */

    Unhandled   (u8),
}

// returns any Operation parsed and the number of bytes consumed
pub fn decode_op(stream : &[u8], addr : u16) -> (Option<Operation>, usize) {
    use ArithmeticOperation::*;
    use InvokeKind::*;
    use JType::*;
    use JvmOps::*;
    use OperandCount::*;
    use Operation::*;

    let signed16   = |x : &[u8]| ((x[0] as i16) << 8) | x[1] as i16;
    let unsigned16 = |x : &[u8]| ((x[0] as u16) << 8) | x[1] as u16;
    let signed32 =
        |x : &[u8]|
            ((x[0] as i32) << 24)
          | ((x[1] as i32) << 16)
          | ((x[2] as i32) <<  8)
          | ((x[3] as i32) <<  0);

    let target = |x : i16| (addr as i32 + x as i32) as u16;

    let byte = stream[0];
    return match JvmOps::from_u8(byte) {
        None => (None, 0),
        Some(code) => {
            let consumed = match code {
                Nop
                    | AconstNull
                    | IconstM1 | Iconst0 | Iconst1 | Iconst2 | Iconst3 | Iconst4 | Iconst5
                    | Lconst0 | Lconst1
                    | Fconst0 | Fconst1 | Fconst2
                    | Dconst0 | Dconst1
                    | Iload0 | Iload1 | Iload2 | Iload3
                    | Lload0 | Lload1 | Lload2 | Lload3
                    | Fload0 | Fload1 | Fload2 | Fload3
                    | Dload0 | Dload1 | Dload2 | Dload3
                    | Aload0 | Aload1 | Aload2 | Aload3
                    | Iaload | Laload | Faload | Daload | Aaload | Baload | Caload | Saload
                    | Istore0 | Istore1 | Istore2 | Istore3
                    | Lstore0 | Lstore1 | Lstore2 | Lstore3
                    | Fstore0 | Fstore1 | Fstore2 | Fstore3
                    | Dstore0 | Dstore1 | Dstore2 | Dstore3
                    | Astore0 | Astore1 | Astore2 | Astore3
                    | Iastore | Lastore | Fastore | Dastore | Aastore | Bastore | Castore | Sastore
                    | Pop | Pop2 | Dup | DupX1 | DupX2 | Dup2 | Dup2X1 | Dup2X2 | Swap
                    | Iadd | Ladd | Fadd | Dadd
                    | Isub | Lsub | Fsub | Dsub
                    | Imul | Lmul | Fmul | Dmul
                    | Idiv | Ldiv | Fdiv | Ddiv
                    | Irem | Lrem | Frem | Drem
                    | Ineg | Lneg | Fneg | Dneg
                    | Ishl  | Lshl
                    | Ishr  | Lshr
                    | Iushr | Lushr
                    | Iand  | Land
                    | Ior   | Lor
                    | Ixor  | Lxor
                    | I2l | I2f | I2d | L2i | L2f | L2d
                    | F2i | F2l | F2d | D2i | D2l | D2f
                    | I2b | I2c | I2s
                    | Lcmp | Fcmpl | Fcmpg | Dcmpl | Dcmpg
                    | Ireturn | Lreturn | Freturn | Dreturn | Areturn | Return
                    | Arraylength
                    | Athrow
                    => 1,
                Bipush
                    | Ldc
                    | Iload | Lload | Fload | Dload | Aload
                    | Istore | Lstore | Fstore | Dstore | Astore
                    | Ret
                    | Newarray
                    => 2,
                Sipush
                    | LdcW | Ldc2W
                    | Iinc
                    | Ifeq | Ifne | Iflt | Ifge | Ifgt | Ifle
                    | IfIcmpeq | IfIcmpne | IfIcmplt | IfIcmpge | IfIcmpgt | IfIcmple
                    | IfAcmpeq | IfAcmpne
                    | Goto | Jsr
                    | Getstatic | Putstatic | Getfield | Putfield
                    | Invokespecial | Invokestatic | Invokevirtual
                    | New | Anewarray
                    | Checkcast
                    | Instanceof
                    | Monitorenter | Monitorexit
                    | Ifnull | Ifnonnull
                    => 3,
                Multianewarray
                    => 4,
                Invokeinterface | Invokedynamic
                    | JsrW | GotoW
                    => 5,
                Tableswitch | Lookupswitch
                    => {
                        let padding = ((4 - ((addr + 1) & 3)) & 3) as usize;
                        let need = 1 + padding + 4 * 3;
                        if stream.len() < need {
                            panic!("ran out of bytes in stream during {:?}", code);
                        }
                        let def = &stream[1 + padding..];
                        let count = match code {
                            Tableswitch => {
                                let low  = &def[4..];
                                let high = &low[4..];
                                let low  = signed32(low);
                                let high = signed32(high);
                                (high - low + 1) as usize
                            },
                            Lookupswitch => {
                                let npairs = &def[4..];
                                signed32(npairs) as usize
                            },
                            _ => unreachable!(),
                        };
                        need + count * 4
                    },
                Wide
                    => match JvmOps::from_u8(stream[1]).expect("bad opcode") {
                        Ret
                            | Iload  | Fload  | Aload  | Lload  | Dload
                            | Istore | Fstore | Astore | Lstore | Dstore
                            => 4,
                        Iinc
                            => 6,
                        _
                            => panic!("unexpected opcode in {:?}", code),
                    },

                _
                    => 0,
            };

            if stream.len() < consumed {
                panic!("expected {} bytes, received only {}", consumed, stream.len());
            }

            let opt = match code {
                Nop
                    => Some(Noop),
                AconstNull
                    => Some(Constant { kind : Object, value : 0 }),
                IconstM1 | Iconst0 | Iconst1 | Iconst2 | Iconst3 | Iconst4 | Iconst5
                    => Some(Constant { kind : Int, value : (byte as i8 - Iconst0 as i8) as i32 }),
                Lconst0 | Lconst1
                    => Some(Constant { kind : Long, value : (byte - Lconst0 as u8) as i32 }),
                Fconst0 | Fconst1 | Fconst2
                    => Some(Constant { kind : Float, value : (byte - Fconst0 as u8) as i32 }),
                Dconst0 | Dconst1
                    => Some(Constant { kind : Double, value : (byte - Dconst0 as u8) as i32 }),
                Bipush
                    => Some(Constant { kind : Int, value : stream[1] as i32 }),
                Sipush
                    => Some(Constant { kind : Int, value : signed16(&stream[1..]) as i32 }),
                Ldc | LdcW | Ldc2W
                    => Some(Unhandled(byte)),

                Iload => Some(LoadLocal { kind : Int   , index : stream[1] }),
                Lload => Some(LoadLocal { kind : Long  , index : stream[1] }),
                Fload => Some(LoadLocal { kind : Float , index : stream[1] }),
                Dload => Some(LoadLocal { kind : Double, index : stream[1] }),
                Aload => Some(LoadLocal { kind : Object, index : stream[1] }),

                Iload0 | Iload1 | Iload2 | Iload3
                    => Some(LoadLocal { kind : Int, index : byte - Iload0 as u8 }),
                Lload0 | Lload1 | Lload2 | Lload3
                    => Some(LoadLocal { kind : Long, index : byte - Lload0 as u8 }),
                Fload0 | Fload1 | Fload2 | Fload3
                    => Some(LoadLocal { kind : Float, index : byte - Fload0 as u8 }),
                Dload0 | Dload1 | Dload2 | Dload3
                    => Some(LoadLocal { kind : Double, index : byte - Dload0 as u8 }),
                Aload0 | Aload1 | Aload2 | Aload3
                    => Some(LoadLocal { kind : Object, index : byte - Aload0 as u8 }),

                Iaload => Some(LoadArray(Int)),
                Laload => Some(LoadArray(Long)),
                Faload => Some(LoadArray(Float)),
                Daload => Some(LoadArray(Double)),
                Aaload => Some(LoadArray(Object)),
                Baload => Some(LoadArray(Byte)),
                Caload => Some(LoadArray(Char)),
                Saload => Some(LoadArray(Short)),

                Istore => Some(StoreLocal { kind : Int   , index : stream[1] }),
                Lstore => Some(StoreLocal { kind : Long  , index : stream[1] }),
                Fstore => Some(StoreLocal { kind : Float , index : stream[1] }),
                Dstore => Some(StoreLocal { kind : Double, index : stream[1] }),
                Astore => Some(StoreLocal { kind : Object, index : stream[1] }),

                Istore0 | Istore1 | Istore2 | Istore3
                    => Some(StoreLocal { kind : Int, index : byte - Istore0 as u8 }),
                Lstore0 | Lstore1 | Lstore2 | Lstore3
                    => Some(StoreLocal { kind : Long, index : byte - Lstore0 as u8 }),
                Fstore0 | Fstore1 | Fstore2 | Fstore3
                    => Some(StoreLocal { kind : Float, index : byte - Fstore0 as u8 }),
                Dstore0 | Dstore1 | Dstore2 | Dstore3
                    => Some(StoreLocal { kind : Double, index : byte - Dstore0 as u8 }),
                Astore0 | Astore1 | Astore2 | Astore3
                    => Some(StoreLocal { kind : Object, index : byte - Astore0 as u8 }),

                Iastore => Some(StoreArray(Int)),
                Lastore => Some(StoreArray(Long)),
                Fastore => Some(StoreArray(Float)),
                Dastore => Some(StoreArray(Double)),
                Aastore => Some(StoreArray(Object)),
                Bastore => Some(StoreArray(Byte)),
                Castore => Some(StoreArray(Char)),
                Sastore => Some(StoreArray(Short)),

                Pop     => Some(StackOp { op : StackOperation::Pop  , size : 1 }),
                Pop2    => Some(StackOp { op : StackOperation::Pop  , size : 2 }),
                Dup     => Some(StackOp { op : StackOperation::Dup  , size : 1 }),
                DupX1   => Some(StackOp { op : StackOperation::DupX1, size : 1 }),
                DupX2   => Some(StackOp { op : StackOperation::DupX2, size : 1 }),
                Dup2    => Some(StackOp { op : StackOperation::Dup  , size : 2 }),
                Dup2X1  => Some(StackOp { op : StackOperation::DupX1, size : 2 }),
                Dup2X2  => Some(StackOp { op : StackOperation::DupX2, size : 2 }),
                Swap    => Some(StackOp { op : StackOperation::Swap , size : 2 }),

                Iadd    => Some(Arithmetic { kind : Int   , op : Add    }),
                Ladd    => Some(Arithmetic { kind : Long  , op : Add    }),
                Fadd    => Some(Arithmetic { kind : Float , op : Add    }),
                Dadd    => Some(Arithmetic { kind : Double, op : Add    }),
                Isub    => Some(Arithmetic { kind : Int   , op : Sub    }),
                Lsub    => Some(Arithmetic { kind : Long  , op : Sub    }),
                Fsub    => Some(Arithmetic { kind : Float , op : Sub    }),
                Dsub    => Some(Arithmetic { kind : Double, op : Sub    }),
                Imul    => Some(Arithmetic { kind : Int   , op : Mul    }),
                Lmul    => Some(Arithmetic { kind : Long  , op : Mul    }),
                Fmul    => Some(Arithmetic { kind : Float , op : Mul    }),
                Dmul    => Some(Arithmetic { kind : Double, op : Mul    }),
                Idiv    => Some(Arithmetic { kind : Int   , op : Div    }),
                Ldiv    => Some(Arithmetic { kind : Long  , op : Div    }),
                Fdiv    => Some(Arithmetic { kind : Float , op : Div    }),
                Ddiv    => Some(Arithmetic { kind : Double, op : Div    }),
                Irem    => Some(Arithmetic { kind : Int   , op : Rem    }),
                Lrem    => Some(Arithmetic { kind : Long  , op : Rem    }),
                Frem    => Some(Arithmetic { kind : Float , op : Rem    }),
                Drem    => Some(Arithmetic { kind : Double, op : Rem    }),
                Ineg    => Some(Arithmetic { kind : Int   , op : Neg    }),
                Lneg    => Some(Arithmetic { kind : Long  , op : Neg    }),
                Fneg    => Some(Arithmetic { kind : Float , op : Neg    }),
                Dneg    => Some(Arithmetic { kind : Double, op : Neg    }),
                Ishl    => Some(Arithmetic { kind : Int   , op : Shl    }),
                Lshl    => Some(Arithmetic { kind : Long  , op : Shl    }),
                Ishr    => Some(Arithmetic { kind : Int   , op : Shr    }),
                Lshr    => Some(Arithmetic { kind : Long  , op : Shr    }),
                Iushr   => Some(Arithmetic { kind : Int   , op : Ushr   }),
                Lushr   => Some(Arithmetic { kind : Long  , op : Ushr   }),
                Iand    => Some(Arithmetic { kind : Int   , op : And    }),
                Land    => Some(Arithmetic { kind : Long  , op : And    }),
                Ior     => Some(Arithmetic { kind : Int   , op : Or     }),
                Lor     => Some(Arithmetic { kind : Long  , op : Or     }),
                Ixor    => Some(Arithmetic { kind : Int   , op : Xor    }),
                Lxor    => Some(Arithmetic { kind : Long  , op : Xor    }),
                Iinc    => Some(Arithmetic { kind : Int   , op : Inc    }),
                I2l     => Some(Conversion { from : Int   , to : Long   }),
                I2f     => Some(Conversion { from : Int   , to : Float  }),
                I2d     => Some(Conversion { from : Int   , to : Double }),
                L2i     => Some(Conversion { from : Long  , to : Int    }),
                L2f     => Some(Conversion { from : Long  , to : Float  }),
                L2d     => Some(Conversion { from : Long  , to : Double }),
                F2i     => Some(Conversion { from : Float , to : Int    }),
                F2l     => Some(Conversion { from : Float , to : Long   }),
                F2d     => Some(Conversion { from : Float , to : Double }),
                D2i     => Some(Conversion { from : Double, to : Int    }),
                D2l     => Some(Conversion { from : Double, to : Long   }),
                D2f     => Some(Conversion { from : Double, to : Float  }),
                I2b     => Some(Conversion { from : Int   , to : Byte   }),
                I2c     => Some(Conversion { from : Int   , to : Char   }),
                I2s     => Some(Conversion { from : Int   , to : Short  }),

                Lcmp    => Some(Compare { kind : Long  , nans : None                          }),
                Fcmpl   => Some(Compare { kind : Float , nans : Some(NanComparisons::Less   ) }),
                Fcmpg   => Some(Compare { kind : Float , nans : Some(NanComparisons::Greater) }),
                Dcmpl   => Some(Compare { kind : Double, nans : Some(NanComparisons::Less   ) }),
                Dcmpg   => Some(Compare { kind : Double, nans : Some(NanComparisons::Greater) }),

                Ifeq    => Some(Branch { kind : Int, way : Comparison::Eq, ops : _1, target : target(signed16(&stream[1..])) }),
                Ifne    => Some(Branch { kind : Int, way : Comparison::Ne, ops : _1, target : target(signed16(&stream[1..])) }),
                Iflt    => Some(Branch { kind : Int, way : Comparison::Lt, ops : _1, target : target(signed16(&stream[1..])) }),
                Ifge    => Some(Branch { kind : Int, way : Comparison::Ge, ops : _1, target : target(signed16(&stream[1..])) }),
                Ifgt    => Some(Branch { kind : Int, way : Comparison::Gt, ops : _1, target : target(signed16(&stream[1..])) }),
                Ifle    => Some(Branch { kind : Int, way : Comparison::Le, ops : _1, target : target(signed16(&stream[1..])) }),

                IfIcmpeq    => Some(Branch { kind : Int   , way : Comparison::Eq, ops : _2, target : target(signed16(&stream[1..])) }),
                IfIcmpne    => Some(Branch { kind : Int   , way : Comparison::Ne, ops : _2, target : target(signed16(&stream[1..])) }),
                IfIcmplt    => Some(Branch { kind : Int   , way : Comparison::Lt, ops : _2, target : target(signed16(&stream[1..])) }),
                IfIcmpge    => Some(Branch { kind : Int   , way : Comparison::Ge, ops : _2, target : target(signed16(&stream[1..])) }),
                IfIcmpgt    => Some(Branch { kind : Int   , way : Comparison::Gt, ops : _2, target : target(signed16(&stream[1..])) }),
                IfIcmple    => Some(Branch { kind : Int   , way : Comparison::Le, ops : _2, target : target(signed16(&stream[1..])) }),
                IfAcmpeq    => Some(Branch { kind : Object, way : Comparison::Eq, ops : _2, target : target(signed16(&stream[1..])) }),
                IfAcmpne    => Some(Branch { kind : Object, way : Comparison::Ne, ops : _2, target : target(signed16(&stream[1..])) }),

                Goto    => Some(Jump { target : target(signed16(&stream[1..])) }),
                GotoW   => Some(Unhandled(byte)),
                Jsr     => Some(Unhandled(byte)),
                JsrW    => Some(Unhandled(byte)),
                Ret     => Some(Unhandled(byte)),

                Tableswitch     => Some(Unhandled(byte)),
                Lookupswitch    => Some(Unhandled(byte)),

                Ireturn => Some(Yield { kind : Int    }),
                Lreturn => Some(Yield { kind : Long   }),
                Freturn => Some(Yield { kind : Float  }),
                Dreturn => Some(Yield { kind : Double }),
                Areturn => Some(Yield { kind : Object }),
                Return  => Some(Yield { kind : Void   }),

                Getstatic   => Some(VarAction { op : VarOp::Get, kind : VarKind::Static }),
                Putstatic   => Some(VarAction { op : VarOp::Put, kind : VarKind::Static }),
                Getfield    => Some(VarAction { op : VarOp::Get, kind : VarKind::Field  }),
                Putfield    => Some(VarAction { op : VarOp::Put, kind : VarKind::Field  }),

                Invokevirtual   => Some(Invocation { kind : Virtual  , index : unsigned16(&stream[1..]), count : 0         }),
                Invokespecial   => Some(Invocation { kind : Special  , index : unsigned16(&stream[1..]), count : 0         }),
                Invokestatic    => Some(Invocation { kind : Static   , index : unsigned16(&stream[1..]), count : 0         }),
                Invokeinterface => Some(Invocation { kind : Interface, index : unsigned16(&stream[1..]), count : stream[3] }),
                Invokedynamic   => Some(Invocation { kind : Dynamic  , index : unsigned16(&stream[1..]), count : 0         }),

                New => Some(Allocation { index : unsigned16(&stream[1..]) }),
                Newarray => Some(ArrayAlloc { kind : ArrayKind::from_u8(stream[1]).unwrap() }),
                Anewarray => Some(Unhandled(byte)),

                Arraylength => Some(Length),

                Athrow
                    | Checkcast
                    | Instanceof
                    | Monitorenter
                    | Monitorexit
                    | Wide
                    | Multianewarray
                    => Some(Unhandled(byte)),

                Ifnull      => Some(Branch { kind : Object, way : Comparison::Eq, ops : _1, target : target(signed16(&stream[1..])) }),
                Ifnonnull   => Some(Branch { kind : Object, way : Comparison::Ne, ops : _1, target : target(signed16(&stream[1..])) }),

                Breakpoint | Impdep1 | Impdep2
                    => panic!("invalid opcode {:?} found", code),
            };

            (opt, consumed)
        }
    };
}

#[test]
fn test_get_op() {
    use JType::*;
    use JvmOps::*;

    assert_eq!((Some(Operation::Constant { kind : Int, value : 3 }), 1),
                decode_op(&vec![ Iconst3 as u8 ], 0));

    for b in 0..=255 {
        if let Some(name) = JvmOps::from_u8(b){
            let mut arr = vec![ b ];
            for _ in 1..20 { arr.push(0) }
            match name {
                Newarray => arr[1] = ArrayKind::Boolean as u8,
                Wide => arr[1] = Iload as u8,
                Breakpoint | Impdep1 | Impdep2 => continue,
                _ => {},
            };
            let v = decode_op(&arr, 0);
            match v {
                (Some(Operation::Unhandled(_)), 0) => {},
                (Some(_), x) => assert!(x != 0),
                _ => panic!("unhandled"),
            };
        }
    }
}

#[test]
#[should_panic(expected = "expected 20")]
fn test_tableswitch() {
    let mut arr = vec![ JvmOps::Tableswitch as u8 ];
    for _ in 1..18 { arr.push(0) }
    let v = decode_op(&arr, 0);
    match v {
        (Some(Operation::Unhandled(_)), 0) => {},
            (Some(_), x) => assert!(x != 0),
            _ => panic!("unhandled"),
    };
}

#[cfg(test)]
fn test_invalids(op : JvmOps) {
    match decode_op(&vec![ op as u8 ], 0) {
        (Some(Operation::Unhandled(_)), 0) => {},
            (Some(_), x) => assert!(x != 0),
            _ => panic!("unhandled"),
    };
}

#[test]
#[should_panic(expected = "invalid opcode Breakpoint found")]
fn test_breakpoint() { test_invalids(JvmOps::Breakpoint) }

#[test]
#[should_panic(expected = "invalid opcode Impdep1 found")]
fn test_impdep1() { test_invalids(JvmOps::Impdep1) }

#[test]
#[should_panic(expected = "invalid opcode Impdep2 found")]
fn test_impdep2() { test_invalids(JvmOps::Impdep2) }

