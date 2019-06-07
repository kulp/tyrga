#![allow(unused_macros)]

use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::fmt;
use std::marker::PhantomData;

use crate::exprtree;

macro_rules! tenyr_op {
    ( |   ) => { BitwiseOr       };     ( |~  ) => { BitwiseOrn      };
    ( &   ) => { BitwiseAnd      };     ( &~  ) => { BitwiseAndn     };
    ( ^   ) => { BitwiseXor      };     ( ^^  ) => { Pack            };
    ( >>  ) => { ShiftRightArith };     ( >>> ) => { ShiftRightLogic };

    ( +   ) => { Add             };     ( -   ) => { Subtract        };
    ( *   ) => { Multiply        };     ( <<  ) => { ShiftLeft       };
    ( ==  ) => { CompareEq       };     ( @   ) => { TestBit         };
    ( <   ) => { CompareLt       };     ( >=  ) => { CompareGe       };
}

pub const NOOP_TYPE0_GEN : InsnGeneral = InsnGeneral {
    y : Register::A,
    imm : Immediate12::ZERO,
    op : Opcode::BitwiseOr,
};
pub const NOOP_TYPE0 : Instruction = Instruction {
    kind : InstructionType::Type0(NOOP_TYPE0_GEN),
    z : Register::A,
    x : Register::A,
    dd : MemoryOpType::NoLoad,
};

// Some tenyr ops are more than one token, so require special treatment
macro_rules! tenyr_get_op {
    ( $callback:ident                       ) => { { use $crate::tenyr::Opcode::*; let op = tenyr_op!( | ); $callback!(op            ) } };

    ( $callback:ident |~     $( $rest:tt )+ ) => { { use $crate::tenyr::Opcode::*; let op = tenyr_op!(|~ ); $callback!(op $( $rest )+) } };
    ( $callback:ident &~     $( $rest:tt )+ ) => { { use $crate::tenyr::Opcode::*; let op = tenyr_op!(&~ ); $callback!(op $( $rest )+) } };
    ( $callback:ident ^^     $( $rest:tt )+ ) => { { use $crate::tenyr::Opcode::*; let op = tenyr_op!(^^ ); $callback!(op $( $rest )+) } };
    ( $callback:ident >>>    $( $rest:tt )+ ) => { { use $crate::tenyr::Opcode::*; let op = tenyr_op!(>>>); $callback!(op $( $rest )+) } };
    ( $callback:ident $op:tt $( $rest:tt )+ ) => { { use $crate::tenyr::Opcode::*; let op = tenyr_op!($op); $callback!(op $( $rest )+) } };
}

pub type InsnResult = Result<Instruction, Box<dyn std::error::Error>>;

macro_rules! tenyr_imm {
    ( $imm:expr ) => { {
        use std::convert::TryInto;
        $imm.try_into().map_err::<Box<dyn std::error::Error>,_>(Into::into)?
    } };
}

macro_rules! tenyr_type013 {
    ( $opname:ident ( $imm:expr ) $( + $y:ident )? ) => {
        Ok($crate::tenyr::Instruction {
            kind : Type1(InsnGeneral {
                $( y : $y, )?
                op : $opname,
                imm : tenyr_imm!($imm),
                ..$crate::tenyr::NOOP_TYPE0_GEN
            }),
            ..$crate::tenyr::NOOP_TYPE0
        }) as $crate::tenyr::InsnResult
    };
    ( $opname:ident $imm:literal $( + $y:ident )? ) => {
        Ok($crate::tenyr::Instruction {
            kind : Type1(InsnGeneral {
                $( y : $y, )?
                op : $opname,
                imm : tenyr_imm!($imm),
                ..$crate::tenyr::NOOP_TYPE0_GEN
            }),
            ..$crate::tenyr::NOOP_TYPE0
        }) as $crate::tenyr::InsnResult
    };
    ( $opname:ident $( $y:ident $( + ( $imm:expr ) )? )? ) => {
        Ok($crate::tenyr::Instruction {
            kind : Type0(InsnGeneral {
                $( y : $y, $( imm : tenyr_imm!($imm), )? )?
                op : $opname,
                ..$crate::tenyr::NOOP_TYPE0_GEN
            }),
            ..$crate::tenyr::NOOP_TYPE0
        }) as $crate::tenyr::InsnResult
    };
    ( $opname:ident $( $y:ident $( + $imm:literal )? )? ) => {
        Ok($crate::tenyr::Instruction {
            kind : Type0(InsnGeneral {
                $( y : $y, $( imm : tenyr_imm!($imm), )? )?
                op : $opname,
                ..$crate::tenyr::NOOP_TYPE0_GEN
            }),
            ..$crate::tenyr::NOOP_TYPE0
        }) as $crate::tenyr::InsnResult
    };
    ( $opname:ident $( $y:ident $( - ( $imm:expr ) )? )? ) => {
        Ok($crate::tenyr::Instruction {
            kind : Type0(InsnGeneral {
                $( y : $y, $( imm : tenyr_imm!(-($imm)), )? )?
                op : $opname,
                ..$crate::tenyr::NOOP_TYPE0_GEN
            }),
            ..$crate::tenyr::NOOP_TYPE0
        }) as $crate::tenyr::InsnResult
    };
    ( $opname:ident $( $y:ident $( - $imm:literal )? )? ) => {
        Ok($crate::tenyr::Instruction {
            kind : Type0(InsnGeneral {
                $( y : $y, $( imm : tenyr_imm!(-($imm)), )? )?
                op : $opname,
                ..$crate::tenyr::NOOP_TYPE0_GEN
            }),
            ..$crate::tenyr::NOOP_TYPE0
        }) as $crate::tenyr::InsnResult
    };
}

macro_rules! tenyr_type2 {
    ( $opname:ident $x:ident $( + $y:ident )? ) => {
        Ok($crate::tenyr::Instruction {
            x : $x,
            kind : Type2(InsnGeneral {
                $( y : $y, )?
                op : $opname,
                ..$crate::tenyr::NOOP_TYPE0_GEN
            }),
            ..$crate::tenyr::NOOP_TYPE0
        }) as $crate::tenyr::InsnResult
    };
}

macro_rules! tenyr_rhs {
    ( $( $x:ident + )? ( $imm:expr ) ) => {
        {
            use $crate::tenyr::*;
            let kind = Type3(tenyr_imm!($imm));
            let base = Instruction { kind, ..$crate::tenyr::NOOP_TYPE0 };
            Ok(Instruction { $( x : $x, )? ..base }) as $crate::tenyr::InsnResult
        }
    };
    ( $( $x:ident + )? $imm:literal ) => {
        {
            use $crate::tenyr::*;
            let kind = Type3(tenyr_imm!($imm));
            let base = Instruction { kind, ..$crate::tenyr::NOOP_TYPE0 };
            Ok(Instruction { $( x : $x, )? ..base }) as $crate::tenyr::InsnResult
        }
    };
    ( $( $x:ident - )? ( $imm:expr ) ) => {
        {
            use $crate::tenyr::*;
            let kind = Type3(tenyr_imm!(-($imm)));
            let base = Instruction { kind, ..$crate::tenyr::NOOP_TYPE0 };
            Ok(Instruction { $( x : $x, )? ..base }) as $crate::tenyr::InsnResult
        }
    };
    ( $( $x:ident - )? $imm:literal ) => {
        {
            use $crate::tenyr::*;
            let kind = Type3(tenyr_imm!(-($imm)));
            let base = Instruction { kind, ..$crate::tenyr::NOOP_TYPE0 };
            Ok(Instruction { $( x : $x, )? ..base }) as $crate::tenyr::InsnResult
        }
    };
    ( $x:ident $( $rest:tt )* ) => {
        {
            use $crate::tenyr::*;
            #[allow(clippy::needless_update)]
            let base = tenyr_get_op!(tenyr_type013 $( $rest )*);
            Ok(Instruction { z : Register::A, x : $x, dd : MemoryOpType::NoLoad, ..base? }) as $crate::tenyr::InsnResult
        }
    };
    ( ( $imm:expr ) $( $rest:tt )+ ) => {
        {
            use $crate::tenyr::*;
            let base = tenyr_get_op!(tenyr_type2 $( $rest )*)?;
            if let Type2(gen) = base.kind {
                let kind = Type2(InsnGeneral { imm : tenyr_imm!($imm), ..gen });
                Ok(Instruction { kind, ..base }) as $crate::tenyr::InsnResult
            } else {
                Err("internal error - did not get expected Type2".into())
            }
        }
    };
    ( $imm:literal $( $rest:tt )+ ) => {
        {
            use $crate::tenyr::*;
            let base = tenyr_get_op!(tenyr_type2 $( $rest )*)?;
            if let Type2(gen) = base.kind {
                let kind = Type2(InsnGeneral { imm : tenyr_imm!($imm), ..gen });
                Ok(Instruction { kind, ..base }) as $crate::tenyr::InsnResult
            } else {
                Err("internal error - did not get expected Type2".into())
            }
        }
    };
}

#[macro_export]
macro_rules! tenyr_insn {
    (   $z:ident   <- [ $( $rhs:tt )+ ] ) => { Ok(Instruction { z : $z, dd : $crate::tenyr::MemoryOpType::LoadRight , ..tenyr_rhs!( $( $rhs )+ )? }) as $crate::tenyr::InsnResult };
    ( [ $z:ident ] <-   $( $rhs:tt )+   ) => { Ok(Instruction { z : $z, dd : $crate::tenyr::MemoryOpType::StoreLeft , ..tenyr_rhs!( $( $rhs )+ )? }) as $crate::tenyr::InsnResult };
    (   $z:ident   -> [ $( $rhs:tt )+ ] ) => { Ok(Instruction { z : $z, dd : $crate::tenyr::MemoryOpType::StoreRight, ..tenyr_rhs!( $( $rhs )+ )? }) as $crate::tenyr::InsnResult };
    (   $z:ident   <-   $( $rhs:tt )+   ) => { Ok(Instruction { z : $z,                                               ..tenyr_rhs!( $( $rhs )+ )? }) as $crate::tenyr::InsnResult };
}

#[test]
fn test_macro_insn() -> Result<(), Box<dyn std::error::Error>> {
    use InstructionType::*;
    use MemoryOpType::*;
    use Opcode::*;
    use Register::*;

    use std::convert::TryInto;

    let three = 3;

    assert_eq!(tenyr_insn!( B  <-  C  |~ D + 3      )?, Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn      , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C >>> D + 3      )?, Instruction { kind : Type0(InsnGeneral { y : D, op : ShiftRightLogic , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  +  D + 3      )?, Instruction { kind : Type0(InsnGeneral { y : D, op : Add             , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  *  D + 3      )?, Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply        , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  *  D - 3      )?, Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply        , imm : (-3i8).into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C                )?, Instruction { kind : Type0(InsnGeneral { y : A, op : BitwiseOr       , imm :   0u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  +  D          )?, Instruction { kind : Type0(InsnGeneral { y : D, op : Add             , imm :   0u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  |~ D          )?, Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn      , imm :   0u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  ^^ 3          )?, Instruction { kind : Type1(InsnGeneral { y : A, op : Pack            , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  3  *  C          )?, Instruction { kind : Type2(InsnGeneral { y : A, op : Multiply        , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  3  *  C + D      )?, Instruction { kind : Type2(InsnGeneral { y : D, op : Multiply        , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!([B] <-  3  ^^ C + D      )?, Instruction { kind : Type2(InsnGeneral { y : D, op : Pack            , imm :   3u8 .into() } ) , z : B, x : C, dd : StoreLeft  });
    assert_eq!(tenyr_insn!( B  -> [3  &~ C + D]     )?, Instruction { kind : Type2(InsnGeneral { y : D, op : BitwiseAndn     , imm :   3u8 .into() } ) , z : B, x : C, dd : StoreRight });
    assert_eq!(tenyr_insn!( B  <- [3  @  C + D]     )?, Instruction { kind : Type2(InsnGeneral { y : D, op : TestBit         , imm :   3u8 .into() } ) , z : B, x : C, dd : LoadRight  });
    assert_eq!(tenyr_insn!( B  <-  C  |~ D + (three))?, Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn      , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C >>> D + (three))?, Instruction { kind : Type0(InsnGeneral { y : D, op : ShiftRightLogic , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  +  D + (three))?, Instruction { kind : Type0(InsnGeneral { y : D, op : Add             , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  *  D + (three))?, Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply        , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  C  ^^ (three)    )?, Instruction { kind : Type1(InsnGeneral { y : A, op : Pack            , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  (three) * C      )?, Instruction { kind : Type2(InsnGeneral { y : A, op : Multiply        , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!( B  <-  (three) * C + D  )?, Instruction { kind : Type2(InsnGeneral { y : D, op : Multiply        , imm :   3u8 .into() } ) , z : B, x : C, dd : NoLoad     });
    assert_eq!(tenyr_insn!([B] <-  (three) ^^ C + D )?, Instruction { kind : Type2(InsnGeneral { y : D, op : Pack            , imm :   3u8 .into() } ) , z : B, x : C, dd : StoreLeft  });
    assert_eq!(tenyr_insn!( B  -> [(three) &~ C + D])?, Instruction { kind : Type2(InsnGeneral { y : D, op : BitwiseAndn     , imm :   3u8 .into() } ) , z : B, x : C, dd : StoreRight });
    assert_eq!(tenyr_insn!( B  <- [(three) @  C + D])?, Instruction { kind : Type2(InsnGeneral { y : D, op : TestBit         , imm :   3u8 .into() } ) , z : B, x : C, dd : LoadRight  });
    assert_eq!(tenyr_insn!( B  <-  C  +  0x12345    )?, Instruction { kind : Type3(  0x12345_i32 .try_into()?), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  -  0x12345    )?, Instruction { kind : Type3((-0x12345_i32).try_into()?), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-        0x12345    )?, Instruction { kind : Type3(  0x12345_i32 .try_into()?), z : B, x : A, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  +  (three)    )?, Instruction { kind : Type3(  3u8 .into())             , z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  -  (three)    )?, Instruction { kind : Type3((-3i8).into())             , z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-        (three)    )?, Instruction { kind : Type3(  3u8 .into())             , z : B, x : A, dd : NoLoad });

    Ok(())
}

#[macro_export]
macro_rules! tenyr_insn_list {
    () => { vec![] };
    ( $lhs:tt <- $a:tt $op:tt$op2:tt $b:tt $( + $c:tt )? ; $( $tok:tt )* ) => { std::iter::once(tenyr_insn!($lhs <- $a $op$op2 $b $( + $c )?  )?).chain(tenyr_insn_list!($( $tok )*)) };
    ( $lhs:tt <- $a:tt $op:tt        $b:tt $( + $c:tt )? ; $( $tok:tt )* ) => { std::iter::once(tenyr_insn!($lhs <- $a $op     $b $( + $c )?  )?).chain(tenyr_insn_list!($( $tok )*)) };
    ( $lhs:tt <- $rhs:tt                                 ; $( $tok:tt )* ) => { std::iter::once(tenyr_insn!($lhs <- $rhs                      )?).chain(tenyr_insn_list!($( $tok )*)) };
    ( $lhs:tt -> $rhs:tt                                 ; $( $tok:tt )* ) => { std::iter::once(tenyr_insn!($lhs -> $rhs                      )?).chain(tenyr_insn_list!($( $tok )*)) };
}

#[test]
fn test_macro_insn_list() -> Result<(), Box<dyn std::error::Error>> {
    use InstructionType::*;
    use MemoryOpType::*;
    use Opcode::*;
    use Register::*;

    let three = 3;

    let from = tenyr_insn_list! {
         B  <-  C  |~ D + 3      ;
         B  <-  C >>> D + 3      ;
         B  <-  C  +  D + 3      ;
         B  <-  C  *  D + 3      ;
         B  <-  C                ;
         B  <-  C  +  D          ;
         B  <-  C  |~ D          ;
         B  <-  C  ^^ 3          ;
         B  <-  3  *  C          ;
         B  <-  3  *  C + D      ;
        [B] <-  3  ^^ C + D      ;
         B  -> [3  &~ C + D]     ;
         B  <- [3  @  C + D]     ;
         B  <-  C  |~ D + (three);
         B  <-  C >>> D + (three);
         B  <-  C  +  D + (three);
         B  <-  C  *  D + (three);
         B  <-  C  ^^ (three)    ;
         B  <-  (three) * C      ;
         B  <-  (three) * C + D  ;
        [B] <-  (three) ^^ C + D ;
         B  -> [(three) &~ C + D];
         B  <- [(three) @  C + D];
         B  <-  C  +  0x12345    ;
         B  <-        0x12345    ;
         B  <-  C  +  (three)    ;
         B  <-        (three)    ;
    };

    let from : Vec<_> = from.collect();

    use std::convert::TryInto;

    let to = vec![
        Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn      , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : D, op : ShiftRightLogic , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Add             , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply        , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : A, op : BitwiseOr       , imm : 0u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Add             , imm : 0u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn      , imm : 0u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type1(InsnGeneral { y : A, op : Pack            , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type2(InsnGeneral { y : A, op : Multiply        , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type2(InsnGeneral { y : D, op : Multiply        , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type2(InsnGeneral { y : D, op : Pack            , imm : 3u8.into() } ) , z : B, x : C, dd : StoreLeft  },
        Instruction { kind : Type2(InsnGeneral { y : D, op : BitwiseAndn     , imm : 3u8.into() } ) , z : B, x : C, dd : StoreRight },
        Instruction { kind : Type2(InsnGeneral { y : D, op : TestBit         , imm : 3u8.into() } ) , z : B, x : C, dd : LoadRight  },
        Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn      , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : D, op : ShiftRightLogic , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Add             , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply        , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type1(InsnGeneral { y : A, op : Pack            , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type2(InsnGeneral { y : A, op : Multiply        , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type2(InsnGeneral { y : D, op : Multiply        , imm : 3u8.into() } ) , z : B, x : C, dd : NoLoad     },
        Instruction { kind : Type2(InsnGeneral { y : D, op : Pack            , imm : 3u8.into() } ) , z : B, x : C, dd : StoreLeft  },
        Instruction { kind : Type2(InsnGeneral { y : D, op : BitwiseAndn     , imm : 3u8.into() } ) , z : B, x : C, dd : StoreRight },
        Instruction { kind : Type2(InsnGeneral { y : D, op : TestBit         , imm : 3u8.into() } ) , z : B, x : C, dd : LoadRight  },
        Instruction { kind : Type3(0x12345_i32.try_into()?), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type3(0x12345_i32.try_into()?), z : B, x : A, dd : NoLoad },
        Instruction { kind : Type3(3u8.into())             , z : B, x : C, dd : NoLoad },
        Instruction { kind : Type3(3u8.into())             , z : B, x : A, dd : NoLoad },
    ];

    assert_eq!(from, to);

    Ok(())
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Register {
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P,
}

impl fmt::Display for Register {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        (self as &dyn fmt::Debug).fmt(f)
    }
}

#[allow(dead_code)] // not all Opcodes are popular
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
        // Support a tiny, inconsistent subset of formatting commands
        match f.align() {
            Some(fmt::Alignment::Center) => write!(f, "{:^3}", s),
            _ => write!(f, "{}", s),
        }
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

impl<T : BitWidth> SizedImmediate<T> {
    pub const ZERO : SizedImmediate<T> = SizedImmediate(0, PhantomData);
}

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

impl<T : BitWidth> Immediate<T> {
    pub const ZERO : Immediate<T> = Immediate::Fixed(SizedImmediate::ZERO);
}

impl<T : BitWidth> fmt::Display for Immediate<T> {
    fn fmt(&self, f : &mut Formatter) -> Result<(), fmt::Error> {
        let d : &dyn Display = match self {
            Immediate::Fixed(x) => x,
            Immediate::Expr(x)  => x,
        };
        write!(f, "{}", d.to_string())
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

impl From<i32> for SmallestImmediate {
    fn from(n : i32) -> Self {
        use SmallestImmediate::*;

        Immediate12::try_from(n).map(Imm12)
            .or_else(|_| Immediate20::try_from(n).map(Imm20))
            .unwrap_or_else(|_| Imm32(n))
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
            Type3(Immediate::Fixed(imm)) if imm == 0u8.into()
                => a,
            Type3(Immediate::Fixed(imm)) if i32::from(imm) < 0
                => format!("{a} - {c}", a=a, c=(-i32::from(imm)).to_string()),
            Type3(..)
                => format!("{a} + {c}", a=a, c=c),
            Type0(Gen { op, imm : Immediate::Fixed(imm), .. }) if imm == 0u8.into()
                => format!("{a} {op:^3} {b}", a=a, b=b, op=op),
            Type1(Gen { op, y, .. }) | Type2(Gen { op, y, .. }) if y == Register::A
                => format!("{a} {op:^3} {b}", a=a, b=b, op=op),
            Type0(Gen { op, imm : Immediate::Fixed(imm), .. }) if i32::from(imm) < 0
                => format!("{a} {op:^3} {b} - {imm}", a=a, b=b, imm=(-i32::from(imm)).to_string(), op=op),
            Type0(Gen { op, .. }) |
            Type1(Gen { op, .. }) |
            Type2(Gen { op, .. })
                => format!("{a} {op:^3} {b} + {c}", a=a, b=b, c=c, op=op),
        };

        use MemoryOpType::*;
        match self.dd {
            NoLoad     => write!(f, " {}  <-  {}" , self.z, rhs),
            StoreRight => write!(f, " {}  -> [{}]", self.z, rhs),
            StoreLeft  => write!(f, "[{}] <-  {}" , self.z, rhs),
            LoadRight  => write!(f, " {}  <- [{}]", self.z, rhs),
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
        (" B  <-  C  |  D - 3"  , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : neg3_12.clone(), y : D, op : BitwiseOr       }) }),
        (" B  <-  C |~  D"      , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : zero_12.clone(), y : D, op : BitwiseOrn      }) }),
        (" B  <-  C  &  -3 + D" , Insn { dd : NoLoad    , z : B, x : C, kind : Type1(Gen { imm : neg3_12.clone(), y : D, op : BitwiseAnd      }) }),
        (" B  <-  C &~  0 + D"  , Insn { dd : NoLoad    , z : B, x : C, kind : Type1(Gen { imm : zero_12.clone(), y : D, op : BitwiseAndn     }) }),
        (" B  <-  -3  ^  C + D" , Insn { dd : NoLoad    , z : B, x : C, kind : Type2(Gen { imm : neg3_12.clone(), y : D, op : BitwiseXor      }) }),
        (" B  <-  0 ^^  C + D"  , Insn { dd : NoLoad    , z : B, x : C, kind : Type2(Gen { imm : zero_12.clone(), y : D, op : Pack            }) }),
        (" B  <-  C >>  D - 3"  , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : neg3_12.clone(), y : D, op : ShiftRightArith }) }),
        (" B  <-  C >>> D"      , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : zero_12.clone(), y : D, op : ShiftRightLogic }) }),
        (" B  <-  C ==  A - 3"  , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : neg3_12.clone(), y : A, op : CompareEq       }) }),
        (" B  <-  C  @  A"      , Insn { dd : NoLoad    , z : B, x : C, kind : Type0(Gen { imm : zero_12.clone(), y : A, op : TestBit         }) }),
        (" P  <-  C - 4"        , Insn { dd : NoLoad    , z : P, x : C, kind : Type3(neg4_20.clone()) }),
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
fn test_basicblock_display() -> Result<(), Box<dyn std::error::Error>> {
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

