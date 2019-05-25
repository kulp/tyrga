#![allow(unused_macros)]

use enum_primitive::*;

use std::convert::Infallible;
use std::convert::TryFrom;
use std::fmt;
use std::marker::PhantomData;

use crate::exprtree;

macro_rules! tenyr_op {
    ( |   ) => { $crate::tenyr::Opcode::BitwiseOr       };
    ( &   ) => { $crate::tenyr::Opcode::BitwiseAnd      };
    ( ^   ) => { $crate::tenyr::Opcode::BitwiseXor      };
    ( >>  ) => { $crate::tenyr::Opcode::ShiftRightArith };

    ( |~  ) => { $crate::tenyr::Opcode::BitwiseOrn      };
    ( &~  ) => { $crate::tenyr::Opcode::BitwiseAndn     };
    ( ^^  ) => { $crate::tenyr::Opcode::Pack            };
    ( >>> ) => { $crate::tenyr::Opcode::ShiftRightLogic };

    ( +   ) => { $crate::tenyr::Opcode::Add             };
    ( *   ) => { $crate::tenyr::Opcode::Multiply        };
    ( ==  ) => { $crate::tenyr::Opcode::CompareEq       };
    ( <   ) => { $crate::tenyr::Opcode::CompareLt       };

    ( -   ) => { $crate::tenyr::Opcode::Subtract        };
    ( <<  ) => { $crate::tenyr::Opcode::ShiftLeft       };
    ( @   ) => { $crate::tenyr::Opcode::TestBit         };
    ( >=  ) => { $crate::tenyr::Opcode::CompareGe       };
}

#[test]
fn test_macro_ops() {
    use Opcode::*;

    assert_eq!(tenyr_op!( |   ), BitwiseOr       );
    assert_eq!(tenyr_op!( &   ), BitwiseAnd      );
    assert_eq!(tenyr_op!( ^   ), BitwiseXor      );
    assert_eq!(tenyr_op!( >>  ), ShiftRightArith );

    assert_eq!(tenyr_op!( |~  ), BitwiseOrn      );
    assert_eq!(tenyr_op!( &~  ), BitwiseAndn     );
    assert_eq!(tenyr_op!( ^^  ), Pack            );
    assert_eq!(tenyr_op!( >>> ), ShiftRightLogic );

    assert_eq!(tenyr_op!( +   ), Add             );
    assert_eq!(tenyr_op!( *   ), Multiply        );
    assert_eq!(tenyr_op!( ==  ), CompareEq       );
    assert_eq!(tenyr_op!( <   ), CompareLt       );

    assert_eq!(tenyr_op!( -   ), Subtract        );
    assert_eq!(tenyr_op!( <<  ), ShiftLeft       );
    assert_eq!(tenyr_op!( @   ), TestBit         );
    assert_eq!(tenyr_op!( >=  ), CompareGe       );
}

// Some tenyr ops are more than one token, so require special treatment
macro_rules! tenyr_get_op {
    ( $callback:ident                       ) => { { let op = tenyr_op!( | ); $callback!(op            ) } };

    ( $callback:ident |~     $( $rest:tt )+ ) => { { let op = tenyr_op!(|~ ); $callback!(op $( $rest )+) } };
    ( $callback:ident &~     $( $rest:tt )+ ) => { { let op = tenyr_op!(&~ ); $callback!(op $( $rest )+) } };
    ( $callback:ident ^^     $( $rest:tt )+ ) => { { let op = tenyr_op!(^^ ); $callback!(op $( $rest )+) } };
    ( $callback:ident >>>    $( $rest:tt )+ ) => { { let op = tenyr_op!(>>>); $callback!(op $( $rest )+) } };
    ( $callback:ident $op:tt $( $rest:tt )+ ) => { { let op = tenyr_op!($op); $callback!(op $( $rest )+) } };
}

macro_rules! tenyr_type013 {
    ( $opname:ident ( $imm:expr ) $( + $y:ident )? ) => {
        {
            use $crate::tenyr::*;
            use std::convert::TryInto;
            let gen = InsnGeneral { y : Register::A, op : Opcode::BitwiseOr, imm : 0u8.into() };
            let kind = Type1(InsnGeneral { $( y : $y, )? op : $opname, imm : $imm.try_into().map_err::<Box<std::error::Error>,_>(Into::into)?, ..gen });
            let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, z : Register::A, x : Register::A, dd : MemoryOpType::NoLoad });
            result
        }
    };
    ( $opname:ident $imm:literal $( + $y:ident )? ) => {
        {
            use $crate::tenyr::*;
            use std::convert::TryInto;
            let gen = InsnGeneral { y : Register::A, op : Opcode::BitwiseOr, imm : 0u8.into() };
            let kind = Type1(InsnGeneral { $( y : $y, )? op : $opname, imm : $imm.try_into().map_err::<Box<std::error::Error>,_>(Into::into)?, ..gen });
            let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, z : Register::A, x : Register::A, dd : MemoryOpType::NoLoad });
            result
        }
    };
    ( $opname:ident $( $y:ident $( + ( $imm:expr ) )? )? ) => {
        {
            use $crate::tenyr::*;
            #[allow(unused_imports)] use std::convert::TryInto;
            let gen = InsnGeneral { y : Register::A, op : Opcode::BitwiseOr, imm : 0u8.into() };
            let kind = Type0(InsnGeneral { op : $opname, $( y : $y, $( imm : $imm.try_into().map_err::<Box<std::error::Error>,_>(Into::into)?, )? )?  ..gen });
            let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, z : Register::A, x : Register::A, dd : MemoryOpType::NoLoad });
            result
        }
    };
    ( $opname:ident $( $y:ident $( + $imm:literal )? )? ) => {
        {
            use $crate::tenyr::*;
            #[allow(unused_imports)] use std::convert::TryInto;
            let gen = InsnGeneral { y : Register::A, op : Opcode::BitwiseOr, imm : 0u8.into() };
            let kind = Type0(InsnGeneral { op : $opname, $( y : $y, $( imm : $imm.try_into().map_err::<Box<std::error::Error>,_>(Into::into)?, )? )?  ..gen });
            let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, z : Register::A, x : Register::A, dd : MemoryOpType::NoLoad });
            result
        }
    };
}

macro_rules! tenyr_type2 {
    ( $opname:ident $x:ident $( + $y:ident )? ) => {
        {
            use $crate::tenyr::*;
            let gen = InsnGeneral { y : Register::A, op : Opcode::BitwiseOr, imm : 0u8.into() };
            let kind = Type2(InsnGeneral { $( y : $y, )? op : $opname, ..gen });
            let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, z : Register::A, x : $x, dd : MemoryOpType::NoLoad });
            result
        }
    };
}

macro_rules! tenyr_rhs {
    ( $( $x:ident + )? ( $imm:expr ) ) => {
        {
            use $crate::tenyr::*;
            use std::convert::TryInto;
            let imm = $imm.try_into().map_err::<Box<std::error::Error>,_>(Into::into)?;
            let kind = Type3(imm);
            #[allow(unused_variables)] let x = Register::A;
            $( let x = $x; )?
            let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, z : Register::A, x, dd : MemoryOpType::NoLoad });
            result
        }
    };
    ( $( $x:ident + )? $imm:literal ) => {
        {
            use $crate::tenyr::*;
            use std::convert::TryInto;
            let imm = $imm.try_into().map_err::<Box<std::error::Error>,_>(Into::into)?;
            let kind = Type3(imm);
            #[allow(unused_variables)] let x = Register::A;
            $( let x = $x; )?
            let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, z : Register::A, x, dd : MemoryOpType::NoLoad });
            result
        }
    };
    ( $x:ident $( $rest:tt )* ) => {
        {
            use $crate::tenyr::*;
            let base = tenyr_get_op!(tenyr_type013 $( $rest )*);
            let result : Result<_, Box<std::error::Error>> = Ok(Instruction { z : Register::A, x : $x, dd : MemoryOpType::NoLoad, ..base? });
            result
        }
    };
    ( ( $imm:expr ) $( $rest:tt )+ ) => {
        {
            use $crate::tenyr::*;
            use std::convert::TryInto;
            let base = tenyr_get_op!(tenyr_type2 $( $rest )*)?;
            if let Type2(gen) = base.kind {
                let kind = Type2(InsnGeneral { imm : $imm.try_into().map_err::<Box<std::error::Error>,_>(Into::into)?, ..gen });
                let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, ..base });
                result
            } else {
                Err("internal error - did not get expected Type2".into())
            }
        }
    };
    ( $imm:literal $( $rest:tt )+ ) => {
        {
            use $crate::tenyr::*;
            use std::convert::TryInto;
            let base = tenyr_get_op!(tenyr_type2 $( $rest )*)?;
            if let Type2(gen) = base.kind {
                let kind = Type2(InsnGeneral { imm : $imm.try_into().map_err::<Box<std::error::Error>,_>(Into::into)?, ..gen });
                let result : Result<_, Box<std::error::Error>> = Ok(Instruction { kind, ..base });
                result
            } else {
                Err("internal error - did not get expected Type2".into())
            }
        }
    };
}

#[macro_export]
macro_rules! tenyr_insn {
    (   $z:ident   <- [ $( $rhs:tt )+ ] ) => { Ok(Instruction { z : $z, dd : $crate::tenyr::MemoryOpType::LoadRight, ..tenyr_rhs!( $( $rhs )+ )? }) as Result<_, Box<std::error::Error>> };
    (   $z:ident   <-   $( $rhs:tt )+   ) => { Ok(Instruction { z : $z, ..tenyr_rhs!( $( $rhs )+ )? }) as Result<_, Box<std::error::Error>> };
    ( [ $z:ident ] <-   $( $rhs:tt )+   ) => { Ok(Instruction { z : $z, dd : $crate::tenyr::MemoryOpType::StoreLeft, ..tenyr_rhs!( $( $rhs )+ )? }) as Result<_, Box<std::error::Error>> };
    (   $z:ident   -> [ $( $rhs:tt )+ ] ) => { Ok(Instruction { z : $z, dd : $crate::tenyr::MemoryOpType::StoreRight, ..tenyr_rhs!( $( $rhs )+ )? }) as Result<_, Box<std::error::Error>> };
}

#[test]
fn test_macro_insn() -> Result<(), Box<std::error::Error>> {
    use InstructionType::*;
    use MemoryOpType::*;
    use Opcode::*;
    use Register::*;

    use std::convert::TryInto;

    assert_eq!(tenyr_insn!( B  <-  C  |~ D + 3      ).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C >>> D + 3      ).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : ShiftRightLogic, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  +  D + 3      ).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : Add     , imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  *  D + 3      ).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  +  0x12345    ).unwrap(), Instruction { kind : Type3(0x12345_i32.try_into()?), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-        0x12345    ).unwrap(), Instruction { kind : Type3(0x12345_i32.try_into()?), z : B, x : A, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C                ).unwrap(), Instruction { kind : Type0(InsnGeneral { y : A, op : BitwiseOr , imm : 0u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  +  D          ).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : Add       , imm : 0u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  |~ D          ).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn, imm : 0u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  ^^ 3          ).unwrap(), Instruction { kind : Type1(InsnGeneral { y : A, op : Pack    , imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  3  *  C          ).unwrap(), Instruction { kind : Type2(InsnGeneral { y : A, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  3  *  C + D      ).unwrap(), Instruction { kind : Type2(InsnGeneral { y : D, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!([B] <-  3  ^^ C + D      ).unwrap(), Instruction { kind : Type2(InsnGeneral { y : D, op : Pack    , imm : 3u8.into() }), z : B, x : C, dd : StoreLeft });
    assert_eq!(tenyr_insn!( B  -> [3  &~ C + D]     ).unwrap(), Instruction { kind : Type2(InsnGeneral { y : D, op : BitwiseAndn, imm : 3u8.into() }), z : B, x : C, dd : StoreRight });
    assert_eq!(tenyr_insn!( B  <- [3  @  C + D]     ).unwrap(), Instruction { kind : Type2(InsnGeneral { y : D, op : TestBit , imm : 3u8.into() }), z : B, x : C, dd : LoadRight });

    let three = 3;

    assert_eq!(tenyr_insn!( B  <-  C  |~ D + (three)).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C >>> D + (three)).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : ShiftRightLogic, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  +  D + (three)).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : Add     , imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  *  D + (three)).unwrap(), Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  +  (three)    ).unwrap(), Instruction { kind : Type3(3u8.into()), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-        (three)    ).unwrap(), Instruction { kind : Type3(3u8.into()), z : B, x : A, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  C  ^^ (three)    ).unwrap(), Instruction { kind : Type1(InsnGeneral { y : A, op : Pack    , imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  (three) * C      ).unwrap(), Instruction { kind : Type2(InsnGeneral { y : A, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!( B  <-  (three) * C + D  ).unwrap(), Instruction { kind : Type2(InsnGeneral { y : D, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad });
    assert_eq!(tenyr_insn!([B] <-  (three) ^^ C + D ).unwrap(), Instruction { kind : Type2(InsnGeneral { y : D, op : Pack    , imm : 3u8.into() }), z : B, x : C, dd : StoreLeft });
    assert_eq!(tenyr_insn!( B  -> [(three) &~ C + D]).unwrap(), Instruction { kind : Type2(InsnGeneral { y : D, op : BitwiseAndn, imm : 3u8.into() }), z : B, x : C, dd : StoreRight });
    assert_eq!(tenyr_insn!( B  <- [(three) @  C + D]).unwrap(), Instruction { kind : Type2(InsnGeneral { y : D, op : TestBit , imm : 3u8.into() }), z : B, x : C, dd : LoadRight });

    Ok(())
}

#[macro_export]
macro_rules! tenyr_insn_list {
    () => { vec![] };
    ( $lhs:tt <- $a:tt $op:tt$op2:tt $b:tt $( + $c:tt )? ; $( $tok:tt )* ) => {
        {
            let insn : Result<Instruction,_> = tenyr_insn!($lhs <- $a $op$op2 $b $( + $c )? );
            std::iter::once(insn?).chain(tenyr_insn_list!($( $tok )*))
        }
    };
    ( $lhs:tt <- $a:tt $op:tt $b:tt $( + $c:tt )? ; $( $tok:tt )* ) => {
        {
            let insn : Result<Instruction,_> = tenyr_insn!($lhs <- $a $op $b $( + $c )? );
            std::iter::once(insn?).chain(tenyr_insn_list!($( $tok )*))
        }
    };
    ( $lhs:tt <- $rhs:tt ; $( $tok:tt )* ) => {
        {
            let insn : Result<Instruction,_> = tenyr_insn!($lhs <- $rhs );
            std::iter::once(insn?).chain(tenyr_insn_list!($( $tok )*))
        }
    };
    ( $lhs:tt -> $rhs:tt ; $( $tok:tt )* ) => {
        {
            let insn : Result<Instruction,_> = tenyr_insn!($lhs -> $rhs );
            std::iter::once(insn?).chain(tenyr_insn_list!($( $tok )*))
        }
    };
}

#[test]
fn test_macro_insn_list() -> Result<(), Box<std::error::Error>> {
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
         B  <-  C  +  0x12345    ;
         B  <-        0x12345    ;
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
         B  <-  C  +  (three)    ;
         B  <-        (three)    ;
         B  <-  C  ^^ (three)    ;
         B  <-  (three) * C      ;
         B  <-  (three) * C + D  ;
        [B] <-  (three) ^^ C + D ;
         B  -> [(three) &~ C + D];
         B  <- [(three) @  C + D];
    };

    let from : Vec<_> = from.collect();

    use std::convert::TryInto;

    let to = vec![
        Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : D, op : ShiftRightLogic, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Add     , imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type3(0x12345_i32.try_into()?), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type3(0x12345_i32.try_into()?), z : B, x : A, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : A, op : BitwiseOr , imm : 0u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Add       , imm : 0u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn, imm : 0u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type1(InsnGeneral { y : A, op : Pack    , imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type2(InsnGeneral { y : A, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type2(InsnGeneral { y : D, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type2(InsnGeneral { y : D, op : Pack    , imm : 3u8.into() }), z : B, x : C, dd : StoreLeft },
        Instruction { kind : Type2(InsnGeneral { y : D, op : BitwiseAndn, imm : 3u8.into() }), z : B, x : C, dd : StoreRight },
        Instruction { kind : Type2(InsnGeneral { y : D, op : TestBit , imm : 3u8.into() }), z : B, x : C, dd : LoadRight },
        Instruction { kind : Type0(InsnGeneral { y : D, op : BitwiseOrn, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : D, op : ShiftRightLogic, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Add     , imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type0(InsnGeneral { y : D, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type3(3u8.into()), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type3(3u8.into()), z : B, x : A, dd : NoLoad },
        Instruction { kind : Type1(InsnGeneral { y : A, op : Pack    , imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type2(InsnGeneral { y : A, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type2(InsnGeneral { y : D, op : Multiply, imm : 3u8.into() }), z : B, x : C, dd : NoLoad },
        Instruction { kind : Type2(InsnGeneral { y : D, op : Pack    , imm : 3u8.into() }), z : B, x : C, dd : StoreLeft },
        Instruction { kind : Type2(InsnGeneral { y : D, op : BitwiseAndn, imm : 3u8.into() }), z : B, x : C, dd : StoreRight },
        Instruction { kind : Type2(InsnGeneral { y : D, op : TestBit , imm : 3u8.into() }), z : B, x : C, dd : LoadRight },
    ];

    assert_eq!(from, to);

    Ok(())
}

enum_from_primitive! {
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Register {
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P,
}
}

impl fmt::Display for Register {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
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

use std::fmt::{Display, Error, Formatter};

impl<T : BitWidth> Display for SizedImmediate<T> {
    fn fmt(&self, f : &mut Formatter) -> Result<(), Error> {
        write!(f, "{}", self.0.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Immediate<T : BitWidth> {
    Fixed(SizedImmediate<T>),
    Expr(exprtree::Atom),
}

impl<T : BitWidth> fmt::Display for Immediate<T> {
    fn fmt(&self, f : &mut Formatter) -> Result<(), Error> {
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
fn test_basicblock_display() {
    let (_, insns) : (Vec<_>, Vec<_>) = instruction_test_cases().iter().cloned().unzip();
    let label = "testbb".to_string();
    let bb = BasicBlock { label, insns };
    let ss = bb.to_string();
    let first_line = ss.lines().nth(0).expect("unexpectedly empty input");
    assert_eq!(':', first_line.chars().last().expect("unexpected empty line"));
    assert_eq!(bb.label, first_line[..first_line.len()-1]);
    assert_eq!(bb.insns.len() + 1, ss.lines().count());
}

