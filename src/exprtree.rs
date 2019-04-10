#![allow(dead_code)]

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Operation::*;
        write!(f, "{}", match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand<'e, 's> {
    Variable(&'s str),
    Immediate(i32),
    Expression(&'e Expr<'e, 's>),
}

impl <'a, 's> fmt::Display for Operand<'a, 's> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Operand::*;
        match self {
            Variable(name) => write!(f, "{}", name),
            Immediate(num) => write!(f, "{}", num),
            Expression(e)  => write!(f, "({})", e.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr<'a, 's> {
    a  : Operand<'a, 's>,
    b  : Operand<'a, 's>,
    op : Operation,
}

impl <'a, 's> fmt::Display for Expr<'a, 's> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{a} {op} {b}", a=self.a, b=self.b, op=self.op)
    }
}

#[test]
fn test_expr_display() {
    use Operand::*;
    use Operation::*;

    let e = Expr { a : Variable("A") , op : Add, b : Immediate(3)   };
    let f = Expr { a : Expression(&e), op : Mul, b : Variable("B")  };
    let g = Expr { a : Expression(&e), op : Sub, b : Expression(&f) };

    assert_eq!(e.to_string(), "A + 3");
    assert_eq!(f.to_string(), "(A + 3) * B");
    assert_eq!(g.to_string(), "(A + 3) - ((A + 3) * B)");
}

