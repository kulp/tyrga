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
pub enum Atom<'e> {
    Variable(String),
    Immediate(i32),
    Expression(&'e Expr<'e>),
}

impl <'a> fmt::Display for Atom<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Atom::*;
        match self {
            Variable(name) => write!(f, "{}", name),
            Immediate(num) => write!(f, "{}", num),
            Expression(e)  => write!(f, "({})", e.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr<'a> {
    pub a  : Atom<'a>,
    pub b  : Atom<'a>,
    pub op : Operation,
}

impl <'a> fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{a} {op} {b}", a=self.a, b=self.b, op=self.op)
    }
}

#[test]
fn test_expr_display() {
    use Atom::*;
    use Operation::*;

    let e = Expr { a : Variable("A".to_string()), op : Add, b : Immediate(3)              };
    let f = Expr { a : Expression(&e)           , op : Mul, b : Variable("B".to_string()) };
    let g = Expr { a : Expression(&e)           , op : Sub, b : Expression(&f)            };

    assert_eq!(e.to_string(), "A + 3");
    assert_eq!(f.to_string(), "(A + 3) * B");
    assert_eq!(g.to_string(), "(A + 3) - ((A + 3) * B)");
}

