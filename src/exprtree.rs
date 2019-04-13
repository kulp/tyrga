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
pub enum Atom {
    Variable(String),
    Immediate(i32),
    Expression(std::rc::Rc<Expr>),
}

impl fmt::Display for Atom {
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
pub struct Expr {
    pub a  : Atom,
    pub b  : Atom,
    pub op : Operation,
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{a} {op} {b}", a=self.a, b=self.b, op=self.op)
    }
}

#[test]
fn test_expr_display() {
    use Atom::*;
    use Operation::*;
    use std::rc::Rc;

    let e = Expr { a : Variable("A".to_string()), op : Add, b : Immediate(3)              };
    let ee = Rc::new(e);
    let f = Expr { a : Expression(ee.clone())   , op : Mul, b : Variable("B".to_string()) };
    let ff = Rc::new(f);
    let g = Expr { a : Expression(ee.clone())   , op : Sub, b : Expression(ff.clone())    };

    assert_eq!(ee.to_string(), "A + 3");
    assert_eq!(ff.to_string(), "(A + 3) * B");
    assert_eq!(g.to_string(), "(A + 3) - ((A + 3) * B)");
}

