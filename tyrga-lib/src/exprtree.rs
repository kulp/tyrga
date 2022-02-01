use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Add,
    Sub,
}

impl fmt::Display for Operation {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        use Operation::*;
        match self {
            Add => write!(f, "+"),
            Sub => write!(f, "-"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Atom {
    Variable(String),
    Immediate(i32),
    Expression(Box<Expr>),
}

impl fmt::Display for Atom {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        use Atom::*;
        match self {
            Variable(name) => write!(f, "{name}"),
            Immediate(num) => write!(f, "{num}"),
            Expression(e) => write!(f, "{e}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr {
    pub a :  Atom,
    pub b :  Atom,
    pub op : Operation,
}

impl fmt::Display for Expr {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        let (a, b, op) = (&self.a, &self.b, &self.op);
        write!(f, "({a} {op} {b})")
    }
}

#[test]
fn test_expr_display() {
    use Atom::*;
    use Operation::*;

    let e = Expr {
        a :  Variable("A".to_string()),
        op : Add,
        b :  Immediate(3),
    };
    let e = Box::new(e);
    let f = Expr {
        a :  Expression(e.clone()),
        op : Sub,
        b :  Variable("B".to_string()),
    };
    let f = Box::new(f);
    let g = Expr {
        a :  Expression(e.clone()),
        op : Sub,
        b :  Expression(f.clone()),
    };

    assert_eq!(e.to_string(), "(A + 3)");
    assert_eq!(f.to_string(), "((A + 3) - B)");
    assert_eq!(g.to_string(), "((A + 3) - ((A + 3) - B))");
}
