use std::borrow::Cow;
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
    Variable(Cow<'static, str>),
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

impl From<&'static str> for Atom {
    fn from(s : &'static str) -> Self { Atom::Variable(Cow::from(s)) }
}

impl From<String> for Atom {
    fn from(s : String) -> Self { Atom::Variable(Cow::from(s)) }
}

impl From<Expr> for Atom {
    fn from(e : Expr) -> Self { Atom::Expression(Box::new(e)) }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr {
    pub a :  Atom,
    pub b :  Atom,
    pub op : Operation,
}

impl Expr {
    pub fn make_atplus_expr(a : Atom) -> Expr {
        Expr {
            a,
            op : Operation::Sub,
            b : Atom::Expression(
                Expr {
                    a :  ".".into(),
                    op : Operation::Add,
                    b :  Atom::Immediate(1),
                }
                .into(),
            ),
        }
    }
    fn is_atplus_shorthand(&self) -> bool {
        matches!(self,
            Expr {
                a: _,
                op: Operation::Sub,
                b: Atom::Expression(b),
            } if matches!(**b,
                Expr {
                    a: Atom::Variable(ref a),
                    op: Operation::Add,
                    b: Atom::Immediate(1),
                } if a == "."))
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        let (a, b, op) = (&self.a, &self.b, &self.op);
        if self.is_atplus_shorthand() {
            write!(f, "@+{a}")
        } else {
            write!(f, "({a} {op} {b})")
        }
    }
}

#[test]
fn test_expr_display() {
    use Atom::*;
    use Operation::*;

    let e = Expr {
        a :  Variable("A".into()),
        op : Add,
        b :  Immediate(3),
    };
    let e = Box::new(e);
    let f = Expr {
        a :  Expression(e.clone()),
        op : Sub,
        b :  Variable("B".into()),
    };
    let f = Box::new(f);
    let g = Expr {
        a :  Expression(e.clone()),
        op : Sub,
        b :  Expression(f.clone()),
    };

    let h = Expr {
        a :  Variable("abc".into()),
        op : Sub,
        b :  Expression(Box::new(Expr {
            a :  ".".into(),
            op : Add,
            b :  Immediate(1),
        })),
    };
    let i = Expr {
        a :  Expression(Box::new(h.clone())),
        op : Add,
        b :  Immediate(4),
    };

    assert_eq!(e.to_string(), "(A + 3)");
    assert_eq!(f.to_string(), "((A + 3) - B)");
    assert_eq!(g.to_string(), "((A + 3) - ((A + 3) - B))");
    assert_eq!(h.to_string(), "@+abc");
    assert_eq!(i.to_string(), "(@+abc + 4)");
}
