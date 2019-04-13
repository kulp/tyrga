mod deadcode;
mod exprtree;
mod jvmtypes;
mod mangling;
mod tenyr;

use std::error::Error;
use std::fmt;
use std::ops::Range;

use jvmtypes::*;

use classfile_parser::ClassFile;

use tenyr::Register;

pub struct StackManager {
    stack : Vec<Register>,
    top : usize,
}

// Someday we will be able to find operands in a stack frame
#[derive(Copy,Clone,Debug,PartialEq,Eq)]
pub enum OperandLocation {
    Register(tenyr::Register),
}

impl From<Register> for OperandLocation {
    fn from(r : Register) -> OperandLocation { OperandLocation::Register(r) }
}

// This simple StackManager implementation does not do spilling to nor reloading from memory.
// For now, it panics if we run out of free registers.
impl StackManager {
    pub fn new(r : Vec<Register>) -> StackManager {
        StackManager { top : 0, stack : r }
    }

    pub fn reserve(&mut self, n : usize) {
        assert!(self.top + n <= self.stack.len());
        self.top += n;
    }

    pub fn release(&mut self, n : usize) {
        assert!(self.top >= n);
        self.top -= n;
    }

    pub fn get(&self, which : usize) -> OperandLocation {
        assert!(which <= self.top);
        // indexing is relative to top of stack, counting backward
        self.stack[self.top - which - 1].into()
    }
}

#[test]
#[should_panic(expected=">= n")]
fn test_underflow() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(v);
    sm.reserve(3);
    sm.release(4);
}

#[test]
#[should_panic(expected="n <=")]
fn test_overflow() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let len = v.len();
    let mut sm = StackManager::new(v);
    sm.reserve(len + 1);
}

#[test]
fn test_normal_stack() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let t = v.clone();
    let mut sm = StackManager::new(v);
    let off = 3;
    sm.reserve(off);
    assert_eq!(sm.get(0), t[off - 1].into());
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Destination {
    Successor,
    Address(usize),
}

fn make_instructions(sm : &mut StackManager, (_addr, op) : (&usize, &Operation))
    -> (Vec<tenyr::Instruction>, Vec<Destination>)
{
    use Operation::*;
    use tenyr::Immediate20;
    use tenyr::Instruction;
    use tenyr::InstructionType::*;
    use tenyr::MemoryOpType::*;
    use tenyr::SizedImmediate;

    let stack_ptr = Register::O;
    let frame_ptr = Register::N;
    let temp_reg  = Register::M;

    let default_dest = vec![Destination::Successor];

    // While the only valid OperandLocation is Register, this shorthand is convenient.
    let get_reg = |x| match x { OperandLocation::Register(r) => r };

    let translate_arithmetic_op =
        |x| {
            use tenyr::Opcode::*;
            match x {
                ArithmeticOperation::Add  => Some(Add),
                ArithmeticOperation::Sub  => Some(Subtract),
                ArithmeticOperation::Mul  => Some(Multiply),
                ArithmeticOperation::Shl  => Some(ShiftLeft),
                ArithmeticOperation::Shr  => Some(ShiftRightArith),
                ArithmeticOperation::Ushr => Some(ShiftRightLogic),
                ArithmeticOperation::And  => Some(BitwiseAnd),
                ArithmeticOperation::Or   => Some(BitwiseOr),
                ArithmeticOperation::Xor  => Some(BitwiseXor),
                _ => None,
            }
        };

    let make_mov   = |to, from| Instruction { dd : NoLoad, kind : Type3(Immediate20::ZERO), z : to, x : from };
    let make_load  = |to, from| Instruction { dd : LoadRight , ..make_mov(to, from) };
    let make_store = |lhs, rhs| Instruction { dd : StoreRight, ..make_mov(lhs, rhs) };

    match *op {
        Constant { kind, value } if kind == JType::Int => {
            let kind = Type3(SizedImmediate::new(value).expect("immediate too large"));
            sm.reserve(1);
            let z = match sm.get(0) {
                OperandLocation::Register(r) => r,
            };
            (vec![ Instruction { kind, z, x : Register::A, dd : NoLoad } ], default_dest)
        },
        Yield { kind } if kind == JType::Void
            => (vec![
                    make_mov(stack_ptr, frame_ptr),
                    make_load(Register::P, stack_ptr),
                ], default_dest),

        Arithmetic { kind, op } if kind == JType::Int && translate_arithmetic_op(op).is_some()
            => {
                use tenyr::*;
                let y = get_reg(sm.get(0));
                let x = get_reg(sm.get(1));
                let z = x;
                let op = translate_arithmetic_op(op).unwrap();
                let dd = MemoryOpType::NoLoad;
                let imm = Immediate12::ZERO;
                let v = vec![ Instruction { kind : Type0(InsnGeneral { y, op, imm }), x, z, dd } ];
                sm.release(1);
                (v, default_dest)
            },
        LoadLocal { kind, index } | StoreLocal { kind, index }
            if kind == JType::Int || kind == JType::Object
            => {
                let index = i32::from(index);
                match *op { LoadLocal { .. } => sm.reserve(1), _ => {} };
                let v = {
                    use tenyr::*;
                    let x = frame_ptr;
                    let z = get_reg(sm.get(0));
                    let dd = match *op {
                        LoadLocal  { .. } => MemoryOpType::LoadRight,
                        StoreLocal { .. } => MemoryOpType::StoreRight,
                        _ => unreachable!(),
                    };
                    let imm = Immediate20::new(-index).unwrap();
                    vec![ Instruction { kind : Type3(imm), x, z, dd } ]
                };
                match *op { StoreLocal { .. } => sm.release(1), _ => {} };
                (v, default_dest)
            },
        Increment { index, value } => {
            use tenyr::*;
            let make_imm20 = |n| Immediate20::new(n).unwrap();
            let make_imm12 = |n| Immediate12::new(n).unwrap();
            let index = i32::from(index);
            let imm = make_imm12(value);
            let y = Register::A;
            let op = Opcode::Add;
            let v = vec![
                Instruction { kind : Type3(make_imm20(-index)), ..make_load(temp_reg, frame_ptr) },
                Instruction { kind : Type1(InsnGeneral { y, op, imm }), ..make_mov(temp_reg, temp_reg) },
                Instruction { kind : Type3(make_imm20(-index)), ..make_store(temp_reg, frame_ptr) },
            ];

            (v, default_dest)
        },

        _ => panic!("unhandled operation {:?}", op),
    }
}

#[test]
fn test_make_instruction() {
    use tenyr::MemoryOpType::*;
    use Register::*;
    use tenyr::Instruction;
    use tenyr::InstructionType::*;
    use tenyr::SizedImmediate;
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(v);
    let op = Operation::Constant { kind : JType::Int, value : 5 };
    let insn = make_instructions(&mut sm, (&0, &op));
    let imm = SizedImmediate::new(5).unwrap();
    assert_eq!(insn.0, vec![ Instruction { kind: Type3(imm), z: C, x: A, dd: NoLoad } ]);
    assert_eq!(insn.0[0].to_string(), " C  <-  5");
}

#[derive(Debug)]
pub struct TranslationError(String);

impl TranslationError {
    fn new(msg: &str) -> TranslationError {
        TranslationError(msg.to_string())
    }
}

impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for TranslationError {
    fn description(&self) -> &str { &self.0 }
}

pub type Result<T> = std::result::Result<T, Box<Error>>;

fn generic_error<E>(e : E) -> Box<TranslationError>
    where E : std::error::Error
{
    Box::new(TranslationError::new(&format!("unknown error: {}", e)))
}

#[cfg(test)]
fn parse_class(stem : &str) -> ClassFile {
    let mut name = String::from(concat!(env!("OUT_DIR"), "/"));
    name.push_str(stem);
    classfile_parser::parse_class(&name).expect("failed to parse class")
}

use classfile_parser::attribute_info::StackMapFrame;
use std::collections::BTreeMap;
fn derive_ranges<'a, T>(body : &[(usize, &'a T)], table : &[StackMapFrame])
    -> Result<(Vec<Range<usize>>, BTreeMap<usize, &'a T>)>
{
    use classfile_parser::attribute_info::StackMapFrame::*;
    let get_delta = |f : &StackMapFrame| match *f {
        SameFrame                           { frame_type }       => u16::from(frame_type),

        SameLocals1StackItemFrame           { frame_type, .. }   => u16::from(frame_type) - 64,

        SameLocals1StackItemFrameExtended   { offset_delta, .. }
            | ChopFrame                     { offset_delta, .. }
            | SameFrameExtended             { offset_delta, .. }
            | AppendFrame                   { offset_delta, .. }
            | FullFrame                     { offset_delta, .. } => offset_delta,
    };
    let deltas : Vec<u16> = table.iter().map(get_delta).collect();

    let before = deltas.iter().take(1);
    let after  = deltas.iter().skip(1);
    let err = Box::new(TranslationError::new("body unexpectedly empty"));
    let max = body.last().ok_or(err)?.0 + 1;
    use std::iter::once;
    let ranges =
        once(0)
            .chain(before.cloned())
            .chain(once(0)).chain(after.map(|&n| n + 1))
            .scan(0, |state, x| { *state += x; Some(usize::from(*state)) })
            .chain(once(max))
            .collect::<Vec<_>>()
            .windows(2)
            .map(|x| x[0]..x[1])
            .filter(|x| x.len() > 0)
            .collect::<Vec<_>>();

    let tree = body.iter().cloned().collect();
    Ok((ranges, tree))
}

use classfile_parser::method_info::MethodInfo;
fn get_ranges_for_method(class : &ClassFile, method : &MethodInfo)
    -> Result<(Vec<Range<usize>>, BTreeMap<usize, Operation>)>
{
    use classfile_parser::attribute_info::AttributeInfo;
    use classfile_parser::attribute_info::code_attribute_parser;
    use classfile_parser::attribute_info::stack_map_table_attribute_parser;
    use classfile_parser::constant_info::ConstantInfo::Utf8;

    let get_constant = |n| &class.const_pool[usize::from(n) - 1];
    let name_of = |a : &AttributeInfo|
        match get_constant(a.attribute_name_index) {
            Utf8(u) => u.utf8_string.to_string(),
            _ => panic!("not a name")
        };

    let code = code_attribute_parser(&method.attributes[0].info).map_err(generic_error)?.1;
    let attr = &code.attributes.iter().find(|a| name_of(a) == "StackMapTable");
    let keep;
    let table = match attr {
        Some(attr) => {
            keep = stack_map_table_attribute_parser(&attr.info).map_err(generic_error)?;
            &keep.1.entries
        },
        _ => &[] as &[StackMapFrame],
    };

    use classfile_parser::code_attribute::code_parser;
    let vec = code_parser(&code.code).map_err(generic_error)?.1;
    let refed = vec.iter().map(|(s, x)| (*s, x)).collect::<Vec<_>>();
    let (ranges, map) = derive_ranges(&refed, table)?;
    let ops = map.into_iter().map(decode_insn).collect::<BTreeMap<_,_>>();
    Ok((ranges, ops))
}

fn make_mangled_method_name(class : &ClassFile, method : &MethodInfo) -> String {
    use classfile_parser::constant_info::ConstantInfo::*;

    let get_constant = |n| &class.const_pool[usize::from(n) - 1];
    let get_string = |i|
        match get_constant(i) {
            Utf8(u) => Some(u.utf8_string.to_string()),
            _ => None,
        };

    let cl = match get_constant(class.this_class) { Class(c) => c, _ => panic!("not a class") };
    let s = vec![
        get_string(cl.name_index).expect("bad class name"),
        get_string(method.name_index).expect("bad method name"),
        get_string(method.descriptor_index).expect("bad method descriptor"),
    ].join(":");

    mangling::mangle(s.bytes()).expect("failed to mangle")
}

fn make_label(class : &ClassFile, method : &MethodInfo, suffix : &str) -> String {
    format!(".L{}{}",
        make_mangled_method_name(class, method),
        mangling::mangle(format!(":__{}", suffix).bytes()).expect("failed to mangle"))
}

#[cfg(test)]
fn test_stack_map_table(stem : &str) {
    let class = parse_class(stem);
    for method in &class.methods {
        let (ranges, ops) = get_ranges_for_method(&class, &method).expect("failed to get ranges for map");
        let it = ranges.into_iter().map(|x| ops.range(x).collect::<Vec<_>>());
        let vov : Vec<_> = it.collect();
        let r : Vec<_> = vov.concat();
        assert!(r.len() > 0);
    }
}

#[cfg(test)]
const CLASS_LIST : &[&'static str] = &[
    "Except",
    "Expr",
    "GCD",
    "Nest",
    "Sieve",
    "Switch",
    "Tiny",
];

#[test]
fn test_parse_classes()
{
    for name in CLASS_LIST {
        test_stack_map_table(name);
    }
}

fn main() {
}

