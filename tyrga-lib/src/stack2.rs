//! Provides operations for manipulating the state of the JVM operand stack.
//!
//! These are the essential quantites:
//! - register_count (constant, independent; number of registers usable)
//! - stack_depth    (variable, independent; number of items on operand stack)
//! - pick_point     (variable, independent; number of "live" registers desired)
//! - spilled_count  (variable,   dependent; number of values stored in memory)
//!
//! The quantities are related in the following ways:
//! - 0 <= stack_depth <= 255
//! - 0 <= pick_point <= stack_depth
//! - spilled_count = max(0, pick_point - register_count)
//!
//! These are the public operations:
//! - new(R)     -- create manager for register list R
//! - reserve(N) -- allocate N locations at the top of stack
//! - release(N) -- deallocate N locations from the top of stack
//! - empty()    -- release all contents of the stack
//! - get(N)     -- get location for offset N >= 0, down from top of stack
//! - freeze()   -- spill all locations to memory
//! - thaw()     -- load as many locations into registers as will fit

use crate::tenyr::Instruction;
use crate::tenyr::Register;

use std::convert::TryFrom;
use std::convert::TryInto;

#[cfg(test)]
use quickcheck::{quickcheck, Gen, TestResult};

/// a list of stack-maintenance instructions that must be executed
#[must_use = "StackActions must be implemented to maintain stack discipline"]
pub type StackActions = Vec<Instruction>;

struct Manager {
    /// registers under our control, excluding `stack_ptr`
    regs : Vec<Register>,
    /// number of registers in our stack
    register_count : u16,
    /// number of items on operand stack
    stack_depth : u16,
    /// number of "live" registers desired
    pick_point : u16,
    /// the register used as a stack pointer
    stack_ptr : Register,
}

impl Manager {
    /// create manager for a given register list
    pub fn new<I : IntoIterator<Item = Register>>(regs : I) -> Manager {
        let mut regs : Vec<_> = regs.into_iter().collect();
        let stack_depth = 0;
        let pick_point = 0;
        let stack_ptr = regs.pop().expect("too few registers");
        let register_count = regs.len().try_into().expect("too many registers");
        Manager {
            regs,
            register_count,
            stack_depth,
            pick_point,
            stack_ptr,
        }
    }
    /// number of values stored in memory
    fn spilled_count(&self) -> u16 {
        let regs : i32 = self.register_count.into();
        let pick : i32 = self.pick_point.into();
        let deep : i32 = self.stack_depth.into();
        let count = 0.max(deep - regs.min(pick));
        count.try_into().expect("too many spilled registers")
    }

    fn unwrap<T>(f : impl FnOnce() -> Result<T, Box<dyn std::error::Error>>) -> T {
        #[allow(clippy::result_unwrap_used)]
        f().unwrap()
    }

    fn unwrapper<P, T>(f : impl Fn(P) -> Result<T, Box<dyn std::error::Error>>) -> impl Fn(P) -> T {
        #[allow(clippy::result_unwrap_used)]
        move |x| f(x).unwrap()
    }

    fn nudge(&mut self, pick_movement : i32, depth_movement : i32) -> StackActions {
        let (prologue, spilling, loading, epilogue) = Self::unwrap(|| {
            let spilled_before = self.spilled_count();
    
            self.pick_point = u16::try_from(i32::from(self.pick_point) + pick_movement)
                .or(Err("overflow in pick_point"))?;
            self.stack_depth = u16::try_from(i32::from(self.stack_depth) + depth_movement)
                .or(Err("overflow in stack_depth"))?;

            let spilled_after = self.spilled_count();

            let sp = self.stack_ptr;
            let n = i32::from(spilled_before) - i32::from(spilled_after);
            let (prologue, epilogue) = {
                let update = vec![tenyr_insn!(sp <- sp + (n))?];
                if n < 0        { (update, vec![]) }
                else if n > 0   { (vec![], update) }
                else            { (vec![], vec![]) }
            };
            let reg = |off| self.regs[usize::from(off % self.register_count)];
            let mover = |n : i32, dir| {
                move |offset : u16| {
                    let r = reg(offset);
                    let insn = tenyr_insn!(r <- [sp + (n - i32::from(offset))])?;
                    Ok(Instruction { dd : dir, ..insn })
                }
            };
            let spiller = mover(-n, crate::tenyr::MemoryOpType::StoreRight);
            let loader  = mover( n, crate::tenyr::MemoryOpType::LoadRight);
            let spilling = (spilled_before..spilled_after).map(Self::unwrapper(spiller));
            let loading  = (spilled_after..spilled_before).map(Self::unwrapper(loader));
    
            Ok((prologue, spilling, loading, epilogue))
        });

        std::iter::empty()
            .chain(prologue)
            .chain(spilling)
            .chain(loading)
            .chain(epilogue)
            .collect()
    }

    /// increases pick-point up to a minimum value, if necessary
    fn require_minimum(&mut self, n : u16) -> StackActions {
        self.nudge(0.max((n - self.pick_point).into()), 0)
    }

    /// reserves a given number of slots, pushing the pick point down
    pub fn reserve(&mut self, n : u16) -> StackActions {
        let n = n.into();
        self.nudge(n, n)
    }

    /// releases a given number of slots, pulling the pick point up
    pub fn release(&mut self, n : u16) -> StackActions {
        let n = i32::from(n);
        self.nudge(-n, -n)
    }

    /// commits all registers to memory
    pub fn freeze(&mut self) -> StackActions { self.nudge(-i32::from(self.pick_point), 0) }

    /// liberates all registers from memory
    pub fn thaw(&mut self) -> StackActions { self.nudge(i32::from(self.stack_depth), 0) }

    /// removes all items from the stack
    pub fn empty(&mut self) -> StackActions {
        self.nudge(-i32::from(self.pick_point), -i32::from(self.stack_depth))
    }
}

#[cfg(test)]
#[derive(Copy, Clone, Debug)]
struct NumRegs(u8);

#[cfg(test)]
impl quickcheck::Arbitrary for NumRegs {
    fn arbitrary<G : Gen>(g : &mut G) -> Self {
        #[allow(clippy::result_unwrap_used)]
        NumRegs((g.next_u32() % 14).try_into().unwrap()) // do not count A and P
    }
}

#[cfg(test)]
fn check_invariants(man : &Manager) {
    // Allow "absurd" comparisons to allow us to write runtime assertions that are known to be
    // infallible at compile time *with the the current types*.
    #![allow(clippy::absurd_extreme_comparisons)]
    #![allow(unused_comparisons)]

    assert!(0 <= man.stack_depth);
    assert!(man.stack_depth <= 255);
    assert!(0 <= man.pick_point);
    assert!(man.pick_point <= man.stack_depth);
}

#[cfg(test)]
fn get_mgr(num_regs : NumRegs) -> Manager {
    use Register::*;
    let regs = [B, C, D, E, F, G, H, I, J, K, L, M, N, O];
    let regs = regs.iter().take(num_regs.0.into()).cloned();
    Manager::new(regs)
}

#[cfg(test)]
const POINTER_UPDATE_INSNS : u16 = 1;

#[cfg(test)]
quickcheck! {
    fn test_new(num_regs : NumRegs) -> TestResult {
        if num_regs.0 < 1 { return TestResult::discard(); }
        let man = get_mgr(num_regs);
        check_invariants(&man);
        assert_eq!(man.register_count, (num_regs.0 - 1).into());
        TestResult::passed()
    }

    fn test_trivial_reserve(n : NumRegs) -> TestResult {
        if n.0 < 2 { return TestResult::discard(); }
        let mut man = get_mgr(n);
        let act = man.reserve(1);
        check_invariants(&man);
        assert!(act.is_empty());
        TestResult::passed()
    }

    fn test_small_spill_and_load(num_regs : NumRegs) -> TestResult {
        if num_regs.0 < 3 { return TestResult::discard(); }
        Manager::unwrap(|| {
            let mut man = get_mgr(num_regs);
            let act = man.reserve((num_regs.0 - 1).into());
            assert!(act.is_empty());
            let act = man.reserve(2);
            let sp = man.stack_ptr;
            let top = man.regs[0];
            let sec = man.regs[1];
            let exp : Vec<_> = tenyr_insn_list!(
                sp  <- sp - (2) ;
                top -> [sp + 2] ;
                sec -> [sp + 1] ;
            ).collect();
            assert_eq!(act, exp);
            let act = man.release(2);
            let exp : Vec<_> = tenyr_insn_list!(
                top <- [sp + 2] ;
                sec <- [sp + 1] ;
                sp  <- sp + (2) ;
            ).collect();
            assert_eq!(act, exp);
            Ok(())
        });
        TestResult::passed()
    }

    fn test_boundary(extra : u16, backoff : u16) -> TestResult {
        if backoff > extra { return TestResult::discard(); }

        let mut man = get_mgr(NumRegs(6));
        let r = man.register_count;

        let first = extra - backoff;
        let update_first = if first != 0 { POINTER_UPDATE_INSNS } else { 0 };
        let update_backoff = if backoff != 0 { POINTER_UPDATE_INSNS } else { 0 };
        let act = man.reserve(r + first);
        assert_eq!(act.len(), (first + update_first).into());
        let act = man.reserve(backoff);
        assert_eq!(act.len(), (backoff + update_backoff).into());
        let act = man.release(first);
        assert_eq!(act.len(), (first + update_first).into());
        let act = man.release(r + backoff);
        assert_eq!(act.len(), (backoff + update_backoff).into());

        assert_eq!(man.pick_point, 0);

        TestResult::passed()
    }
}

#[should_panic(expected = "overflow")]
#[test]
fn test_trivial_release() {
    let mut man = get_mgr(NumRegs(6));
    let _ = man.release(1);
    check_invariants(&man);
}
