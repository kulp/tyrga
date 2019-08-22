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

use crate::tenyr;
use crate::tenyr::Instruction;
use crate::tenyr::Register;

use std::convert::TryFrom;
use std::convert::TryInto;

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

    fn nudge(&mut self, pick_movement : i32, depth_movement : i32) -> StackActions {
        let spilled_before = self.spilled_count();

        self.pick_point = u16::try_from(i32::from(self.pick_point) + pick_movement)
            .expect("overflow in pick_point");
        self.stack_depth = u16::try_from(i32::from(self.stack_depth) + depth_movement)
            .expect("overflow in stack_depth");

        let spilled_after = self.spilled_count();

        // TODO implement real behaviors
        let spilling = (spilled_before..spilled_after).map(|_| tenyr::NOOP_TYPE0);
        let loading = (spilled_after..spilled_before).map(|_| tenyr::NOOP_TYPE0);

        spilling.chain(loading).collect()
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
        let n = n.into();
        self.nudge(n, -n)
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

#[test]
fn test_new() {
    use Register::*;
    let man = Manager::new(vec![B, C, D]);
    check_invariants(&man);
    assert_eq!(man.register_count, 2);
}

#[cfg(test)]
fn get_mgr() -> Manager {
    use Register::*;
    Manager::new(vec![B, C, D, E, F, G])
}

#[test]
fn test_trivial_reserve() {
    let mut man = get_mgr();
    let act = man.reserve(1);
    check_invariants(&man);
    assert!(act.is_empty());
}

#[should_panic(expected = "overflow")]
#[test]
fn test_trivial_release() {
    let mut man = get_mgr();
    let _ = man.release(1);
    check_invariants(&man);
}

#[test]
fn test_boundary() {
    let mut man = get_mgr();
    let r = man.register_count;

    let act = man.reserve(r);
    assert!(act.is_empty());
    let act = man.release(r);
    assert!(act.is_empty());

    let act = man.reserve(r + 1);
    assert_eq!(act.len(), 1);
    let act = man.release(1);
    assert_eq!(act.len(), 1);
    let act = man.release(r);
    assert!(act.is_empty());
}
