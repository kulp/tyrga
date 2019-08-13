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
//! - reserve(N) -- allocate N locations at the top of stack
//! - release(N) -- deallocate N locations from the top of stack
//! - empty()    -- release all contents of the stack
//! - get(N)     -- get location for offset N >= 0, down from top of stack
//! - freeze()   -- spill all locations to memory
//! - thaw()     -- load as many locations into registers as will fit

use crate::tenyr::Register;

use std::convert::TryFrom;

struct Manager {
    /// registers under our control
    regs : Vec<Register>,
    /// number of items on operand stack
    stack_depth : u16,
    /// number of "live" registers desired
    pick_point : u16,
}

impl Manager {
    /// number of registers usable
    fn register_count(&self) -> u16 { u16::try_from(self.regs.len()).expect("too many registers") }
    /// number of values stored in memory
    fn spilled_count(&self) -> u16 { 0.max(self.pick_point - self.register_count()) }

    fn check_invariants(&self) {
        // Allow "absurd" comparisons to allow us to write runtime assertions that are known to be
        // infallible at compile time *with the the current types*.
        #![allow(clippy::absurd_extreme_comparisons)]
        #![allow(unused_comparisons)]

        assert!(0 <= self.stack_depth);
        assert!(self.stack_depth <= 255);
        assert!(0 <= self.pick_point);
        assert!(self.pick_point <= self.stack_depth);
    }
}

