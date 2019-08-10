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
