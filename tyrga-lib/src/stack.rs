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
//! - freeze(N)  -- spill all locations to memory and release N
//! - thaw(N)    -- load registers from memory and reserve N

use crate::tenyr::Instruction;
use crate::tenyr::Register;

use std::convert::TryFrom;
use std::convert::TryInto;

/// a list of stack-maintenance instructions that must be executed
/// Note: the `must_use` attribute here does not appear to be effective on
/// functions that return `StackActions`, so the `must_use` directive is
/// reproduced on multiple functions below.
#[must_use = "StackActions must be implemented to maintain stack discipline"]
pub type StackActions = Vec<Instruction>;

#[derive(Clone)]
pub struct Manager {
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
    pub fn new(regs : &[Register]) -> Self {
        let mut regs = regs.to_owned();
        let stack_depth = 0;
        let pick_point = 0;
        let stack_ptr = regs.pop().expect("too few registers");
        let register_count = regs.len().try_into().expect("too many registers");
        Self {
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
        use crate::tenyr::InstructionType::Type3;
        use crate::tenyr::MemoryOpType::{LoadRight, NoLoad, StoreRight};
        use std::cmp::Ordering::*;

        let spilled_before = self.spilled_count();

        // pick point will never go negative
        self.pick_point = u16::try_from(0.max(i32::from(self.pick_point) + pick_movement))
            .expect("overflow in pick_point");
        self.stack_depth = u16::try_from(i32::from(self.stack_depth) + depth_movement)
            .expect("overflow in stack_depth");

        let spilled_after = self.spilled_count();

        let sp = self.stack_ptr;
        let n = i32::from(spilled_before) - i32::from(spilled_after);
        let reg = |off| self.regs[usize::from(off % self.register_count)];
        let mover = |dd, base| {
            move |offset| {
                let off = (n.abs() - i32::from(offset) + i32::from(base))
                    .try_into()
                    .expect("immediate value is too large");
                Instruction {
                    dd,
                    z : reg(offset),
                    x : sp,
                    kind : Type3(off),
                }
            }
        };
        let off = n.try_into().expect("immediate value is too large");
        let update = std::iter::once(Instruction {
            dd :   NoLoad,
            z :    sp,
            x :    sp,
            kind : Type3(off),
        });

        match n.cmp(&0) {
            Less =>
                update
                    .chain((spilled_before..spilled_after).map(mover(StoreRight, spilled_before)))
                    .collect(),
            Equal =>
                std::iter::empty().collect(),
            Greater =>
                (spilled_after..spilled_before)
                    .map(mover(LoadRight, spilled_after))
                    .chain(update)
                    .collect(),
        }
    }

    /// increases pick-point up to a minimum value, if necessary
    fn require_minimum(&mut self, n : u16) -> StackActions {
        self.nudge(0.max(i32::from(n) - i32::from(self.pick_point)), 0)
    }

    /// reserves a given number of slots, pushing the pick point down
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn reserve(&mut self, n : u16) -> StackActions { self.adjust(i32::from(n)) }

    /// reserves one slot, pushing the pick point down, and returning the top
    /// register for convenience
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn reserve_one(&mut self) -> (Register, StackActions) {
        let act = self.reserve(1);
        let (reg, rest) = self.get(0);
        assert!(rest.is_empty());
        (reg, act)
    }

    /// releases a given number of slots, pulling the pick point up
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn release(&mut self, n : u16) -> StackActions { self.adjust(-i32::from(n)) }

    /// reserves (positive argument) or releases (negative input) a given number
    /// of slots (zero means no operation)
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    fn adjust(&mut self, n : i32) -> StackActions { self.nudge(n, n) }

    /// commits all registers to memory, optionally releasing afterward
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn freeze(&mut self, count : u16) -> StackActions {
        let act = self.nudge(-i32::from(self.pick_point), 0);
        // discarding actions is correct because we are actually accounting for
        // actions that will be taken by an ensuing function call
        let _ = self.nudge(0, -i32::from(count));
        act
    }

    /// liberates all registers from memory, optionally reserving beforehand
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn thaw(&mut self, count : u16) -> StackActions {
        // discarding actions is correct because we are actually accounting for
        // actions that have already been taken prior to this function's entry
        let _ = self.nudge(0, i32::from(count));
        // lazily avoid calling nudge, so that actually thawing is deferred
        // until needed
        vec![]
    }

    /// removes all items from the stack
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn empty(&mut self) -> StackActions {
        let all = self.nudge(-i32::from(self.pick_point), -i32::from(self.stack_depth));
        all.last().into_iter().cloned().collect() // preserve only the stack-update instruction
    }

    /// gets a register at depth from top of stack, panicking if requested depth
    /// is greater than the number of registers that can be alive at once
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn get(&mut self, n : u16) -> (Register, StackActions) {
        let act = self.require_minimum(n + 1); // convert register index into depth
        assert!(n < self.register_count);
        let len : usize = self.register_count.into();
        let n : usize = n.into();
        let deep : usize = self.stack_depth.into();
        let reg = self.regs[(deep - 1 - n) % len];
        (reg, act)
    }

    /// copies a register at depth from top of stack to a newly reserved register
    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn get_copy(&mut self, n : u16) -> (Register, StackActions) {
        assert!(n < self.stack_depth);
        if n < self.pick_point {
            let (from, from_actions) = self.get(n);
            assert!(from_actions.is_empty());
            let to_actions = self.reserve(1);
            let (to, to_actions_addl) = self.get(0);
            assert!(to_actions_addl.is_empty());
            let insn = Instruction {
                z : to,
                x : from,
                ..crate::tenyr::NOOP_TYPE0
            };
            let v = std::iter::empty()
                .chain(to_actions)
                .chain(std::iter::once(insn))
                .collect();
            (to, v)
        } else {
            use crate::tenyr::InstructionType::Type3;
            use crate::tenyr::MemoryOpType::LoadRight;
            let to_actions = self.reserve(1);
            let (to, to_actions_addl) = self.get(0);
            assert!(to_actions_addl.is_empty());
            // add one for reservation above
            // add one since stack pointer points to empty slot
            let offset = n + 1 + 1 - self.pick_point;
            let insn = Instruction {
                z :    to,
                x :    self.get_stack_ptr(),
                dd :   LoadRight,
                kind : Type3(offset.into()),
            };

            let v = std::iter::empty()
                .chain(to_actions)
                .chain(std::iter::once(insn))
                .collect();
            (to, v)
        }
    }

    /// returns the register that points to the highest empty slot in memory
    pub fn get_stack_ptr(&self) -> Register { self.stack_ptr }

    /// returns an Instruction that sets a given register to the address of the
    /// nth element on the operand stack, regardless of the number of spilled
    /// slots
    pub fn get_frame_offset(
        &self,
        reg : crate::tenyr::Register,
        n : i32,
    ) -> crate::tenyr::Instruction {
        let off = i32::from(self.spilled_count()) - n;
        let kind =
            crate::tenyr::InstructionType::Type3(off.try_into().expect("immediate too large"));
        crate::tenyr::Instruction {
            dd : crate::tenyr::MemoryOpType::NoLoad,
            z : reg,
            x : self.stack_ptr,
            kind,
        }
    }
}

#[cfg(test)]
mod test {
    use super::Manager;
    use crate::tenyr::Instruction;

    use quickcheck::{quickcheck, Gen, TestResult};

    #[derive(Copy, Clone, Debug)]
    struct NumRegs(u8);

    impl quickcheck::Arbitrary for NumRegs {
        fn arbitrary<G : Gen>(g : &mut G) -> Self {
            // to be useful, we need a stack pointer and a non-stack pointer
            let min = 2;
            // do not count A and P registers
            let max = 14;
            NumRegs((g.next_u32() as u8) % (max - min) + min) // lossy cast is fine
        }
    }

    fn get_mgr(num_regs : NumRegs) -> Manager {
        use crate::Register::*;
        let regs = [B, C, D, E, F, G, H, I, J, K, L, M, N, O];
        Manager::new(&regs[0..num_regs.0.into()])
    }

    const POINTER_UPDATE_INSNS : u16 = 1;

    fn unwrap<T>(f : impl FnOnce() -> Result<T, Box<dyn std::error::Error>>) -> T {
        #[allow(clippy::unwrap_used)]
        f().unwrap()
    }

    quickcheck! {
        fn test_new(num_regs : NumRegs) -> () {
            let man = get_mgr(num_regs);
            assert_eq!(man.register_count, (num_regs.0 - 1).into());
        }

        fn test_trivial_reserve(n : NumRegs) -> () {
            let mut man = get_mgr(n);
            let act = man.reserve(1);
            assert!(act.is_empty());
        }

        fn test_trivial_release(num_regs : NumRegs) -> TestResult {
            let mut man = get_mgr(num_regs);
            TestResult::must_fail(move || { let _ = man.release(1); })
        }

        fn test_small_spill_and_load(num_regs : NumRegs) -> TestResult {
            if num_regs.0 < 4 { return TestResult::discard(); }
            unwrap(|| {
                let mut man = get_mgr(num_regs);
                let act = man.reserve((num_regs.0 - 1).into());
                assert!(act.is_empty());

                let sp = man.stack_ptr;
                let top = man.regs[0];
                let sec = man.regs[1];
                let thr = man.regs[2];

                let act = man.reserve(3);
                let exp : Vec<_> = tenyr_insn_list!(
                    sp  <-  sp - 3i8    ;
                    top -> [sp + 3i8]   ;
                    sec -> [sp + 2i8]   ;
                    thr -> [sp + 1i8]   ;
                ).collect();
                assert_eq!(act, exp);

                let act = man.release(3);
                let exp : Vec<_> = tenyr_insn_list!(
                    top <- [sp + 3i8]   ;
                    sec <- [sp + 2i8]   ;
                    thr <- [sp + 1i8]   ;
                    sp  <-  sp + 3i8    ;
                ).collect();
                assert_eq!(act, exp);

                Ok(())
            });
            TestResult::passed()
        }

        fn test_boundary(num_regs : NumRegs, extra : u16, backoff : u16) -> TestResult {
            if backoff > extra { return TestResult::discard(); }

            let mut man = get_mgr(num_regs);
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

        fn test_get(num_regs : NumRegs) -> TestResult {
            if num_regs.0 < 3 { return TestResult::discard(); }

            let mut man = get_mgr(num_regs);
            let free_regs : u16 = (num_regs.0 - 1).into();
            let act = man.reserve(free_regs * 2); // ensure we spill
            assert_eq!(act.len(), num_regs.0.into());

            // The expected register is computed using the same logic that is used in
            // the `get` function, so ths is not an independent check, but it does help
            // avoid regressions.
            let len : usize = man.register_count.into();
            let deep : usize = man.stack_depth.into();
            let from_top = |n| (deep - 1 - usize::from(n)) % len;

            let (r, act) = man.get(0);
            let exp = man.regs[from_top(0)];
            assert_eq!(r, exp);
            assert!(act.is_empty());

            let (r, act) = man.get(free_regs - 1);
            let exp = man.regs[from_top(free_regs - 1)];
            assert_eq!(r, exp);
            assert!(act.is_empty());

            TestResult::passed()
        }

        fn test_get_too_deep(num_regs : NumRegs) -> TestResult {
            if num_regs.0 < 3 { return TestResult::discard(); }

            let mut man = get_mgr(num_regs);
            let free_regs : u16 = (num_regs.0 - 1).into();
            let act = man.reserve(free_regs * 2); // ensure we spill
            assert_eq!(act.len(), num_regs.0.into());

            TestResult::must_fail(move || { let _ = man.get(free_regs); })
        }

        fn test_pick_underflow(num_regs : NumRegs, n : u16) -> TestResult {
            if u16::from(num_regs.0 + 1) < n { return TestResult::discard(); }

            let mut man = get_mgr(num_regs);
            let _ = man.reserve(n);
            let _ = man.nudge(-i32::from(n + 3), 0); // force underflow
            assert_eq!(man.pick_point, 0);
            TestResult::passed()
        }

        fn test_repeated_spills(num_regs : NumRegs) -> () {
            use crate::tenyr::InstructionType::Type3;

            unwrap(|| {
                let n : u16 = num_regs.0.into();
                let mut man = get_mgr(num_regs);
                let sp = man.stack_ptr;

                let _  = man.reserve(n - 1);
                let act = man.reserve(1);
                assert_eq!(act.len(), 2);
                assert_eq!(act[0], tenyr_insn!( sp <- sp - 1i8 ));
                assert_eq!(act[1].kind, Type3(1_i16.into()));

                let act = man.reserve(1);
                assert_eq!(act.len(), 2);
                assert_eq!(act[0], tenyr_insn!( sp <- sp - 1i8 ));
                assert_eq!(act[1].kind, Type3(1_i16.into()));

                let act = man.reserve(1);
                assert_eq!(act.len(), 2);
                assert_eq!(act[0], tenyr_insn!( sp <- sp - 1i8 ));
                assert_eq!(act[1].kind, Type3(1_i16.into()));

                Ok(())
            });
        }

        fn test_get_copy(num_regs : NumRegs, off : i8) -> TestResult {
            use crate::tenyr::MemoryOpType::{LoadRight, NoLoad};
            if num_regs.0 < 3 { return TestResult::discard(); }

            let mut man = get_mgr(num_regs);
            let half : u16 = (num_regs.0 / 2).into();
            let diff = i32::from(half) - i32::from(off);
            if diff <= 0 || diff >= num_regs.0.into() { return TestResult::discard(); }

            let _ = man.reserve(half);
            let _ = man.nudge(-(i32::from(off)), 0);
            let before = man.stack_depth;
            let (reg, act) = man.get_copy(half - 1);
            let after = man.stack_depth;
            assert_eq!(before + 1, after);
            assert_eq!(act.len(), 1);
            assert_eq!(man.regs[usize::from(half)], reg);
            if off <= 0 {
                assert_eq!(act[0].dd, NoLoad);
                assert_eq!(act[0].kind, crate::tenyr::NOOP_TYPE0.kind);
            } else {
                assert_eq!(act[0].dd, LoadRight);
                assert_eq!(act[0].kind, crate::tenyr::InstructionType::Type3(off.into()));
            }

            TestResult::passed()
        }
    }
}
