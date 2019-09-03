use crate::tenyr;
use std::convert::TryInto;
use tenyr::Register;

#[cfg(test)]
use quickcheck::{quickcheck, Gen, TestResult};

#[derive(Clone, Debug)]
pub struct Manager {
    stack_ptr : Register,
    stack : Vec<Register>,
    // stack sizes are inherently constrained by JVM to be 16-bit
    count : u16,
    frozen : u16,
}

type StackActions = Vec<tenyr::Instruction>;

// This simple Manager implementation does not do spilling to nor reloading from memory.
// For now, it panics if we run out of free registers.
impl Manager {
    pub fn new(sp : Register, regs : impl IntoIterator<Item=Register>) -> Self {
        Self {
            count : 0,
            frozen : 0,
            stack_ptr : sp,
            stack : regs.into_iter().collect()
        }
    }

    pub fn get_frame_offset(&self, reg : tenyr::Register, n : i32) -> tenyr::Instruction {
        let off = i32::from(self.frozen) - n;
        #[allow(clippy::result_unwrap_used)]
        let kind = tenyr::InstructionType::Type3(off.try_into().unwrap());
        tenyr::Instruction {
            dd : tenyr::MemoryOpType::NoLoad,
            z : reg,
            x : self.stack_ptr,
            kind,
        }
    }

    pub fn get_stack_ptr(&self) -> Register { self.stack_ptr }

    // Sometimes we need to accommodate actions by external agents upon our frozen stack
    pub fn adjust(&mut self, n : i32) {
        self.count = (i32::from(self.count) + n) as u16;
        self.frozen = (i32::from(self.frozen) + n) as u16;
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn reserve(&mut self, n : u16) -> StackActions {
        assert!(
            (self.count + n) as usize <= self.stack.len(),
            "operand stack overflow"
        );
        self.count += n;
        vec![] // TODO support spilling
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn release(&mut self, n : u16) -> StackActions {
        assert!(self.count >= n, "operand stack underflow");
        self.count -= n;
        vec![] // TODO support reloading
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn empty(&mut self) -> StackActions {
        let mut v = self.release(self.count);
        v.extend(self.thaw()); // capture instruction moving stack pointer, if any
        v
    }

    fn get_reg(&self, which : u16) -> Register {
        assert!(which <= self.count, "attempt to access nonexistent depth");
        // indexing is relative to top of stack, counting backward
        self.stack[usize::from(self.count - which - 1) % self.stack.len()]
    }

    pub fn get(&self, which : u16) -> (tenyr::Register, StackActions) {
        assert!(
            which as usize <= self.stack.len(),
            "attempt to access register deeper than register depth"
        );
        // indexing is relative to top of stack, counting backward
        (self.get_reg(which), vec![]) // empty StackActions during migration
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    fn set_watermark(&mut self, level : u16) -> StackActions {
        use tenyr::InstructionType::*;
        use tenyr::MemoryOpType::*;
        use tenyr::*;

        // TODO remove casts

        // `self.frozen` counts the number of spilled registers (from bottom of operand stack) and
        // takes values in the interval [ 0, +infinity )
        // `level` requests a number of free registers (from top of operand stack) and takes values
        // in the interval [ 0, min(self.stack.len(), self.count) )
        // `unfrozen` derives the current watermark in the same units as `level`
        let level = i32::from(std::cmp::min(self.count, level)); // cannot freeze more than we have
        let count = i32::from(self.count);
        let frozen = i32::from(self.frozen);
        let unfrozen = count - frozen;

        let stack_ptr = self.stack_ptr;

        let stack_movement = -(unfrozen as i32 - level as i32) as i32; // TODO check overflow issues here
        if stack_movement == 0 {
            return vec![];
        }

        let new_frozen = frozen as i32 - stack_movement;
        self.frozen = new_frozen as u16;

        #[allow(clippy::result_unwrap_used)]
        let make_insn =
            |reg, offset : i32| Instruction {
                dd : NoLoad,
                kind : Type3(offset.try_into().unwrap()),
                z : reg,
                x : stack_ptr
            };
        let make_move = |i, offset| make_insn(self.get_reg(i as u16), i + offset + 1);
        // Only one of { `freezing`, `thawing` } will have any elements in it
        let freezing = (level..unfrozen).map(|i| Instruction { dd : StoreRight, ..make_move(i, 0) });
        let thawing  = (unfrozen..level).map(|i| Instruction { dd : LoadRight , ..make_move(i, -stack_movement) });
        let update = make_insn(stack_ptr, stack_movement);

        std::iter::once(update)
            .chain(freezing)
            .chain(thawing)
            .collect()
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn freeze(&mut self) -> StackActions { self.set_watermark(0) }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn thaw(&mut self) -> StackActions { self.set_watermark(self.stack.len() as u16) }
}

#[cfg(test)]
impl quickcheck::Arbitrary for tenyr::Register {
    fn arbitrary<G : Gen>(g : &mut G) -> Self {
        // TODO use Range once iterating on it works
        use Register::*;
        [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P][(g.next_u32() % 16) as usize]
    }
}

#[cfg(test)]
quickcheck! {
fn test_get_reg(v : Vec<Register>) -> TestResult {
    if v.is_empty() {
        return TestResult::discard();
    }
    let mut sm = Manager::new(Register::O, v.clone());
    let _ = sm.reserve(v.len() as u16);
    TestResult::from_bool(v[0] == sm.get_reg(v.len() as u16 - 1))
}
}

#[test]
#[should_panic(expected = "underflow")]
fn test_underflow() {
    use Register::*;
    let v = vec![C, D, E, F, G];
    let mut sm = Manager::new(O, v);
    let _ = sm.reserve(3);
    let _ = sm.release(4);
}

#[test]
#[should_panic(expected = "overflow")]
fn test_overflow() {
    use Register::*;
    let v = vec![C, D, E, F, G];
    let len = v.len() as u16;
    let mut sm = Manager::new(O, v);
    let _ = sm.reserve(len + 1);
}

#[test]
fn test_normal_stack() {
    use Register::*;
    let v = vec![C, D, E, F, G];
    let t = v.clone();
    let mut sm = Manager::new(O, v);
    let off = 3;
    let _ = sm.reserve(off as u16);
    assert_eq!(sm.get(0).0, t[off - 1]);
}

#[test]
fn test_watermark() {
    use Register::*;
    let v = vec![C, D, E, F, G];
    let mut sm = Manager::new(O, v);
    let mut insns = sm.reserve(4);

    insns.extend(sm.set_watermark(0));
    assert_eq!(insns.len(), 5);
    let insns = sm.set_watermark(0);
    assert!(insns.is_empty());

    let insns = sm.set_watermark(3);
    assert_eq!(insns.len(), 4);
    let insns = sm.set_watermark(3);
    assert!(insns.is_empty());

    let insns = sm.set_watermark(1);
    assert_eq!(insns.len(), 3);
    let insns = sm.set_watermark(1);
    assert!(insns.is_empty());
}
