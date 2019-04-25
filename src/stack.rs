use crate::tenyr;
use tenyr::Register;

#[derive(Clone, Debug)]
pub struct StackManager {
    stack_ptr : Register,
    frame_ptr : Register,
    stack : Vec<Register>,
    count : usize,
    frozen : usize,
}

type StackActions = Vec<tenyr::Instruction>;

// This simple StackManager implementation does not do spilling to nor reloading from memory.
// For now, it panics if we run out of free registers.
impl StackManager {
    pub fn new(sp : Register, fp : Register, r : Vec<Register>) -> StackManager {
        StackManager { count : 0, frozen : 0, stack_ptr : sp, frame_ptr : fp, stack : r }
    }

    pub fn get_stack_ptr(&self) -> Register { self.stack_ptr }
    pub fn get_frame_ptr(&self) -> Register { self.frame_ptr }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn reserve(&mut self, n : usize) -> StackActions {
        assert!(self.count + n <= self.stack.len(), "operand stack overflow");
        self.count += n;
        vec![] // TODO support spilling
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn release(&mut self, n : usize) -> StackActions {
        assert!(self.count >= n, "operand stack underflow");
        self.count -= n;
        vec![] // TODO support reloading
    }

    pub fn depth(&self) -> usize { self.count }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn empty(&mut self) -> StackActions {
        let mut v = self.release(self.count);
        v.extend(self.thaw()); // capture instruction moving stack pointer, if any
        v
    }

    fn get_reg(&self, which : usize) -> Register {
        assert!(which <= self.count, "attempt to access nonexistent depth");
        // indexing is relative to top of stack, counting backward
        self.stack[(self.count - which - 1) % self.stack.len()]
    }

    pub fn get(&self, which : usize) -> Option<tenyr::Register> {
        // TODO handle Stacked
        assert!(which <= self.stack.len(), "attempt to access register deeper than register depth");
        // indexing is relative to top of stack, counting backward
        self.get_reg(which).into()
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    fn set_watermark(&mut self, level : usize) -> StackActions {
        // TODO remove casts

        // `self.frozen` counts the number of spilled registers (from bottom of operand stack) and
        // takes values in the interval [ 0, +infinity )
        // `level` requests a number of free registers (from top of operand stack) and takes values
        // in the interval [ 0, min(self.stack.len(), self.count) )
        // `unfrozen` derives the current watermark in the same units as `level`
        let level = std::cmp::min(self.count, level) as i32; // cannot freeze more than we have
        let count = self.count as i32;
        let frozen = self.frozen as i32;
        let unfrozen = count - frozen;

        use tenyr::*;
        use tenyr::InstructionType::*;
        use tenyr::MemoryOpType::*;
        let stack_ptr = self.stack_ptr;

        let stack_movement = -(unfrozen as i32 - level as i32) as i32; // TODO check overflow issues here
        if stack_movement == 0 {
            return vec![];
        }

        let new_frozen = frozen as i32 - stack_movement;
        self.frozen = new_frozen as usize;

        let make_insn = |reg, offset| Instruction { dd : NoLoad, kind : Type3(Immediate20::new(offset).unwrap()), z : reg, x : stack_ptr };
        let make_move = |i, offset| make_insn(self.get_reg(i as usize), i + offset + 1);
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
    pub fn freeze(&mut self) -> StackActions {
        self.set_watermark(0)
    }

    #[must_use = "StackActions must be implemented to maintain stack discipline"]
    pub fn thaw(&mut self) -> StackActions {
        self.set_watermark(self.stack.len())
    }
}

#[test]
fn test_get_reg() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(O, N, v.clone());
    let _ = sm.reserve(v.len());
    assert_eq!(&v[0], &sm.get_reg(v.len() - 1));
}


#[test]
#[should_panic(expected="underflow")]
fn test_underflow() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(O, N, v);
    let _ = sm.reserve(3);
    let _ = sm.release(4);
}

#[test]
#[should_panic(expected="overflow")]
fn test_overflow() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let len = v.len();
    let mut sm = StackManager::new(O, N, v);
    let _ = sm.reserve(len + 1);
}

#[test]
fn test_normal_stack() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let t = v.clone();
    let mut sm = StackManager::new(O, N, v);
    let off = 3;
    let _ = sm.reserve(off);
    assert_eq!(sm.get(0), t[off - 1].into());
}

#[test]
fn test_watermark() {
    use Register::*;
    let v = vec![ C, D, E, F, G ];
    let mut sm = StackManager::new(O, N, v);
    let _ = sm.reserve(4);

    let insns = sm.set_watermark(0);
    assert!(insns.len() == 5);
    let insns = sm.set_watermark(0);
    assert!(insns.len() == 0);

    let insns = sm.set_watermark(3);
    assert!(insns.len() == 4);
    let insns = sm.set_watermark(3);
    assert!(insns.len() == 0);

    let insns = sm.set_watermark(1);
    assert!(insns.len() == 3);
    let insns = sm.set_watermark(1);
    assert!(insns.len() == 0);
}

