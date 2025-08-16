use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    emu.fpu_mut().subr(1, 0);
    emu.fpu_mut().pop_f64();
    emu.sync_fpu_ip();
    true
}
