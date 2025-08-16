use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);
    let st0 = emu.fpu_mut().get_st(0);
    emu.fpu_mut().set_st(0, st0.sin());
    emu.sync_fpu_ip();
    true
}
