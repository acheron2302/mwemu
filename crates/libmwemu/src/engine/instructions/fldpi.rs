use crate::emu::Emu;
use crate::fpu::f80::F80;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    emu.fpu_mut().push_f80(F80::PI());
    emu.sync_fpu_ip();
    true
}
