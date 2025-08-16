use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Blue"), ins);

    // 32bits only instruction
    let flags = emu.flags().dump();
    if !emu.stack_push32(flags) {
        return false;
    }
    true
}
