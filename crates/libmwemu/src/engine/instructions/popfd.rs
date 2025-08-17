use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Blue"), ins);

    let flags = match emu.stack_pop32(true) {
        Some(v) => v,
        None => return false,
    };
    emu.flags_mut().load(flags);
    true
}
