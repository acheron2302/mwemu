use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    let value: u64 = if emu.cfg.is_64bits {
        match emu.stack_pop64(true) {
            Some(v) => v,
            None => return false,
        }
    } else {
        match emu.stack_pop32(true) {
            Some(v) => v as u64,
            None => return false,
        }
    };

    emu.show_instruction_pushpop(color!("Blue"), ins, value);

    if !emu.set_operand_value(ins, 0, value) {
        return false;
    }
    true
}
