use crate::color;
use crate::emu::Emu;
use iced_x86::Instruction;

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);
    let value0 = emu.get_operand_value(ins, 0, false).unwrap_or(0) as usize;
    let value1 = emu.get_operand_value(ins, 1, false).unwrap_or(0) as usize;

    emu.fpu_mut().subr(value0 as usize, value1 as usize);
    emu.sync_fpu_ip();
    true
}
