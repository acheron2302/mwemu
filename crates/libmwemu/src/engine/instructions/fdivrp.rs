use crate::color;
use crate::emu::Emu;
use iced_x86::Instruction;

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);
    let value0 = emu.get_operand_value(ins, 0, false).unwrap_or(0) as usize;
    let value1 = emu.get_operand_value(ins, 1, false).unwrap_or(0) as usize;
    let st0 = emu.fpu_mut().get_st(value0);
    let st7 = emu.fpu_mut().get_st(value1);

    let result = st7 / st0;

    emu.fpu_mut().set_st(value1, result);
    emu.fpu_mut().pop_f64();
    emu.sync_fpu_ip();
    true
}
