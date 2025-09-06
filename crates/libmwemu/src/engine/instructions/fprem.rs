use crate::color;
use crate::emu::Emu;
use iced_x86::Instruction;

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    let st0 = emu.fpu_mut().get_st(0);
    let st1 = emu.fpu_mut().get_st(1);

    let quotient = (st0 / st1).floor();
    let result = st0 - quotient * st1;

    emu.fpu_mut().set_st(0, result);
    emu.sync_fpu_ip();
    true
}
