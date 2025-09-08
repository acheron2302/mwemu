use crate::color;
use crate::emu::Emu;
use iced_x86::Instruction;

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);
    let st0 = emu.fpu_mut().get_st(0);
    let (mantissa, exponent) = emu.fpu().frexp(st0);
    emu.fpu_mut().set_st(0, mantissa);
    emu.fpu_mut().push_f64(exponent as f64);
    emu.sync_fpu_ip();
    true
}
