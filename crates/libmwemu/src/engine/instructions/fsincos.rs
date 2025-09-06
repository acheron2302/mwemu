use crate::color;
use crate::emu::Emu;
use iced_x86::Instruction;

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    let st0 = emu.fpu_mut().get_st(0);
    let sin_value = st0.sin();
    let cos_value = st0.cos();

    emu.fpu_mut().set_st(0, sin_value);
    emu.fpu_mut().push_f64(cos_value);
    true
}
