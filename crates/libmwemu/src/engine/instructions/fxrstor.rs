use crate::color;
use crate::emu::Emu;
use crate::fpu::FPUState;
use iced_x86::Instruction;

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    let addr = match emu.get_operand_value(ins, 0, false) {
        Some(v) => v,
        None => return false,
    };

    let state = FPUState::load(addr, emu);

    emu.fpu_mut().fxrstor(state);
    emu.sync_fpu_ip();
    true
}
