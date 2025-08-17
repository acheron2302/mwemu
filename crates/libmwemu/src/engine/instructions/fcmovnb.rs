use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    if !emu.flags().f_cf {
        emu.fpu_mut().move_reg_to_st0(ins.op_register(1));
    }

    emu.sync_fpu_ip();
    true
}
