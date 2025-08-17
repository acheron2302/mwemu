use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);
    assert!(ins.op_count() == 2);

    let value0 = match emu.get_operand_xmm_value_128(ins, 0, true) {
        Some(v) => v,
        None => {
            log::info!("error getting xmm value0");
            return false;
        }
    };
    let value1 = match emu.get_operand_xmm_value_128(ins, 1, true) {
        Some(v) => v,
        None => {
            log::info!("error getting xmm value1");
            return false;
        }
    };

    let result: u128 = value0 | value1;
    emu.flags_mut().calc_flags(result as u64, 32);

    emu.set_operand_xmm_value_128(ins, 0, result);
    true
}
