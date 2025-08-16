use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    let value0 = match emu.get_operand_xmm_value_128(ins, 0, true) {
        Some(v) => v,
        None => {
            log::info!("error getting value0");
            return false;
        }
    };
    let value1 = match emu.get_operand_xmm_value_128(ins, 1, true) {
        Some(v) => v,
        None => {
            log::info!("error getting value1");
            return false;
        }
    };

    let left: u128 = ((value0 & 0xffffffffffffffff0000000000000000) >> 64)
        * ((value1 & 0xffffffffffffffff0000000000000000) >> 64);
    let right: u128 = (value0 & 0xffffffffffffffff) * (value1 & 0xffffffffffffffff);
    let result: u128 = left << 64 | right;

    emu.set_operand_xmm_value_128(ins, 0, result);
    true
}
