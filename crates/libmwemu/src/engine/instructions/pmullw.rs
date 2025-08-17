use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    let source0 = emu.get_operand_xmm_value_128(ins, 0, true).unwrap_or(0);
    let source1 = emu.get_operand_xmm_value_128(ins, 1, true).unwrap_or(0);
    let mut result = 0u128;

    for i in 0..8 {
        let mask = 0xFFFFu128;
        let shift = i * 16;
        let word0 = ((source0 >> shift) & mask) as u16;
        let word1 = ((source1 >> shift) & mask) as u16;
        let product = word0.wrapping_mul(word1) as u128;
        result |= (product & mask) << shift;
    }

    emu.set_operand_xmm_value_128(ins, 0, result);
    true
}
