use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Cyan"), ins);

    let sz0 = emu.get_operand_sz(ins, 0);
    let sz1 = emu.get_operand_sz(ins, 1);

    if sz0 == 128 && sz1 == 128 {
        let value0 = match emu.get_operand_xmm_value_128(ins, 0, true) {
            Some(v) => v,
            None => {
                log::info!("error getting xmm value1");
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

        let mut result = 0u128;
        for i in 0..4 {
            let dword0 = (value0 >> (32 * i)) & 0xFFFFFFFF;
            let dword1 = (value1 >> (32 * i)) & 0xFFFFFFFF;
            let res_dword = dword0.wrapping_sub(dword1);
            result |= res_dword << (32 * i);
        }

        emu.set_operand_xmm_value_128(ins, 0, result);
    } else {
        unimplemented!();
    }
    true
}
