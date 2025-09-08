use crate::color;
use crate::emu::Emu;
use iced_x86::Instruction;

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

    let a: u128 = (value0 & 0xffffffff) * (value1 & 0xffffffff);
    let b: u128 = (value0 & 0xffffffff00000000) * (value1 & 0xffffffff00000000);
    let c: u128 = (value0 & 0xffffffff0000000000000000) * (value1 & 0xffffffff0000000000000000);
    let d: u128 = (value0 & 0xffffffff000000000000000000000000)
        * (value1 & 0xffffffff000000000000000000000000);

    let result: u128 = a | b | c | d;

    emu.set_operand_xmm_value_128(ins, 0, result);
    true
}
