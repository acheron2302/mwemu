use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Green"), ins);

    assert!(ins.op_count() == 1 || ins.op_count() == 2);

    let value0 = match emu.get_operand_value(ins, 0, true) {
        Some(v) => v,
        None => return false,
    };

    if ins.op_count() == 1 {
        // 1 param

        let sz = emu.get_operand_sz(ins, 0);
        let result = match sz {
            64 => emu.flags_mut().sar1p64(value0),
            32 => emu.flags_mut().sar1p32(value0),
            16 => emu.flags_mut().sar1p16(value0),
            8 => emu.flags_mut().sar1p8(value0),
            _ => panic!("weird size"),
        };

        if !emu.set_operand_value(ins, 0, result) {
            return false;
        }
    } else {
        // 2 params

        let value1 = match emu.get_operand_value(ins, 1, true) {
            Some(v) => v,
            None => return false,
        };

        let sz = emu.get_operand_sz(ins, 0);
        let result = match sz {
            64 => emu.flags_mut().sar2p64(value0, value1),
            32 => emu.flags_mut().sar2p32(value0, value1),
            16 => emu.flags_mut().sar2p16(value0, value1),
            8 => emu.flags_mut().sar2p8(value0, value1),
            _ => panic!("weird size"),
        };

        if !emu.set_operand_value(ins, 0, result) {
            return false;
        }
    }
    true
}
