use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    if emu.rep.is_some() {
        if emu.rep.unwrap() == 0 || emu.cfg.verbose >= 3 {
            emu.show_instruction(color!("LightCyan"), ins);
        }
    } else {
        emu.show_instruction(color!("LightCyan"), ins);
    }

    let value0 = match emu.get_operand_value(ins, 0, true) {
        Some(v) => v,
        None => return false,
    };

    let ax = emu.regs().get_ax();
    emu.flags_mut().sub16(ax, value0);

    if emu.cfg.is_64bits {
        if emu.flags().f_df {
            emu.regs_mut().rdi -= 2;
        } else {
            emu.regs_mut().rdi += 2;
        }
    } else {
        // 32bits
        if emu.flags().f_df {
            let edi = emu.regs().get_edi() - 2;
            emu.regs_mut().set_edi(edi);
        } else {
            let edi = emu.regs().get_edi() + 2;
            emu.regs_mut().set_edi(edi);
        }
    }
    true
}
