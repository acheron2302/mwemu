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
    emu.pos += 1;

    assert!(emu.cfg.is_64bits);

    let val = emu
        .maps
        .read_qword(emu.regs().rsi)
        .expect("cannot read memory");
    emu.maps.write_qword(emu.regs().rdi, val);

    if !emu.flags().f_df {
        emu.regs_mut().rsi += 8;
        emu.regs_mut().rdi += 8;
    } else {
        emu.regs_mut().rsi -= 8;
        emu.regs_mut().rdi -= 8;
    }
    true
}
