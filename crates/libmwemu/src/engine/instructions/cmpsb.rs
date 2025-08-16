use crate::emu::Emu;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    let value0: u8;
    let value1: u8;

    if emu.rep.is_some() {
        if emu.rep.unwrap() == 0 || emu.cfg.verbose >= 3 {
            emu.show_instruction(color!("LightCyan"), ins);
        }
    } else {
        emu.show_instruction(color!("LightCyan"), ins);
    }

    if emu.cfg.is_64bits {
        value0 = match emu.maps.read_byte(emu.regs().rsi) {
            Some(v) => v,
            None => {
                log::info!("cannot read rsi");
                return false;
            }
        };
        value1 = match emu.maps.read_byte(emu.regs().rdi) {
            Some(v) => v,
            None => {
                log::info!("cannot read rdi");
                return false;
            }
        };

        if emu.flags().f_df {
            emu.regs_mut().rsi -= 1;
            emu.regs_mut().rdi -= 1;
        } else {
            emu.regs_mut().rsi += 1;
            emu.regs_mut().rdi += 1;
        }
    } else {
        // 32bits
        value0 = match emu.maps.read_byte(emu.regs().get_esi()) {
            Some(v) => v,
            None => {
                log::info!("cannot read esi");
                return false;
            }
        };
        value1 = match emu.maps.read_byte(emu.regs().get_edi()) {
            Some(v) => v,
            None => {
                log::info!("cannot read edi");
                return false;
            }
        };

        if emu.flags().f_df {
            let esi = emu.regs().get_esi() - 1;
            let edi = emu.regs().get_edi() - 1;
            emu.regs_mut().set_esi(esi);
            emu.regs_mut().set_edi(edi);
        } else {
            let esi = emu.regs().get_esi() + 1;
            let edi = emu.regs().get_edi() + 1;
            emu.regs_mut().set_esi(esi);
            emu.regs_mut().set_edi(edi);
        }
    } // end 32bits

    emu.flags_mut().sub8(value0 as u64, value1 as u64);

    if emu.cfg.verbose >= 2 {
        if value0 > value1 {
            log::info!("\tcmp: 0x{:x} > 0x{:x}", value0, value1);
        } else if value0 < value1 {
            log::info!("\tcmp: 0x{:x} < 0x{:x}", value0, value1);
        } else {
            log::info!("\tcmp: 0x{:x} == 0x{:x}", value0, value1);
        }
    }
    true
}
