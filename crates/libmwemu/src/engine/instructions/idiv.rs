use crate::emu::Emu;
use crate::engine::logic;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Cyan"), ins);

    assert!(ins.op_count() == 1);

    let value0 = match emu.get_operand_value(ins, 0, true) {
        Some(v) => v,
        None => return false,
    };

    let pre_rax = emu.regs().rax;
    let pre_rdx = emu.regs().rdx;

    let sz = emu.get_operand_sz(ins, 0);
    match sz {
        64 => logic::idiv64(emu, value0),
        32 => logic::idiv32(emu, value0),
        16 => logic::idiv16(emu, value0),
        8 => logic::idiv8(emu, value0),
        _ => unimplemented!("wrong size"),
    }
    true
}
