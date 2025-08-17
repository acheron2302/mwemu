use crate::emu::Emu;
use crate::console::Console;
use crate::{color};
use iced_x86::{Instruction};

pub fn execute(emu: &mut Emu, ins: &Instruction, instruction_sz: usize, _rep_step: bool) -> bool {
    emu.show_instruction(color!("Orange"), ins);

    assert!(ins.op_count() == 2);

    if emu.break_on_next_cmp {
        Console::spawn_console(emu);
        emu.break_on_next_cmp = false;
    }

    let value0 = match emu.get_operand_value(ins, 0, true) {
        Some(v) => v,
        None => return false,
    };

    let value1 = match emu.get_operand_value(ins, 1, true) {
        Some(v) => v,
        None => return false,
    };

    let sz = emu.get_operand_sz(ins, 0);

    emu.flags_mut().test(value0, value1, sz);
    true
}
