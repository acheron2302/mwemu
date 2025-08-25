use crate::emu;
use crate::constants;

pub fn GetComputerNameA(emu: &mut emu::Emu) {
    let buff_ptr = emu
        .maps
        .read_dword(emu.regs().get_esp())
        .expect("kernel32!GetComputerNameA cannot read buff param") as u64;
    let size_ptr = emu
        .maps
        .read_dword(emu.regs().get_esp() + 4)
        .expect("kernel32!GetComputerNameA cannot read size param") as u64;

    if buff_ptr > 0 {
        emu.maps.write_string(buff_ptr, constants::HOST_NAME);
        emu.regs_mut().rax = 1;
    } else {
        emu.regs_mut().rax = 0;
    }

    if size_ptr > 0 {
        emu.maps.write_dword(size_ptr, 6);
        emu.regs_mut().rax = 1;
    }

    log_red!(
        emu,
        "kernel32!GetComputerName '{}'",
        constants::HOST_NAME
    );

    emu.stack_pop32(false);
    emu.stack_pop32(false);
}