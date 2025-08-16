use crate::emu;
use crate::constants;

pub fn RegSetValueExA(emu: &mut emu::Emu) {
    let hKey = emu
        .maps
        .read_dword(emu.regs().get_esp())
        .expect("kernel32!RegSetValueExA: error reading param") as u64;
    let value_name_ptr = emu
        .maps
        .read_dword(emu.regs().get_esp() + 4)
        .expect("kernel32!RegSetValueExA: error reading param") as u64;
    let reserved = emu
        .maps
        .read_dword(emu.regs().get_esp() + 8)
        .expect("kernel32!RegSetValueExA: error reading param") as u64;
    let value_type = emu
        .maps
        .read_dword(emu.regs().get_esp() + 12)
        .expect("kernel32!RegSetValueExA: error reading param") as u64;
    let data_ptr = emu
        .maps
        .read_dword(emu.regs().get_esp() + 16)
        .expect("kernel32!RegSetValueExA: error reading param") as u64;
    let data_size = emu
        .maps
        .read_dword(emu.regs().get_esp() + 20)
        .expect("kernel32!RegSetValueExA: error reading param") as u64;

    let value_name = emu.maps.read_string(value_name_ptr);

    log::info!(
        "{}** {} kernel32!RegSetValueExA `{}` type: {} data: 0x{:x} {}",
        emu.colors.light_red,
        emu.pos,
        value_name,
        value_type,
        data_ptr,
        emu.colors.nc
    );
    emu.regs_mut().rax = constants::ERROR_SUCCESS;

    for _ in 0..6 {
        emu.stack_pop32(false);
    }
}