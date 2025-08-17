use crate::emu;

pub fn Thread32First(emu: &mut emu::Emu) {
    let hndl = emu
        .maps
        .read_dword(emu.regs().get_esp())
        .expect("kernel32!Thread32First cannot read the handle");
    let entry = emu
        .maps
        .read_dword(emu.regs().get_esp() + 4)
        .expect("kernel32!Thread32First cannot read the entry32");

    log::info!(
        "{}** {} kernel32!Thread32First {}",
        emu.colors.light_red,
        emu.pos,
        emu.colors.nc
    );

    emu.stack_pop32(false);
    emu.stack_pop32(false);

    emu.regs_mut().rax = 1;
}