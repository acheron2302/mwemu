use crate::emu;
use crate::structures;

pub fn GetStartupInfoA(emu: &mut emu::Emu) {
    let startup_info_ptr =
        emu.maps
            .read_dword(emu.regs().get_esp())
            .expect("kernel32!GetStartupInfoA cannot read startup_info_ptr param") as u64;

    log::info!(
        "{}** {} kernel32!GetStartupInfoA {}",
        emu.colors.light_red,
        emu.pos,
        emu.colors.nc
    );
    if startup_info_ptr > 0 {
        let startupinfo = structures::StartupInfo32::new();
        startupinfo.save(startup_info_ptr, &mut emu.maps);
    }

    emu.stack_pop32(false);
}