
use crate::{emu, structures};

pub fn GetStartupInfoW(emu: &mut emu::Emu) {
    let startup_info_ptr = emu.regs().rcx;

    log_red!(
        emu,
        "kernel32!GetStartupInfoW"
    );
    if startup_info_ptr > 0 {
        let startupinfo = structures::StartupInfo64::new();
        startupinfo.save(startup_info_ptr, &mut emu.maps);
    }
}