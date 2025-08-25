use crate::{constants, emu};

pub fn GetCommandLineA(emu: &mut emu::Emu) {
    log_red!(
        emu,
        "kernel32!GetCommandLineA"
    );

    let addr = emu.maps.alloc(1024).expect("out of memory");
    let name = format!("alloc_{:x}", addr);
    emu.maps.create_map(&name, addr, 1024);
    emu.maps.write_string(addr, constants::EXE_NAME);
    emu.regs_mut().rax = addr;
}