use crate::emu;
use crate::structures;

pub fn VirtualQuery(emu: &mut emu::Emu) {
    let addr = emu
        .maps
        .read_dword(emu.regs().get_esp())
        .expect("kernel32!VirtualQuery cannot read addr") as u64;
    let out_buff = emu
        .maps
        .read_dword(emu.regs().get_esp() + 4)
        .expect("kernel32!VirtualQuery cannot read out_buff") as u64;
    let size = emu
        .maps
        .read_dword(emu.regs().get_esp() + 8)
        .expect("kernel32!VirtualQuery cannot read size");

    log_red!(
        emu,
        "kernel32!VirtualQuery 0x{:x} 0x{:x} {}",
        addr,
        out_buff,
        size
    );

    if size < 30 {
        log::info!("buffer to short: {}", size);
        emu.regs_mut().rax = 0;
    } else {
        let mbi = structures::MemoryBasicInformation::guess(addr, &mut emu.maps);
        mbi.save(out_buff, &mut emu.maps);
        emu.regs_mut().rax = 1;
    }

    emu.stack_pop32(false);
    emu.stack_pop32(false);
    emu.stack_pop32(false);
}
