use crate::emu;
use crate::maps::mem64::Permission;

pub fn HeapAlloc(emu: &mut emu::Emu) {
    let hndl = emu.regs().rcx;
    let flags = emu.regs().rdx;
    let size = emu.regs().r8;

    let heap_addr: u64 = if size < 0x8000 {
        let heap_manage = emu.heap_management.as_mut().unwrap();
        heap_manage.allocate(size as usize).expect("failed to allocate heap")
    } else {
        let allocation = emu.maps.alloc(size).unwrap_or_default();
        emu.maps
            .create_map(
                format!("alloc_{:x}", emu.regs().rax).as_str(),
                emu.regs().rax,
                size,
                Permission::READ_WRITE,
            )
            .expect("kernel32!HeapAlloc out of memory");
        allocation
    };

    emu.regs_mut().rax = heap_addr;

    log_red!(
        emu,
        "kernel32!HeapAlloc rip: 0x{:x} flags: 0x{:x} size: {} =0x{:x}",
        emu.regs().rip,
        flags,
        size,
        emu.regs().rax
    );
}
