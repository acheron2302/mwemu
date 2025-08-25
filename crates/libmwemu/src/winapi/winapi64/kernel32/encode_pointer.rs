
use crate::emu;

pub fn EncodePointer(emu: &mut emu::Emu) {
    let ptr = emu.regs().rcx;

    log_red!(
        emu,
        "kernel32!EncodePointer ptr: 0x{:x}",
        ptr
    );

    emu.regs_mut().rax = ptr;
}