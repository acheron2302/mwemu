use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::Mutex;
use lazy_static::lazy_static;

use crate::emu;

pub fn lstrcpy(emu: &mut emu::Emu) {
    let dst = emu.regs().rcx;
    let src = emu.regs().rdx;

    let s = emu.maps.read_string(src);
    emu.maps.write_string(dst, &s);
    emu.maps.write_byte(dst + (s.len() as u64), 0);

    log::info!(
        "{}** {} kernel32!lstrcpy 0x{:x} 0x{:x} {}  {}",
        emu.colors.light_red,
        emu.pos,
        dst,
        src,
        &s,
        emu.colors.nc
    );

    if s.is_empty() {
        emu.regs_mut().rax = 0;
    } else {
        emu.regs_mut().rax = dst;
    }
}