
use crate::emu;

pub fn lstrcpyW(emu: &mut emu::Emu) {
    let dst = emu.regs().rcx;
    let src = emu.regs().rdx;

    let s = emu.maps.read_wide_string(src);
    emu.maps.write_wide_string(dst, &s);
    emu.maps.write_byte(dst + (s.len() as u64 * 2), 0);

    log::info!(
        "{}** {} kernel32!lstrcpyW 0x{:x} 0x{:x} {}  {}",
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