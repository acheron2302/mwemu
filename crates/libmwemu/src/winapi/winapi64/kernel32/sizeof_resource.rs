use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::Mutex;
use lazy_static::lazy_static;

use crate::emu;
use crate::winapi::helper;

pub fn SizeofResource(emu: &mut emu::Emu) {
    let hModule = emu.regs().rcx;
    let hResInfo = emu.regs().rdx as u64;

    if helper::handler_exist(hResInfo) {
        let uri = helper::handler_get_uri(hResInfo);
        let size = uri.split("_").last().unwrap().parse::<usize>().unwrap();
        log::info!("** {} kernel32!SizeofResource {:x} {:x} size: {}", emu.pos, hModule, hResInfo, size);
        emu.regs_mut().rax = size as u64;
        return;
    }

    log_red!(emu, "** {} kernel32!SizeofResource {:x} {:x} not found", emu.pos, hModule, hResInfo);

    emu.regs_mut().rax = 0;
}