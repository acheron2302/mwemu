
use crate::emu;
use crate::maps::mem64::Permission;

const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READONLY: u32 = 0x02;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_WRITECOPY: u32 = 0x08;
const PAGE_EXECUTE: u32 = 0x10;
const PAGE_EXECUTE_READ: u32 = 0x20;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;
const PAGE_EXECUTE_WRITECOPY: u32 = 0x80;
const PAGE_GUARD: u32 = 0x100;
const PAGE_NOCACHE: u32 = 0x200;

pub fn VirtualAllocEx(emu: &mut emu::Emu) {
    let proc_hndl = emu.regs().rcx;
    let addr = emu.regs().rdx;
    let size = emu.regs().r8;
    let alloc_type = emu.regs().r9;
    let protect = emu
        .maps
        .read_dword(emu.regs().rsp + 0x20)
        .expect("kernel32!VirtualAllocEx cannot read_dword protect");

    let can_read = (protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
        PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
        PAGE_EXECUTE_WRITECOPY)) != 0;

    let can_write = (protect & (PAGE_READWRITE | PAGE_WRITECOPY |
        PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;

    let can_execute = (protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ |
        PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;


    let base = emu
        .maps
        .alloc(size)
        .expect("kernel32!VirtualAllocEx out of memory");

    log::info!(
        "{}** {} kernel32!VirtualAllocEx hproc: 0x{:x} addr: 0x{:x} sz: {} = 0x{:x} {}",
        emu.colors.light_red,
        emu.pos,
        proc_hndl,
        addr,
        size,
        base,
        emu.colors.nc
    );

    emu.maps
        .create_map(format!("alloc_{:x}", base).as_str(), base, size, Permission::from_flags(can_read, can_write, can_execute))
        .expect("kernel32!VirtualAllocEx out of memory");

    emu.regs_mut().rax = base;
}