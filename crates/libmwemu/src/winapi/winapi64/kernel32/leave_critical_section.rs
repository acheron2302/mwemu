use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::Mutex;
use lazy_static::lazy_static;

use crate::emu;
use crate::winapi::helper;

pub fn LeaveCriticalSection(emu: &mut emu::Emu) {
    let cs_ptr = emu.regs().rcx;
    let tid = emu.current_thread().id;

    log::info!(
        "{}** {} kernel32!LeaveCriticalSection thread: 0x{:x} cs: 0x{:x} {}",
        emu.colors.light_red,
        emu.pos,
        tid,
        cs_ptr,
        emu.colors.nc
    );

    // Release the lock and get any thread that should be woken up
    if let Some(wake_tid) = emu.global_locks.leave(cs_ptr, tid) {
        // Find the thread to wake up and clear its blocked state
        for thread in &mut emu.threads {
            if thread.id == wake_tid {
                thread.blocked_on_cs = None;
                thread.wake_tick = 0; // Make it runnable immediately
                log::info!("  Waking thread 0x{:x}", wake_tid);
                break;
            }
        }
    }

    // Small delay to simulate atomic operation overhead
    helper::advance_tick(emu, 1);

    emu.regs_mut().rax = cs_ptr;
}