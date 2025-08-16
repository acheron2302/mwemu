use crate::emu;
use crate::thread_context::ThreadContext;
use crate::winapi::helper;

pub fn CreateRemoteThread(emu: &mut emu::Emu) {
    let proc_hndl = emu.regs().rcx;
    let sec = emu.regs().rdx;
    let stack_size = emu.regs().r8;
    let addr = emu.regs().r9;
    let param = emu
        .maps
        .read_qword(emu.regs().rsp + 0x20)
        .expect("krenel32!CreateRemoteThread cannot read the param");
    let flags = emu
        .maps
        .read_qword(emu.regs().rsp + 0x28)
        .expect("kernel32!CreateRemoteThread cannot read the flags");
    let out_tid = emu
        .maps
        .read_qword(emu.regs().rsp + 0x30)
        .expect("kernel32!CreateRemoteThread cannot read the tid");

    log::info!(
        "{}** {} kernel32!CreateRemoteThread hproc: 0x{:x} addr: 0x{:x} {}",
        emu.colors.light_red,
        emu.pos,
        proc_hndl,
        addr,
        emu.colors.nc
    );
    let new_thread_id = 0x1000 + emu.threads.len();
    let mut new_thread = ThreadContext::new(new_thread_id as u64);
    
    // Initialize thread context with entry point and parameter
    new_thread.regs.rip = addr;
    new_thread.regs.rcx = param;
    new_thread.regs.rax = 0;
    
    // Allocate stack if requested (otherwise will share/reuse current stack)
    if stack_size > 0 {
        if let Some(stack_base) = emu.maps.alloc(stack_size) {
            new_thread.regs.rsp = stack_base + stack_size - 8; // Stack grows down
            new_thread.regs.rbp = new_thread.regs.rsp;
            emu.maps.create_map(
                &format!("remote_thread_stack_{:x}", new_thread_id),
                stack_base,
                stack_size,
            ).ok();
        }
    }
    
    // Sync FPU instruction pointer
    new_thread.fpu.set_ip(addr);
    
    emu.threads.push(new_thread);
    
    if out_tid > 0 {
        emu.maps.write_dword(out_tid, new_thread_id as u32);
    }
    emu.regs_mut().rax = helper::handler_create(&format!("tid://0x{:x}", new_thread_id));
}