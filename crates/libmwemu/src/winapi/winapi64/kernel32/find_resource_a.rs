use crate::emu;
use crate::winapi::helper;

pub fn FindResourceA(emu: &mut emu::Emu) {
    let handle = emu.regs().rcx as usize;
    let lpName = emu.regs().rdx as usize;
    let lpType = emu.regs().r8 as usize;

    let x: Option<(u64, usize)>;

    if lpName > 0xff && lpType > 0xff {
        let name = emu.maps.read_string(lpName as u64);
        let ntype = emu.maps.read_string(lpType as u64);
        log_red!(emu, "** {} kernel32!FindResourceA {:x} `{}` `{}`", emu.pos, handle, name, ntype);

        x = emu.pe64.as_ref().unwrap().get_resource(None, None, Some(&name), Some(&ntype));
    } else if lpName > 0xff && lpType <= 0xff {
        let name = emu.maps.read_string(lpName as u64);
        log_red!(emu, "** {} kernel32!FindResourceA {:x} `{}` {}", emu.pos, handle, name, lpType);

        x = emu.pe64.as_ref().unwrap().get_resource(Some(lpType as u32), None, Some(&name), None);
    } else if lpName <= 0xff && lpType > 0xff {
        let ntype = emu.maps.read_string(lpType as u64);
        log_red!(emu, "** {} kernel32!FindResourceA {:x} `{}` {}", emu.pos, handle, lpName, ntype);

        x = emu.pe64.as_ref().unwrap().get_resource(None, Some(lpName as u32), None, Some(&ntype));
    } else if lpName <= 0xff && lpType <= 0xff {
        log_red!(emu, "** {} kernel32!FindResourceA {:x} `{}` {}", emu.pos, handle, lpName, lpType);

        x = emu.pe64.as_ref().unwrap().get_resource(Some(lpType as u32), Some(lpName as u32), None, None);
    } else {
        unreachable!();
    }

    if x.is_none() {
        log::info!("{} resource not found!", emu.pos);
        emu.regs_mut().rax = 0;
        return;
    }

    let (addr, size) = x.unwrap();


    log::info!("resource addr: 0x{:x} sz: {}", addr, size);
    let hndl = helper::handler_create(&format!("rsrc://{:x}_{}", addr, size));

    emu.regs_mut().rax = hndl;
}