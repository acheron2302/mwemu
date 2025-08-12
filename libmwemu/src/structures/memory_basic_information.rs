use crate::maps::Maps;

#[derive(Debug)]
pub struct MemoryBasicInformation {
    pub base_address: u32,
    pub allocation_base: u32,
    pub allocation_protect: u32,
    pub partition_id: u16,
    pub region_size: u32,
    pub state: u32,
    pub protect: u32,
    pub typ: u32,
}

impl MemoryBasicInformation {
    pub fn guess(addr: u64, maps: &mut Maps) -> MemoryBasicInformation {
        match maps.get_mem_by_addr_mut(addr) {
            Some(mem) => MemoryBasicInformation {
                base_address: mem.get_base() as u32,
                allocation_base: mem.get_base() as u32,
                allocation_protect: 0xff,
                partition_id: 0,
                region_size: mem.size() as u32,
                state: 0,
                protect: 0xff,
                typ: 0,
            },
            None => MemoryBasicInformation {
                base_address: 0,
                allocation_base: 0,
                allocation_protect: 0xff,
                partition_id: 0,
                region_size: 0,
                state: 0,
                protect: 0xff,
                typ: 0,
            },
        }
    }

    pub fn load(addr: u64, maps: &Maps) -> MemoryBasicInformation {
        MemoryBasicInformation {
            base_address: maps.read_dword(addr).unwrap(),
            allocation_base: maps.read_dword(addr + 4).unwrap(),
            allocation_protect: maps.read_dword(addr + 8).unwrap(),
            partition_id: maps.read_word(addr + 12).unwrap(),
            region_size: maps.read_dword(addr + 14).unwrap(),
            state: maps.read_dword(addr + 18).unwrap(),
            protect: maps.read_dword(addr + 22).unwrap(),
            typ: maps.read_dword(addr + 26).unwrap(),
        }
    }

    pub fn size() -> u64 {
        30
    }

    pub fn save(&self, addr: u64, maps: &mut Maps) {
        maps.write_dword(addr, self.base_address);
        maps.write_dword(addr + 4, self.allocation_base);
        maps.write_dword(addr + 8, self.allocation_protect);
        maps.write_word(addr + 12, self.partition_id);
        maps.write_dword(addr + 14, self.region_size);
        maps.write_dword(addr + 18, self.state);
        maps.write_dword(addr + 22, self.protect);
        maps.write_dword(addr + 26, self.typ);
    }

    pub fn print(&self) {
        log::info!("{:#x?}", self);
    }
}
