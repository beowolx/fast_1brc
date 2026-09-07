mod mapping {
    use std::ffi::{c_int, c_void};
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;

    #[cfg(not(all(
        target_pointer_width = "64",
        any(target_os = "linux", target_os = "macos")
    )))]
    compile_error!("fast_1brc supports 64-bit Linux and macOS");

    extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
        fn getpagesize() -> c_int;
    }

    pub struct Mapping {
        ptr: *mut c_void,
        len: usize,
    }

    impl Mapping {
        /// The file must not be modified or truncated until this mapping is dropped.
        pub unsafe fn open(file: &File) -> io::Result<Self> {
            let metadata = file.metadata()?;
            if !metadata.is_file() || metadata.len() > (isize::MAX as u64 >> 7) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "expected a regular file smaller than 64 PiB",
                ));
            }
            let len = metadata.len() as usize;
            if len == 0 {
                return Ok(Self {
                    ptr: std::ptr::null_mut(),
                    len,
                });
            }
            // PROT_READ = 1 and MAP_PRIVATE = 2 on both supported operating systems.
            let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) };
            if ptr as isize == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self { ptr, len })
        }

        pub fn page_size() -> io::Result<usize> {
            let size = unsafe { getpagesize() };
            usize::try_from(size)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| io::Error::other("invalid system page size"))
        }

        /// The offset must be page aligned and the file must contain the entire
        /// immutable region until this mapping is dropped.
        pub unsafe fn region(file: &File, offset: usize, len: usize) -> io::Result<Self> {
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    len,
                    1,
                    2,
                    file.as_raw_fd(),
                    offset as i64,
                )
            };
            if ptr as isize == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self { ptr, len })
        }

        pub fn bytes(&self) -> &[u8] {
            if self.len == 0 {
                return &[];
            }
            // The mapping owns this readable region; open's caller guarantees immutability.
            unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
        }
    }

    impl Drop for Mapping {
        fn drop(&mut self) {
            if self.len != 0 {
                unsafe {
                    munmap(self.ptr, self.len);
                }
            }
        }
    }
}
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{
    mpsc::{self, Receiver, Sender, SyncSender},
    Arc, Mutex,
};

const CHUNK_SIZE: usize = 8 * 1024 * 1024;
const MAX_STATIONS: usize = 10_000;
const TABLE_LEN: usize = 16_384;
const READ_MARGIN: usize = 109; // 100-byte name, ';', and an eight-byte temperature load.
const HASH_SEED: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Stats {
    sum: i64,
    count: u32,
    min: i16,
    max: i16,
}

impl Stats {
    #[inline(always)]
    fn add(&mut self, value: i16) {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
        unsafe {
            use std::arch::x86_64::*;
            // Stats is exactly 16 initialized bytes. Extrema use the original lanes before count increments.
            let pointer = std::ptr::from_mut(self).cast::<__m128i>();
            let current = _mm_loadu_si128(pointer);
            let added = _mm_add_epi64(current, _mm_set_epi64x(1, i64::from(value)));
            let values = _mm_set1_epi16(value);
            let minimum = _mm_min_epi16(current, values);
            let maximum = _mm_max_epi16(current, values);
            let result = _mm_blend_epi16::<0x40>(added, minimum);
            _mm_storeu_si128(pointer, _mm_blend_epi16::<0x80>(result, maximum));
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.1")))]
        {
            self.sum += i64::from(value);
            self.count += 1;
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
    }

    fn merge(&mut self, other: Self) {
        self.sum += other.sum;
        self.count += other.count;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
struct Slot {
    signature: u64,
    tail: u64,
    stats: Stats,
    // Low seven bits are the name length; the rest are its offset in the table name arena.
    name: u64,
}

struct Table {
    // The common lookup uses one prefix-keyed slot. Collisions use full-name hashes.
    slots: Vec<Slot>,
    secondary: Vec<Slot>,
    names: Vec<u8>,
    used: usize,
}

#[derive(Clone, Copy)]
struct Record {
    signature: u64,
    start: usize,
    len: usize,
    value: i16,
    next: usize,
}

#[cold]
fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[inline(always)]
fn bucket(signature: u64, mask: usize) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    unsafe {
        (std::arch::x86_64::_mm_crc32_u64(0, signature) as usize) & mask
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.2")))]
    {
        ((signature.wrapping_mul(HASH_SEED) >> 32) as usize) & mask
    }
}

impl Table {
    fn new() -> Self {
        Self {
            slots: vec![Slot::default(); TABLE_LEN],
            secondary: Vec::new(),
            names: Vec::with_capacity(64 * 1024),
            used: 0,
        }
    }

    #[cold]
    #[inline(never)]
    fn add_slow(&mut self, data: &[u8], mut record: Record) -> io::Result<()> {
        let primary = bucket(record.signature, TABLE_LEN - 1);
        let slot = &mut self.slots[primary];
        if slot.stats.count == 0 {
            self.slots[primary] = self.make_slot(data, record)?;
            return Ok(());
        }
        let offset = (slot.name >> 7) as usize;
        if slot.signature == record.signature
            && slot.name as usize & 127 == record.len
            && self.names[offset..offset + record.len]
                == data[record.start..record.start + record.len]
        {
            slot.stats.add(record.value);
            return Ok(());
        }
        if self.secondary.is_empty() {
            self.secondary = vec![Slot::default(); TABLE_LEN];
        }
        record.signature = full_name_hash(&data[record.start..record.start + record.len]);
        let mut index = bucket(record.signature, TABLE_LEN - 1);
        loop {
            let slot = &mut self.secondary[index];
            if slot.stats.count == 0 {
                self.secondary[index] = self.make_slot(data, record)?;
                return Ok(());
            }
            let offset = (slot.name >> 7) as usize;
            if slot.signature == record.signature
                && slot.name as usize & 127 == record.len
                && self.names[offset..offset + record.len]
                    == data[record.start..record.start + record.len]
            {
                slot.stats.add(record.value);
                return Ok(());
            }
            index = (index + 1) & (TABLE_LEN - 1);
        }
    }

    #[cold]
    fn make_slot(&mut self, data: &[u8], record: Record) -> io::Result<Slot> {
        if self.used == MAX_STATIONS {
            return Err(invalid("more than 10,000 stations"));
        }
        let name = &data[record.start..record.start + record.len];
        if name.contains(&b'\n') {
            return Err(invalid("newline inside station name"));
        }
        std::str::from_utf8(name).map_err(|_| invalid("station name is not UTF-8"))?;
        let mut tail = [0u8; 8];
        if name.len() > 8 {
            let bytes = &name[8..name.len().min(16)];
            tail[..bytes.len()].copy_from_slice(bytes);
        }
        let slot = Slot {
            signature: record.signature,
            tail: u64::from_le_bytes(tail),
            name: ((self.names.len() as u64) << 7) | record.len as u64,
            stats: Stats {
                sum: i64::from(record.value),
                count: 1,
                min: record.value,
                max: record.value,
            },
        };
        self.names.extend_from_slice(name);
        self.used += 1;
        Ok(slot)
    }
}

#[cold]
fn full_name_hash(name: &[u8]) -> u64 {
    let (chunks, remainder) = name.as_chunks::<8>();
    let mut hash = HASH_SEED;
    for chunk in chunks {
        hash = (hash ^ u64::from_le_bytes(*chunk)).wrapping_mul(HASH_SEED);
    }
    let mut tail = 0u64;
    for (index, &byte) in remainder.iter().enumerate() {
        tail |= u64::from(byte) << (index * 8);
    }
    (hash ^ tail ^ name.len() as u64).wrapping_mul(HASH_SEED)
}

#[inline(always)]
unsafe fn load_word(data: &[u8], pos: usize) -> u64 {
    debug_assert!(pos + 8 <= data.len());
    // Caller guarantees eight accessible bytes, including for every tail load.
    unsafe { data.as_ptr().add(pos).cast::<u64>().read_unaligned() }.to_le()
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn semicolon_match(word: u64) -> u64 {
    let x = word ^ 0x3b3b_3b3b_3b3b_3b3b;
    x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "ssse3",
    target_feature = "bmi2"
))]
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct TemperatureLayout {
    shuffle: [u8; 16],
    pad: u64,
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "ssse3",
    target_feature = "bmi2"
))]
const TEMPERATURE_LAYOUTS: [TemperatureLayout; 16] = {
    // Invalid bit layouts normalize to zero and fail the same format checks.
    let mut layouts = [TemperatureLayout {
        shuffle: [0x80; 16],
        pad: 4u64 << 48,
    }; 16];
    layouts[5] = TemperatureLayout {
        shuffle: [
            0x80, 0, 1, 2, 3, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        ],
        pad: 0x30 | (0x2du64 << 40) | (4u64 << 48),
    };
    layouts[11] = TemperatureLayout {
        shuffle: [
            0, 1, 2, 3, 4, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        ],
        pad: (0x2du64 << 40) | (5u64 << 48),
    };
    layouts[10] = TemperatureLayout {
        shuffle: [
            0x80, 1, 2, 3, 4, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        ],
        pad: 0x30 | (5u64 << 48) | (1u64 << 63),
    };
    layouts[6] = TemperatureLayout {
        shuffle: [
            1, 2, 3, 4, 5, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        ],
        pad: (6u64 << 48) | (1u64 << 63),
    };
    layouts
};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "ssse3",
    target_feature = "bmi2"
))]
#[inline(always)]
fn parse_temp(word: u64) -> io::Result<(i16, usize)> {
    use std::arch::x86_64::{
        _mm_cvtsi128_si64, _mm_cvtsi64_si128, _mm_loadu_si128, _mm_shuffle_epi8, _pext_u64,
    };
    let normalized = unsafe {
        let shape = _pext_u64(word, 0x1010_1010) as usize;
        // PEXT extracts four bits, so shape is always within this 16-entry table.
        let layout = TEMPERATURE_LAYOUTS.get_unchecked(shape);
        let shuffle = _mm_loadu_si128(layout.shuffle.as_ptr().cast());
        let bytes = _mm_shuffle_epi8(_mm_cvtsi64_si128(word as i64), shuffle);
        _mm_cvtsi128_si64(bytes) as u64 | layout.pad
    };
    let digits = normalized & 0x0f00_0f0f;
    if normalized & 0x0000_ffff_f0ff_f0f0 != 0x0000_2d0a_302e_3030
        || (digits + 0x0600_0606) & 0x1000_1010 != 0
    {
        return Err(invalid("invalid temperature"));
    }
    let absolute = (digits.wrapping_mul(0x640a_0001) >> 24) & 0x3ff;
    let sign = (normalized as i64) >> 63;
    Ok((
        ((absolute as i64 ^ sign) - sign) as i16,
        ((normalized >> 48) as u8) as usize,
    ))
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "ssse3",
    target_feature = "bmi2"
)))]
#[inline(always)]
fn parse_temp(word: u64) -> io::Result<(i16, usize)> {
    let negative = (word as u8 == b'-') as u32;
    let positive = word >> (negative * 8);
    let short = ((positive >> 8) as u8 == b'.') as u32;
    // Normalize all four layouts to the five bytes `dd.d\n`.
    let normalized = (positive << (short * 8)) | (u64::from(short) * 0x30);
    let digits = normalized & 0x0f00_0f0f;
    if normalized & 0x0000_00ff_f0ff_f0f0 != 0x0000_000a_302e_3030
        || (digits + 0x0600_0606) & 0x1000_1010 != 0
    {
        return Err(invalid("invalid temperature"));
    }
    let absolute = (digits.wrapping_mul(0x640a_0001) >> 24) & 0x3ff;
    let sign = -(negative as i64);
    Ok((
        ((absolute as i64 ^ sign) - sign) as i16,
        (5 + negative - short) as usize,
    ))
}

/// Requires READ_MARGIN accessible bytes from start.
#[cold]
#[inline(never)]
unsafe fn read_record(data: &[u8], start: usize) -> io::Result<Record> {
    let len = data[start..start + 101]
        .iter()
        .position(|&byte| byte == b';')
        .ok_or_else(|| invalid("station name exceeds 100 bytes or missing semicolon"))?;
    if len == 0 {
        return Err(invalid("station name must be 1..100 bytes"));
    }
    let (value, bytes) = parse_temp(unsafe { load_word(data, start + len + 1) })?;
    Ok(Record {
        signature: mask_name_word(unsafe { load_word(data, start) }, len),
        start,
        len,
        value,
        next: start + len + 1 + bytes,
    })
}

const SCAN_BYTES: usize = 16;
#[inline(always)]
unsafe fn scan_name(data: &[u8], start: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{
            _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
        };
        let bytes = _mm_loadu_si128(data.as_ptr().add(start).cast());
        let semicolons = _mm_cmpeq_epi8(bytes, _mm_set1_epi8(b';' as i8));
        (_mm_movemask_epi8(semicolons) as u32).trailing_zeros() as usize
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let first = semicolon_match(unsafe { load_word(data, start) });
        if first != 0 {
            return (first.trailing_zeros() >> 3) as usize;
        }
        let second = semicolon_match(unsafe { load_word(data, start + 8) });
        8 + (second.trailing_zeros() >> 3) as usize
    }
}

#[inline(always)]
fn mask_name_word(word: u64, len: usize) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        std::arch::x86_64::_bzhi_u64(word, (len * 8) as u32)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        if len >= 8 {
            word
        } else {
            word & ((1u64 << (len * 8)) - 1)
        }
    }
}

// start has READ_MARGIN accessible bytes; inserted station lengths are <=100.
#[inline(always)]
unsafe fn step(data: &[u8], start: usize, table: &mut Table) -> io::Result<usize> {
    let len = unsafe { scan_name(data, start) };
    if len >= SCAN_BYTES {
        return unsafe { slow_step(data, start, table) };
    }
    if len == 0 {
        return Err(invalid("station name must be 1..100 bytes"));
    }
    let key = mask_name_word(unsafe { load_word(data, start) }, len);
    let tail = mask_name_word(unsafe { load_word(data, start + 8) }, len.saturating_sub(8));
    let (value, bytes) = parse_temp(unsafe { load_word(data, start + len + 1) })?;
    let next = start + len + 1 + bytes;
    let slot = unsafe { table.slots.get_unchecked_mut(bucket(key, TABLE_LEN - 1)) };
    if ((slot.signature ^ key) | (slot.tail ^ tail) | ((slot.name & 127) ^ len as u64)) == 0 {
        slot.stats.add(value);
        return Ok(next);
    }
    table.add_slow(
        data,
        Record {
            signature: key,
            start,
            len,
            value,
            next,
        },
    )?;
    Ok(next)
}

#[cold]
#[inline(never)]
unsafe fn slow_step(data: &[u8], start: usize, table: &mut Table) -> io::Result<usize> {
    let key = unsafe { load_word(data, start) };
    let tail = unsafe { load_word(data, start + 8) };
    let slot = unsafe { table.slots.get_unchecked_mut(bucket(key, TABLE_LEN - 1)) };
    let len = slot.name as usize & 127;
    let offset = (slot.name >> 7) as usize;
    if len >= 16
        && ((slot.signature ^ key) | (slot.tail ^ tail)) == 0
        && data[start + len] == b';'
        && data[start + 16..start + len] == table.names[offset + 16..offset + len]
    {
        let (value, bytes) = parse_temp(unsafe { load_word(data, start + len + 1) })?;
        slot.stats.add(value);
        return Ok(start + len + 1 + bytes);
    }
    let record = unsafe { read_record(data, start) }?;
    table.add_slow(data, record)?;
    Ok(record.next)
}

fn read_tail(data: &[u8], start: usize) -> io::Result<Record> {
    let remaining = &data[start..];
    let mut padded = [0u8; READ_MARGIN + 1];
    padded[..remaining.len()].copy_from_slice(remaining);
    // Support a final row without LF without reading the last mapping page's padding.
    padded[remaining.len()] = b'\n';
    let mut record = unsafe { read_record(&padded, 0) }?;
    if record.next > remaining.len() + 1 {
        return Err(invalid("truncated final row"));
    }
    record.start = start;
    record.next = start + record.next.min(remaining.len());
    Ok(record)
}

fn line_start(data: &[u8], mut pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    while pos < data.len() && data[pos - 1] != b'\n' {
        pos += 1;
    }
    pos
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn parse_temperatures(words: [u64; 2]) -> io::Result<([i64; 2], [u64; 2])> {
    use std::arch::x86_64::*;
    unsafe {
        let input = _mm_loadu_si128(words.as_ptr().cast());
        let negative = _mm_cmpeq_epi64(
            _mm_and_si128(input, _mm_set1_epi64x(255)),
            _mm_set1_epi64x(i64::from(b'-')),
        );
        let positive = _mm_srlv_epi64(input, _mm_and_si128(negative, _mm_set1_epi64x(8)));
        let short = _mm_cmpeq_epi64(
            _mm_and_si128(_mm_srli_epi64::<8>(positive), _mm_set1_epi64x(255)),
            _mm_set1_epi64x(i64::from(b'.')),
        );
        let normalized = _mm_or_si128(
            _mm_sllv_epi64(positive, _mm_and_si128(short, _mm_set1_epi64x(8))),
            _mm_and_si128(short, _mm_set1_epi64x(0x30)),
        );
        let digits = _mm_and_si128(normalized, _mm_set1_epi64x(0x0f00_0f0f));
        let bad_layout = _mm_xor_si128(
            _mm_and_si128(normalized, _mm_set1_epi64x(0x0000_00ff_f0ff_f0f0)),
            _mm_set1_epi64x(0x0000_000a_302e_3030),
        );
        let bad_digits = _mm_and_si128(
            _mm_add_epi64(digits, _mm_set1_epi64x(0x0600_0606)),
            _mm_set1_epi64x(0x1000_1010),
        );
        let bad = _mm_or_si128(bad_layout, bad_digits);
        if _mm_testz_si128(bad, bad) == 0 {
            return Err(invalid("invalid temperature"));
        }
        // Each masked digit word fits u32; the full unsigned product preserves the scalar formula.
        let absolute = _mm_and_si128(
            _mm_srli_epi64::<24>(_mm_mul_epu32(digits, _mm_set1_epi64x(0x640a_0001))),
            _mm_set1_epi64x(0x3ff),
        );
        let signed = _mm_sub_epi64(_mm_xor_si128(absolute, negative), negative);
        let lengths = _mm_sub_epi64(
            _mm_add_epi64(
                _mm_set1_epi64x(5),
                _mm_and_si128(negative, _mm_set1_epi64x(1)),
            ),
            _mm_and_si128(short, _mm_set1_epi64x(1)),
        );
        let mut values = [0i64; 2];
        let mut bytes = [0u64; 2];
        _mm_storeu_si128(values.as_mut_ptr().cast(), signed);
        _mm_storeu_si128(bytes.as_mut_ptr().cast(), lengths);
        Ok((values, bytes))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
// Both selected positions have READ_MARGIN accessible bytes; group + 2 <= STREAMS.
#[inline(always)]
unsafe fn step_batch<const STREAMS: usize>(
    data: &[u8],
    positions: &mut [usize; STREAMS],
    group: usize,
    table: &mut Table,
) -> io::Result<()> {
    let mut lengths = [0; 2];
    let mut keys = [0; 2];
    let mut tails = [0; 2];
    let mut indices = [0; 2];
    let mut words = [u64::from_le_bytes(*b"0.0\n0000"); 2];
    for i in 0..2 {
        let start = positions[group + i];
        let len = unsafe { scan_name(data, start) };
        if len == 0 {
            return Err(invalid("station name must be 1..100 bytes"));
        }
        if len >= SCAN_BYTES {
            positions[group + i] = unsafe { slow_step(data, start, table) }?;
            continue;
        }
        lengths[i] = len;
        keys[i] = mask_name_word(unsafe { load_word(data, start) }, len);
        tails[i] = mask_name_word(unsafe { load_word(data, start + 8) }, len.saturating_sub(8));
        indices[i] = bucket(keys[i], TABLE_LEN - 1);
        // Overlap the random station cache-line fetch with independent vector temperature arithmetic.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                table.slots.as_ptr().add(indices[i]).cast(),
            );
        }
        words[i] = unsafe { load_word(data, start + len + 1) };
    }
    let (values, bytes) = unsafe { parse_temperatures(words) }?;
    for i in 0..2 {
        let len = lengths[i];
        if len == 0 {
            continue;
        }
        let start = positions[group + i];
        let value = values[i] as i16;
        let next = start + len + 1 + bytes[i] as usize;
        let slot = unsafe { table.slots.get_unchecked_mut(indices[i]) };
        if ((slot.signature ^ keys[i]) | (slot.tail ^ tails[i]) | ((slot.name & 127) ^ len as u64))
            == 0
        {
            slot.stats.add(value);
        } else {
            table.add_slow(
                data,
                Record {
                    signature: keys[i],
                    start,
                    len,
                    value,
                    next,
                },
            )?;
        }
        positions[group + i] = next;
    }
    Ok(())
}

fn process_chunk<const STREAMS: usize>(
    data: &[u8],
    start: usize,
    end: usize,
    table: &mut Table,
) -> io::Result<()> {
    let mut positions = [start; STREAMS];
    let mut ends = [end; STREAMS];
    for i in 1..STREAMS {
        let split = line_start(data, start + (end - start) * i / STREAMS);
        positions[i] = split;
        ends[i - 1] = split;
    }
    let fast_end = data.len().saturating_sub(READ_MARGIN);
    while (0..STREAMS).all(|i| positions[i] < ends[i] && positions[i] <= fast_end)
        && data.len() >= READ_MARGIN
    {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if STREAMS.is_multiple_of(2) {
            for group in (0..STREAMS).step_by(2) {
                unsafe { step_batch(data, &mut positions, group, table) }?;
            }
            continue;
        }
        for position in &mut positions {
            *position = unsafe { step(data, *position, table) }?;
        }
    }
    for i in 0..STREAMS {
        while positions[i] < ends[i] {
            if data.len() - positions[i] >= READ_MARGIN {
                positions[i] = unsafe { step(data, positions[i], table) }?;
            } else {
                let record = read_tail(data, positions[i])?;
                table.add_slow(data, record)?;
                positions[i] = record.next;
            }
        }
    }
    Ok(())
}

fn worker<const STREAMS: usize>(
    data: &[u8],
    next: &AtomicUsize,
    chunk_size: usize,
) -> io::Result<Table> {
    let mut table = Table::new();
    loop {
        let offset = next.fetch_add(chunk_size, Ordering::Relaxed);
        if offset >= data.len() {
            break;
        }
        let start = line_start(data, offset);
        let end = (offset + chunk_size).min(data.len());
        if start < end {
            process_chunk::<STREAMS>(data, start, end, &mut table)?;
        }
    }
    Ok(table)
}

fn advise_uncached(file: &File) {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        extern "C" {
            fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        }
        // F_NOCACHE is advisory; reading remains correct if the kernel rejects it.
        unsafe {
            fcntl(file.as_raw_fd(), 48, 1i32);
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = file;
}

struct Job {
    buffer: Vec<u8>,
    base: usize,
    len: usize,
    limit: usize,
}

// A flag alone cannot wake readers and parsers waiting on each other's queues.
fn receive<T>(queue: &Mutex<Receiver<T>>, cancel: &AtomicBool) -> io::Result<Option<T>> {
    while !cancel.load(Ordering::Relaxed) {
        let result = queue
            .lock()
            .map_err(|_| io::Error::other("pipeline queue poisoned"))?
            .recv_timeout(std::time::Duration::from_millis(10));
        match result {
            Ok(item) => return Ok(Some(item)),
            Err(mpsc::RecvTimeoutError::Timeout) => (),
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
        }
    }
    Ok(None)
}

fn pipeline_reader(
    file: &File,
    region: std::ops::Range<usize>,
    file_len: usize,
    chunk: usize,
    free: Arc<Mutex<Receiver<Vec<u8>>>>,
    work: SyncSender<Job>,
    cancel: &AtomicBool,
) -> io::Result<()> {
    let start = region.start;
    let end = region.end;
    const PAGE: usize = 16 * 1024;
    const OVERLAP: usize = 128;
    let mut previous: Option<(Vec<u8>, usize, usize)> = None;
    let mut preceding = b'\n';
    if start > 0 {
        file.read_exact_at(std::slice::from_mut(&mut preceding), (start - 1) as u64)?;
    }
    let mut offset = start;
    while offset < end && !cancel.load(Ordering::Relaxed) {
        let mut buffer = match receive(&free, cancel)? {
            Some(buffer) => buffer,
            None if cancel.load(Ordering::Relaxed) => return Ok(()),
            None => return Err(io::Error::other("all parser workers stopped")),
        };
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        let base = match buffer.as_ptr().align_offset(PAGE) {
            0 => PAGE,
            base => base,
        };
        let len = (end - offset).min(chunk);
        file.read_exact_at(&mut buffer[base..base + len], offset as u64)?;
        buffer[base - 1] = preceding;
        preceding = buffer[base + len - 1];
        if let Some((mut old, old_base, old_len)) = previous.take() {
            let tail = (file_len - offset).min(OVERLAP);
            let copied = len.min(tail);
            old[old_base + old_len..old_base + old_len + copied]
                .copy_from_slice(&buffer[base..base + copied]);
            if copied < tail {
                file.read_exact_at(
                    &mut old[old_base + old_len + copied..old_base + old_len + tail],
                    (offset + copied) as u64,
                )?;
            }
            if work
                .send(Job {
                    buffer: old,
                    base: old_base - 1,
                    len: 1 + old_len + tail,
                    limit: 1 + old_len,
                })
                .is_err()
            {
                return if cancel.load(Ordering::Relaxed) {
                    Ok(())
                } else {
                    Err(io::Error::other("all parser workers stopped"))
                };
            }
        }
        previous = Some((buffer, base, len));
        offset += len;
    }
    if !cancel.load(Ordering::Relaxed) {
        if let Some((mut buffer, base, len)) = previous {
            let tail = (file_len - end).min(OVERLAP);
            if tail > 0 {
                file.read_exact_at(&mut buffer[base + len..base + len + tail], end as u64)?;
            }
            if work
                .send(Job {
                    buffer,
                    base: base - 1,
                    len: 1 + len + tail,
                    limit: 1 + len,
                })
                .is_err()
                && !cancel.load(Ordering::Relaxed)
            {
                return Err(io::Error::other("all parser workers stopped"));
            }
        }
    }
    Ok(())
}

fn pipeline_worker<const STREAMS: usize>(
    work: Arc<Mutex<Receiver<Job>>>,
    free: Sender<Vec<u8>>,
    cancel: &AtomicBool,
) -> io::Result<Table> {
    let mut table = Table::new();
    while !cancel.load(Ordering::Relaxed) {
        let job = match receive(&work, cancel)? {
            Some(job) => job,
            None => break,
        };
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let data = &job.buffer[job.base..job.base + job.len];
        let start = line_start(data, 1);
        let result = if start < job.limit {
            process_chunk::<STREAMS>(data, start, job.limit, &mut table)
        } else {
            Ok(())
        };
        let _ = free.send(job.buffer);
        result?;
    }
    Ok(table)
}

fn aggregate_pipeline(
    file: &File,
    threads: usize,
    streams: usize,
    chunk_size: usize,
    readers: usize,
) -> io::Result<Vec<(String, Stats)>> {
    const PAGE: usize = 16 * 1024;
    const OVERLAP: usize = 128;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > (isize::MAX as u64 >> 7) {
        return Err(invalid("expected a regular file smaller than 64 PiB"));
    }
    let file_len = metadata.len() as usize;
    if file_len == 0 {
        return Ok(Vec::new());
    }
    let readers = readers.min(file_len.div_ceil(chunk_size).max(1));
    let threads = threads.min(file_len.div_ceil(chunk_size).max(1));
    let pool = 32.max(readers + 1);
    let (free_tx, free_rx) = mpsc::channel();
    let (work_tx, work_rx) = mpsc::sync_channel(pool);
    for _ in 0..pool {
        free_tx
            .send(vec![0u8; PAGE + chunk_size + OVERLAP])
            .map_err(|_| io::Error::other("free-buffer initialization failed"))?;
    }
    let free_rx = Arc::new(Mutex::new(free_rx));
    let work_rx = Arc::new(Mutex::new(work_rx));
    let cancel = AtomicBool::new(false);
    let tables = std::thread::scope(|scope| -> io::Result<Vec<Table>> {
        let mut reader_handles = Vec::new();
        for r in 0..readers {
            let start = (file_len * r / readers) & !(PAGE - 1);
            let end = if r + 1 == readers {
                file_len
            } else {
                (file_len * (r + 1) / readers) & !(PAGE - 1)
            };
            let free = free_rx.clone();
            let work = work_tx.clone();
            let cancelled = &cancel;
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                let result = pipeline_reader(
                    file,
                    start..end,
                    file_len,
                    chunk_size,
                    free,
                    work,
                    cancelled,
                );
                if result.is_err() {
                    cancelled.store(true, Ordering::Relaxed);
                }
                result
            }) {
                Ok(handle) => reader_handles.push(handle),
                Err(error) => {
                    cancel.store(true, Ordering::Relaxed);
                    drop(free_rx);
                    drop(work_tx);
                    drop(work_rx);
                    drop(free_tx);
                    return Err(error);
                }
            }
        }
        drop(free_rx);
        drop(work_tx);
        let mut worker_handles = Vec::new();
        for _ in 0..threads {
            let work = work_rx.clone();
            let free = free_tx.clone();
            let cancelled = &cancel;
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                let result = match streams {
                    1 => pipeline_worker::<1>(work, free, cancelled),
                    2 => pipeline_worker::<2>(work, free, cancelled),
                    3 => pipeline_worker::<3>(work, free, cancelled),
                    4 => pipeline_worker::<4>(work, free, cancelled),
                    _ => Err(io::Error::other("invalid stream count")),
                };
                if result.is_err() {
                    cancelled.store(true, Ordering::Relaxed);
                }
                result
            }) {
                Ok(handle) => worker_handles.push(handle),
                Err(error) => {
                    cancel.store(true, Ordering::Relaxed);
                    drop(work_rx);
                    drop(free_tx);
                    return Err(error);
                }
            }
        }
        drop(work_rx);
        drop(free_tx);
        let mut tables = Vec::new();
        let mut error = None;
        for handle in worker_handles {
            match handle.join() {
                Ok(Ok(table)) => tables.push(table),
                Ok(Err(failure)) => {
                    cancel.store(true, Ordering::Relaxed);
                    if error.is_none() {
                        error = Some(failure);
                    }
                }
                Err(_) => {
                    cancel.store(true, Ordering::Relaxed);
                    if error.is_none() {
                        error = Some(io::Error::other("parser worker panicked"));
                    }
                }
            }
        }
        for handle in reader_handles {
            match handle.join() {
                Ok(Ok(())) => (),
                Ok(Err(failure)) => {
                    if error.is_none() {
                        error = Some(failure);
                    }
                }
                Err(_) => {
                    if error.is_none() {
                        error = Some(io::Error::other("reader panicked"));
                    }
                }
            }
        }
        if let Some(error) = error {
            Err(error)
        } else {
            Ok(tables)
        }
    })?;
    merge_tables(tables)
}

fn env_usize(name: &str, default: usize, max: usize) -> io::Result<usize> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(default),
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0 && n <= max)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{name} must be 1..{max}"),
                )
            }),
        Err(error) => Err(io::Error::new(io::ErrorKind::InvalidInput, error)),
    }
}

fn mapped_window_worker<const STREAMS: usize>(
    file: &File,
    file_len: usize,
    next: &AtomicUsize,
    chunk_size: usize,
    window_size: usize,
    page_size: usize,
) -> io::Result<Table> {
    let mut table = Table::new();
    loop {
        let offset = next.fetch_add(window_size, Ordering::Relaxed);
        if offset >= file_len {
            break;
        }
        let end = (offset + window_size).min(file_len);
        let map_start = offset.saturating_sub(1) / page_size * page_size;
        let map_end = (end + READ_MARGIN).min(file_len);
        let mapped = unsafe { mapping::Mapping::region(file, map_start, map_end - map_start) }?;
        let data = mapped.bytes();
        let mut position = offset - map_start;
        while position < end - map_start {
            let start = line_start(data, position);
            let limit = (position + chunk_size).min(end - map_start);
            if start < limit {
                process_chunk::<STREAMS>(data, start, limit, &mut table)?;
            }
            position = limit;
        }
    }
    Ok(table)
}

fn aggregate_windowed(
    file: &File,
    threads: usize,
    streams: usize,
    chunk_size: usize,
    window_size: usize,
) -> io::Result<Vec<(String, Stats)>> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > (isize::MAX as u64 >> 7) {
        return Err(invalid("expected a regular file smaller than 64 PiB"));
    }
    let file_len = metadata.len() as usize;
    if file_len <= window_size {
        let mapped = unsafe { mapping::Mapping::open(file) }?;
        return aggregate(mapped.bytes(), threads, streams, chunk_size);
    }
    let next = AtomicUsize::new(0);
    let page_size = mapping::Mapping::page_size()?;
    let run_worker = || match streams {
        1 => mapped_window_worker::<1>(file, file_len, &next, chunk_size, window_size, page_size),
        2 => mapped_window_worker::<2>(file, file_len, &next, chunk_size, window_size, page_size),
        3 => mapped_window_worker::<3>(file, file_len, &next, chunk_size, window_size, page_size),
        4 => mapped_window_worker::<4>(file, file_len, &next, chunk_size, window_size, page_size),
        _ => unreachable!(),
    };
    aggregate_workers(threads.min(file_len.div_ceil(window_size)), run_worker)
}

fn aggregate(
    data: &[u8],
    threads: usize,
    streams: usize,
    chunk_size: usize,
) -> io::Result<Vec<(String, Stats)>> {
    let next = AtomicUsize::new(0);
    let run_worker = || match streams {
        1 => worker::<1>(data, &next, chunk_size),
        2 => worker::<2>(data, &next, chunk_size),
        3 => worker::<3>(data, &next, chunk_size),
        4 => worker::<4>(data, &next, chunk_size),
        _ => unreachable!(),
    };
    let threads = threads.min(data.len().div_ceil(chunk_size).max(1));
    aggregate_workers(threads, run_worker)
}

fn aggregate_workers<F>(threads: usize, run_worker: F) -> io::Result<Vec<(String, Stats)>>
where
    F: Fn() -> io::Result<Table> + Sync,
{
    let run_worker = &run_worker;
    let tables = if threads == 1 {
        vec![run_worker()?]
    } else {
        std::thread::scope(|scope| -> io::Result<Vec<Table>> {
            let mut handles = Vec::with_capacity(threads - 1);
            for _ in 1..threads {
                handles.push(std::thread::Builder::new().spawn_scoped(scope, run_worker)?);
            }
            let first = run_worker();
            let mut tables = Vec::with_capacity(threads);
            let mut error = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(table)) => tables.push(table),
                    Ok(Err(e)) => error = Some(e),
                    Err(_) => error = Some(io::Error::other("worker panicked")),
                }
            }
            if let Some(error) = error {
                return Err(error);
            }
            tables.push(first?);
            Ok(tables)
        })?
    };
    merge_tables(tables)
}

fn merge_tables(tables: Vec<Table>) -> io::Result<Vec<(String, Stats)>> {
    let mut merged = BTreeMap::<&[u8], Stats>::new();
    for table in &tables {
        for slot in table
            .slots
            .iter()
            .chain(&table.secondary)
            .filter(|slot| slot.stats.count != 0)
        {
            let offset = (slot.name >> 7) as usize;
            let name = &table.names[offset..offset + (slot.name as usize & 127)];
            merged
                .entry(name)
                .and_modify(|stats| stats.merge(slot.stats))
                .or_insert(slot.stats);
        }
    }
    if merged.len() > MAX_STATIONS {
        return Err(invalid("more than 10,000 stations"));
    }
    let mut result = Vec::with_capacity(merged.len());
    for (name, stats) in merged {
        let name = std::str::from_utf8(name).map_err(|_| invalid("station name is not UTF-8"))?;
        result.push((name.to_owned(), stats));
    }
    // Java String.compareTo orders UTF-16 code units, including supplementary names.
    result.sort_unstable_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
    Ok(result)
}
fn push_tenths(out: &mut Vec<u8>, tenths: i64) {
    let mut v = tenths;
    if v < 0 {
        out.push(b'-');
        v = -v;
    }
    let whole = v / 10;
    if whole >= 10 {
        out.push(b'0' + (whole / 10) as u8);
    }
    out.push(b'0' + (whole % 10) as u8);
    out.push(b'.');
    out.push(b'0' + (v % 10) as u8);
}

fn mean_tenths(sum: i64, count: u64) -> i64 {
    let count = count as i64;
    (2 * sum + count).div_euclid(2 * count)
}

fn output(rows: &[(String, Stats)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * 48 + 3);
    out.push(b'{');
    for (index, (name, stats)) in rows.iter().enumerate() {
        if index != 0 {
            out.extend_from_slice(b", ");
        }
        out.extend_from_slice(name.as_bytes());
        out.push(b'=');
        push_tenths(&mut out, i64::from(stats.min));
        out.push(b'/');
        push_tenths(&mut out, mean_tenths(stats.sum, u64::from(stats.count)));
        out.push(b'/');
        push_tenths(&mut out, i64::from(stats.max));
    }
    out.extend_from_slice(b"}\n");
    out
}

#[derive(Clone, Copy)]
enum InputMode {
    Read,
    Mmap,
}

impl InputMode {
    fn from_env() -> io::Result<Self> {
        match std::env::var("FAST1BRC_IO") {
            Err(std::env::VarError::NotPresent) => Ok(if cfg!(target_os = "linux") {
                Self::Mmap
            } else {
                Self::Read
            }),
            Ok(value) if value == "read" => Ok(Self::Read),
            Ok(value) if value == "mmap" => Ok(Self::Mmap),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "FAST1BRC_IO must be read or mmap",
            )),
        }
    }
}

fn main() -> io::Result<()> {
    let filename = std::env::args_os()
        .nth(1)
        .unwrap_or_else(|| "measurements.txt".into());
    let threads = env_usize(
        "FAST1BRC_THREADS",
        std::thread::available_parallelism().map_or(1, usize::from),
        1024,
    )?;
    let streams = env_usize("FAST1BRC_STREAMS", 2, 4)?;
    let mode = InputMode::from_env()?;
    let default_chunk = match mode {
        InputMode::Read => CHUNK_SIZE / 1024,
        InputMode::Mmap => 1024,
    };
    let chunk_size = env_usize("FAST1BRC_CHUNK_KIB", default_chunk, 65536)? * 1024;
    let file = File::open(filename)?;
    let rows = match mode {
        InputMode::Read => {
            if std::env::var_os("FAST1BRC_CACHED").is_none() {
                advise_uncached(&file);
            }
            let readers = env_usize("FAST1BRC_READERS", 4, 64)?;
            aggregate_pipeline(&file, threads, streams, chunk_size, readers)?
        }
        InputMode::Mmap => {
            let window_size = env_usize("FAST1BRC_WINDOW_MIB", 16, 1024)? * 1024 * 1024;
            aggregate_windowed(&file, threads, streams, chunk_size, window_size)?
        }
    };
    io::stdout().lock().write_all(&output(&rows))
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn all_temperatures_and_rounding() {
        for value in -999i16..=999 {
            let line = format!(
                "{}{}.{}\n",
                if value < 0 { "-" } else { "" },
                value.abs() / 10,
                value.abs() % 10
            );
            let mut padded = [0; 8];
            padded[..line.len()].copy_from_slice(line.as_bytes());
            assert_eq!(
                parse_temp(u64::from_le_bytes(padded)).unwrap(),
                (value, line.len())
            );
        }
        for (sum, count, mean) in [
            (15, 10, 2),
            (-15, 10, -1),
            (-16, 10, -2),
            (0, 5, 0),
            (999_000_000_000, 1_000_000_000, 999),
        ] {
            assert_eq!(mean_tenths(sum, count), mean);
        }
        assert_eq!(std::mem::size_of::<Slot>(), 64);
    }

    #[test]
    fn boundaries_and_collisions() {
        let mut data = Vec::new();
        for i in 0..10_000 {
            writeln!(
                data,
                "station{i:05}{};{}.{}",
                "x".repeat(i % 89),
                i % 100,
                i % 10
            )
            .unwrap();
        }
        let expected = output(&aggregate(&data, 1, 1, CHUNK_SIZE).unwrap());
        for streams in 1..=4 {
            assert_eq!(
                output(&aggregate(&data, 4, streams, 1024).unwrap()),
                expected
            );
        }
        let name1 = b"abcdefgh12345678";
        let name2 = b"hgfedcba87654321";
        let mut table = Table::new();
        let mut names = name1.to_vec();
        names.extend(name2);
        for start in [0, 16, 0, 16] {
            table
                .add_slow(
                    &names,
                    Record {
                        signature: 42,
                        start,
                        len: 16,
                        value: start as i16,
                        next: 0,
                    },
                )
                .unwrap();
        }
        assert_eq!(table.used, 2);
        assert!(table
            .slots
            .iter()
            .chain(&table.secondary)
            .filter(|s| s.stats.count != 0)
            .all(|s| s.stats.count == 2));
    }

    #[test]
    fn short_tail_unicode_and_invalid_input() {
        for data in [
            b"".as_slice(),
            b"a;0.0",
            b"a;-0.0\na;-0.1\n",
            "\u{e000};1.0\n\u{10000};2.0\n".as_bytes(),
        ] {
            for streams in 1..=4 {
                aggregate(data, 4, streams, 1).unwrap();
            }
        }
        assert_eq!(
            output(&aggregate(b"a;-0.1\na;0.0", 1, 3, CHUNK_SIZE).unwrap()),
            b"{a=-0.1/0.0/0.0}\n"
        );
        for data in [
            b"a;\n".as_slice(),
            b";1.0\n",
            b"a;100.0\n",
            b"a;1.23\n",
            b"a\nb;1.0\n",
            b"\xff;1.0\n",
            &[b'x'; 200],
        ] {
            assert!(aggregate(data, 1, 3, CHUNK_SIZE).is_err(), "{data:?}");
        }
    }
}

#[cfg(test)]
mod packed_validation_tests {
    use crate::*;

    #[test]
    fn temperature_rejects_non_digits_and_bad_layouts() {
        for value in -999i16..=999 {
            let text = format!(
                "{}{}.{}\n",
                if value < 0 { "-" } else { "" },
                value.abs() / 10,
                value.abs() % 10
            );
            for position in 0..text.len() {
                for replacement in 0..=255u8 {
                    let mut bytes = [0u8; 8];
                    bytes[..text.len()].copy_from_slice(text.as_bytes());
                    bytes[position] = replacement;
                    let line = &bytes[..text.len()];
                    let negative = line[0] == b'-';
                    let start = usize::from(negative);
                    let dot = line.len() - 3;
                    let integer = &line[start..dot];
                    let valid = (1..=2).contains(&integer.len())
                        && integer.iter().all(u8::is_ascii_digit)
                        && line[dot] == b'.'
                        && line[dot + 1].is_ascii_digit()
                        && line[dot + 2] == b'\n';
                    let parsed = parse_temp(u64::from_le_bytes(bytes));
                    assert_eq!(parsed.is_ok(), valid, "{line:?}");
                    if valid {
                        let whole = integer
                            .iter()
                            .fold(0i16, |acc, byte| acc * 10 + i16::from(byte - b'0'));
                        let absolute = whole * 10 + i16::from(line[dot + 1] - b'0');
                        assert_eq!(
                            parsed.unwrap(),
                            (if negative { -absolute } else { absolute }, line.len()),
                            "{line:?}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod file_tests {
    use crate::*;
    #[test]
    fn buffered_reuse_and_boundaries() {
        let path =
            std::env::temp_dir().join(format!("fast1brc-buffer-test-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        for data in [
            b"".to_vec(),
            b"a;0.0".to_vec(),
            b"a;-0.1\na;0.0\n".to_vec(),
            (0..5000)
                .map(|i| {
                    format!(
                        "samefirst{}{};{}.{}\n",
                        i % 333,
                        "x".repeat(i % 85),
                        i % 99,
                        i % 10
                    )
                })
                .collect::<String>()
                .into_bytes(),
        ] {
            file.set_len(0).unwrap();
            file.write_all_at(&data, 0).unwrap();
            let expected = output(&aggregate(&data, 1, 1, CHUNK_SIZE).unwrap());
            for streams in 1..=4 {
                for chunk in [113, 1024, 8192] {
                    assert_eq!(
                        output(&aggregate_pipeline(&file, 4, streams, chunk, 4).unwrap()),
                        expected
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod prefix_tests {
    use crate::*;
    #[test]
    fn prefix_matches_are_exact_and_validate_reused_names() {
        let names = [
            "a",
            "a\0",
            "a\0\0\0\0\0\0\0X",
            "abcdefgh",
            "abcdefghI",
            "abcdefgh12345678",
            "abcdefgh12345679",
            "abcdefgh12345678long",
        ];
        let mut data = Vec::new();
        for _ in 0..100 {
            for (index, name) in names.iter().enumerate() {
                writeln!(data, "{name};{}.0", index as i16 - 4).unwrap();
            }
        }
        for streams in 1..=4 {
            let rows = aggregate(&data, 4, streams, 131).unwrap();
            assert_eq!(rows.len(), names.len());
            for (index, name) in names.iter().enumerate() {
                let stats = &rows.iter().find(|(n, _)| n == name).unwrap().1;
                let value = (index as i16 - 4) * 10;
                assert_eq!(
                    (stats.count, stats.sum, stats.min, stats.max),
                    (100, i64::from(value) * 100, value, value)
                );
            }
        }
        let mut bad = b"abcdefghLong;1.0\n".repeat(100);
        bad.extend_from_slice(b"abcdefghLong;1.x\n");
        bad.extend_from_slice(&b"abcdefghLong;1.0\n".repeat(100));
        assert!(aggregate(&bad, 1, 2, CHUNK_SIZE).is_err());
    }
}

#[cfg(test)]
mod pipeline_tests {
    use crate::*;
    #[test]
    fn cancellation_wakes_empty_queues_with_live_senders() {
        let (sender, receiver) = mpsc::channel::<()>();
        let queue = Mutex::new(receiver);
        let cancel = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| receive(&queue, &cancel));
            std::thread::sleep(std::time::Duration::from_millis(20));
            cancel.store(true, Ordering::Relaxed);
            assert_eq!(waiting.join().unwrap().unwrap(), None);
        });
        // Keep the channel connected until the blocked receiver has returned.
        drop(sender);
    }

    #[test]
    fn pipeline_errors_and_reader_boundaries() {
        let path =
            std::env::temp_dir().join(format!("fast1brc-pipeline-test-{}", std::process::id()));
        let mut write_only = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        write_only
            .write_all(&b"abcdefghLong;1.0\n".repeat(20000))
            .unwrap();
        assert!(aggregate_pipeline(&write_only, 4, 2, 1024, 4).is_err());
        let file = File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let expected = output(&aggregate_pipeline(&file, 1, 1, 16384, 1).unwrap());
        for readers in [1, 2, 4, 33] {
            for chunk in [1, 113, 1024] {
                assert_eq!(
                    output(&aggregate_pipeline(&file, 4, 2, chunk, readers).unwrap()),
                    expected
                );
            }
        }
        write_only.write_all_at(b"abcdefghLong;1.x\n", 0).unwrap();
        assert!(aggregate_pipeline(&file, 4, 2, 1024, 4).is_err());
    }
}

#[cfg(test)]
mod secondary_tests {
    use crate::*;
    #[test]
    fn ten_thousand_identical_prefixes_use_secondary_and_remain_distinct() {
        let mut table = Table::new();
        for pass in 0..2 {
            for index in 0..10_000 {
                let text = format!("station-{index:05};{}.0\n", pass + 1);
                let mut padded = [0u8; READ_MARGIN];
                padded[..text.len()].copy_from_slice(text.as_bytes());
                unsafe { step(&padded, 0, &mut table) }.unwrap();
            }
        }
        assert_eq!(table.used, 10_000);
        assert_eq!(table.slots.iter().filter(|s| s.stats.count != 0).count(), 1);
        assert_eq!(
            table
                .secondary
                .iter()
                .filter(|s| s.stats.count != 0)
                .count(),
            9_999
        );
        let rows = merge_tables(vec![table]).unwrap();
        assert_eq!(rows.len(), 10_000);
        for (index, (name, stats)) in rows.iter().enumerate() {
            assert_eq!(name, &format!("station-{index:05}"));
            assert_eq!(
                (stats.count, stats.sum, stats.min, stats.max),
                (2, 30, 10, 20)
            );
        }
    }
}

#[cfg(test)]
mod scanner_tests {
    use crate::*;

    #[test]
    fn inline_keys_remain_exact_at_all_lengths_and_alignments() {
        let mut table = Table::new();
        let mut data = Vec::new();
        for pass in 0..2 {
            for len in 1..=100 {
                for zeros in 0..2 {
                    let first = if zeros == 0 { b'x' } else { b'\0' };
                    let name = vec![first; len];
                    let start = data.len();
                    data.extend_from_slice(&name);
                    data.extend_from_slice(if pass == 0 { b";-99.9\n" } else { b";99.9\n" });
                    data.resize(data.len() + READ_MARGIN, 0);
                    unsafe { step(&data, start, &mut table) }.unwrap();
                    data.truncate(data.len() - READ_MARGIN);
                }
            }
        }
        let rows = merge_tables(vec![table]).unwrap();
        assert_eq!(rows.len(), 200);
        assert!(rows
            .iter()
            .all(|(_, stats)| (stats.count, stats.sum, stats.min, stats.max) == (2, 0, -999, 999)));
    }
}

#[cfg(test)]
mod window_tests {
    use crate::*;

    #[test]
    fn independent_mapping_windows_preserve_boundary_rows_and_tails() {
        let path =
            std::env::temp_dir().join(format!("fast1brc-window-test-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        let mut data: Vec<u8> = (0..4_000)
            .flat_map(|i| {
                format!(
                    "station-{i:04}{};{}.{}\n",
                    "x".repeat(i % 85),
                    i % 99,
                    i % 10
                )
                .into_bytes()
            })
            .collect();
        for name in ["😀".repeat(25), format!("a{}", "\0".repeat(99))] {
            for _ in 0..40 {
                writeln!(data, "{name};-99.9").unwrap();
            }
        }
        for data in [&data[..], &data[..data.len() - 1]] {
            file.set_len(0).unwrap();
            file.write_all_at(data, 0).unwrap();
            let expected = output(&aggregate(data, 1, 1, CHUNK_SIZE).unwrap());
            for streams in 1..=4 {
                for window in [257, 4096, 65536] {
                    assert_eq!(
                        output(&aggregate_windowed(&file, 4, streams, 113, window).unwrap()),
                        expected
                    );
                }
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64", target_feature = "avx2"))]
mod batched_temperature_tests {
    use crate::*;

    fn word(value: i16) -> u64 {
        let text = format!(
            "{}{}.{}\n",
            if value < 0 { "-" } else { "" },
            value.abs() / 10,
            value.abs() % 10
        );
        let mut bytes = [0; 8];
        bytes[..text.len()].copy_from_slice(text.as_bytes());
        u64::from_le_bytes(bytes)
    }

    #[test]
    fn vector_temperatures_match_scalar_for_all_values_and_malformed_lanes() {
        for first in -999..=999 {
            let words =
                std::array::from_fn(|lane| word((first + lane as i16 * 433 + 999) % 1999 - 999));
            let (values, lengths) = unsafe { parse_temperatures(words) }.unwrap();
            for lane in 0..2 {
                let expected = parse_temp(words[lane]).unwrap();
                assert_eq!(
                    (values[lane], lengths[lane]),
                    (i64::from(expected.0), expected.1 as u64)
                );
            }
        }
        for text in [
            b"1.0\n".as_slice(),
            b"12.3\n",
            b"-1.0\n",
            b"-12.3\n",
            b"09.0\n",
            b"-09.0\n",
        ] {
            for lane in 0..2 {
                for position in 0..text.len() {
                    for replacement in 0..=255 {
                        let mut words = [word(321); 2];
                        let mut bytes = [0; 8];
                        bytes[..text.len()].copy_from_slice(text);
                        bytes[position] = replacement;
                        words[lane] = u64::from_le_bytes(bytes);
                        let expected = parse_temp(words[lane]);
                        let actual = unsafe { parse_temperatures(words) };
                        assert_eq!(actual.is_ok(), expected.is_ok(), "{bytes:?}");
                        if let (Ok((values, lengths)), Ok((value, len))) = (actual, expected) {
                            assert_eq!(
                                (values[lane], lengths[lane]),
                                (i64::from(value), len as u64)
                            );
                        }
                    }
                }
            }
        }
    }
}
