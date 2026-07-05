use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Mutex;

const DEFAULT_FILE: &str = "measurements.txt";

const CHUNK_SIZE: u64 = 8 * 1024 * 1024;
const POOL_SIZE: usize = 32;
const MAX_LINE: usize = 107;
const OVERLAP: usize = 128;
const PAD: usize = 16;

const TABLE_BITS: u32 = 15;
const TABLE_LEN: usize = 1 << TABLE_BITS;
const HASH_SEED: u64 = 0x517c_c1b7_2722_0a95;

#[repr(C)]
#[derive(Clone)]
struct Slot {
    hash: u64,
    sum: i64,
    count: u32,
    min: i16,
    max: i16,
    name_len: u16,
    name: [u8; 102],
}

const EMPTY_SLOT: Slot = Slot {
    hash: 0,
    sum: 0,
    count: 0,
    min: 0,
    max: 0,
    name_len: 0,
    name: [0; 102],
};

#[inline(always)]
fn load_u64(buf: &[u8], pos: usize) -> u64 {
    debug_assert!(pos + 8 <= buf.len());
    unsafe { (buf.as_ptr().add(pos) as *const u64).read_unaligned() }.to_le()
}

#[inline(always)]
fn semicolon_match(word: u64) -> u64 {
    let x = word ^ 0x3b3b_3b3b_3b3b_3b3b;
    x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080
}

#[inline(always)]
fn parse_temp(word: u64) -> (i16, usize) {
    let negated = !word;
    let dot_pos = (negated & 0x1010_1000).trailing_zeros();
    let signed = ((negated as i64) << 59) >> 63;
    let design_mask = !(signed as u64 & 0xff);
    let digits = ((word & design_mask) << (28 - dot_pos)) & 0x0f000f0f00u64;
    let abs_value = (digits.wrapping_mul(0x640a0001) >> 32) & 0x3ff;
    let value = ((abs_value as i64) ^ signed) - signed;
    let len = (dot_pos as usize >> 3) + 3;
    (value as i16, len)
}

#[inline(always)]
fn names_equal(a: &[u8], b: &[u8], len: usize) -> bool {
    let ld = |s: &[u8], i: usize| -> u64 {
        unsafe { (s.as_ptr().add(i) as *const u64).read_unaligned() }.to_le()
    };
    let mut i = 0;
    while i + 8 <= len {
        if ld(a, i) != ld(b, i) {
            return false;
        }
        i += 8;
    }
    if i < len {
        let mask = (1u64 << (8 * (len - i))) - 1;
        if (ld(a, i) ^ ld(b, i)) & mask != 0 {
            return false;
        }
    }
    true
}

#[inline(always)]
fn step(buf: &[u8], pos: usize, table: &mut [Slot], used: &mut usize) -> usize {
    let start = pos;
    let mut hash = 0u64;
    let mut p = pos;
    let semi;
    loop {
        let word = load_u64(buf, p);
        let m = semicolon_match(word);
        if m != 0 {
            let tz = m.trailing_zeros() as usize; // 8*i + 7 for byte i
            let keep = word & ((1u64 << (tz - 7)) - 1);
            hash = (hash ^ keep).wrapping_mul(HASH_SEED);
            semi = p + (tz >> 3);
            break;
        }
        hash = (hash ^ word).wrapping_mul(HASH_SEED);
        p += 8;
    }
    let name_len = (semi - start).min(102);

    let (value, temp_len) = parse_temp(load_u64(buf, semi + 1));
    let next = semi + 1 + temp_len;

    let mut idx = (hash >> (64 - TABLE_BITS)) as usize;
    loop {
        let slot = unsafe { table.get_unchecked_mut(idx) };
        if slot.count != 0 {
            if slot.hash == hash
                && slot.name_len == name_len as u16
                && names_equal(&buf[start..], &slot.name, name_len)
            {
                slot.count += 1;
                slot.sum += value as i64;
                if value < slot.min {
                    slot.min = value;
                }
                if value > slot.max {
                    slot.max = value;
                }
                break;
            }
            idx = (idx + 1) & (TABLE_LEN - 1);
            continue;
        }
        *used += 1;
        assert!(*used < TABLE_LEN, "too many unique stations for the table");
        slot.hash = hash;
        slot.name_len = name_len as u16;
        slot.name[..name_len].copy_from_slice(&buf[start..start + name_len]);
        slot.min = value;
        slot.max = value;
        slot.sum = value as i64;
        slot.count = 1;
        break;
    }
    next
}

fn process_chunk(buf: &[u8], pos: usize, limit: usize, table: &mut [Slot], used: &mut usize) {
    let mut mid = limit;
    if limit - pos > 4096 {
        let target = pos + (limit - pos) / 2;
        if let Some(i) = buf[target..limit].iter().position(|&c| c == b'\n') {
            mid = target + i + 1;
        }
    }

    let mut a = pos;
    let mut b = mid;
    while a < mid && b < limit {
        a = step(buf, a, table, used);
        b = step(buf, b, table, used);
    }
    while a < mid {
        a = step(buf, a, table, used);
    }
    while b < limit {
        b = step(buf, b, table, used);
    }
}

struct Job {
    buf: Vec<u8>,
    base: usize,
    n: usize,
    limit: usize,
}

const PAGE: usize = 16384;
const BUF_CAPACITY: usize = PAGE + CHUNK_SIZE as usize + OVERLAP + PAD;

fn page_base(buf: &[u8]) -> usize {
    match buf.as_ptr().align_offset(PAGE) {
        0 => PAGE,
        a => a,
    }
}

fn advise_uncached(file: &File) {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        const F_NOCACHE: i32 = 48;
        extern "C" {
            fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        }
        unsafe {
            fcntl(file.as_raw_fd(), F_NOCACHE, 1i32);
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = file;
}

fn run_reader(
    file: &File,
    region_start: u64,
    region_end: u64,
    file_size: u64,
    free_rx: &Mutex<Receiver<Vec<u8>>>,
    work_tx: SyncSender<Job>,
) -> io::Result<()> {
    let mut prev: Option<(Vec<u8>, usize, usize)> = None; // (buf, base, len)
    let mut offset = region_start;
    let mut prev_last_byte = b'\n';
    if region_start > 0 {
        let mut b = [0u8; 1];
        file.read_exact_at(&mut b, region_start - 1)?;
        prev_last_byte = b[0];
    }

    let finalize = |buf: &mut Vec<u8>, base: usize, len: usize, tail: usize| -> Job {
        let n = 1 + len + tail;
        let sentinel = base - 1 + n;
        buf[sentinel] = b';';
        buf[sentinel + 1..sentinel + PAD].fill(0);
        Job {
            buf: std::mem::take(buf),
            base,
            n,
            limit: 1 + len,
        }
    };

    while offset < region_end {
        let mut buf = free_rx
            .lock()
            .unwrap()
            .recv()
            .expect("all workers exited early");
        let base = page_base(&buf);
        let len = (region_end - offset).min(CHUNK_SIZE) as usize;
        file.read_exact_at(&mut buf[base..base + len], offset)?;
        buf[base - 1] = prev_last_byte;
        prev_last_byte = buf[base + len - 1];

        if let Some((mut pbuf, pbase, plen)) = prev.take() {
            let tail = len.min(OVERLAP);
            pbuf[pbase + plen..pbase + plen + tail].copy_from_slice(&buf[base..base + tail]);
            let job = finalize(&mut pbuf, pbase, plen, tail);
            if work_tx.send(job).is_err() {
                return Ok(());
            }
        }
        prev = Some((buf, base, len));
        offset += len as u64;
    }

    if let Some((mut pbuf, pbase, plen)) = prev.take() {
        let mut tail = 0;
        if region_end < file_size {
            tail = ((file_size - region_end) as usize).min(OVERLAP);
            file.read_exact_at(&mut pbuf[pbase + plen..pbase + plen + tail], region_end)?;
        }
        let job = finalize(&mut pbuf, pbase, plen, tail);
        let _ = work_tx.send(job);
    }
    Ok(())
}

fn run_worker(work_rx: &Mutex<Receiver<Job>>, free_tx: SyncSender<Vec<u8>>) -> Vec<Slot> {
    let mut table = vec![EMPTY_SLOT; TABLE_LEN + 1];
    let mut used = 0usize;

    loop {
        let job = match work_rx.lock().unwrap().recv() {
            Ok(job) => job,
            Err(_) => break,
        };

        let data = &job.buf[job.base - 1..job.base - 1 + job.n + PAD];
        let pos = match data[..job.n.min(MAX_LINE + 1)]
            .iter()
            .position(|&b| b == b'\n')
        {
            Some(i) => i + 1,
            None => job.limit,
        };
        process_chunk(data, pos, job.limit, &mut table, &mut used);

        let _ = free_tx.send(job.buf);
    }

    table
}

struct Stats {
    min: i16,
    max: i16,
    sum: i64,
    count: u64,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn process_file(filename: &str) -> io::Result<BTreeMap<Vec<u8>, Stats>> {
    let file = File::open(filename)?;
    let file_size = file.metadata()?.len();
    let num_workers = env_usize(
        "FAST1BRC_THREADS",
        std::thread::available_parallelism().map_or(8, |n| n.get()),
    );
    let num_readers = env_usize("FAST1BRC_READERS", 4).max(1);

    if std::env::var_os("FAST1BRC_CACHED").is_none() {
        advise_uncached(&file);
    }

    let bounds: Vec<u64> = (0..=num_readers as u64)
        .map(|i| {
            let b = file_size * i / num_readers as u64;
            if i == num_readers as u64 {
                file_size
            } else {
                b & !(PAGE as u64 - 1)
            }
        })
        .collect();

    let (work_tx, work_rx) = sync_channel::<Job>(POOL_SIZE);
    let (free_tx, free_rx) = sync_channel::<Vec<u8>>(POOL_SIZE);
    for _ in 0..POOL_SIZE {
        free_tx.send(vec![0u8; BUF_CAPACITY]).unwrap();
    }
    let work_rx = Mutex::new(work_rx);
    let free_rx = Mutex::new(free_rx);

    let (reader_results, tables) = std::thread::scope(|s| {
        let file_ref = &file;
        let free_rx_ref = &free_rx;
        let readers: Vec<_> = (0..num_readers)
            .map(|r| {
                let work_tx = work_tx.clone();
                let (start, end) = (bounds[r], bounds[r + 1]);
                s.spawn(move || run_reader(file_ref, start, end, file_size, free_rx_ref, work_tx))
            })
            .collect();
        drop(work_tx);
        let work_rx_ref = &work_rx;
        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let free_tx = free_tx.clone();
                s.spawn(move || run_worker(work_rx_ref, free_tx))
            })
            .collect();
        drop(free_tx); // readers' recv unblocks once every worker is done
        let tables: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("worker thread panicked"))
            .collect();
        let reader_results: Vec<_> = readers
            .into_iter()
            .map(|h| h.join().expect("reader thread panicked"))
            .collect();
        (reader_results, tables)
    });
    for r in reader_results {
        r?;
    }

    let mut merged: BTreeMap<Vec<u8>, Stats> = BTreeMap::new();
    for table in &tables {
        for slot in table.iter().filter(|s| s.count != 0) {
            let name = &slot.name[..slot.name_len as usize];
            match merged.get_mut(name) {
                Some(stats) => {
                    stats.min = stats.min.min(slot.min);
                    stats.max = stats.max.max(slot.max);
                    stats.sum += slot.sum;
                    stats.count += slot.count as u64;
                }
                None => {
                    merged.insert(
                        name.to_vec(),
                        Stats {
                            min: slot.min,
                            max: slot.max,
                            sum: slot.sum,
                            count: slot.count as u64,
                        },
                    );
                }
            }
        }
    }
    Ok(merged)
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

fn main() -> io::Result<()> {
    let filename = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_FILE.to_string());

    let merged = process_file(&filename)?;

    let mut out = Vec::with_capacity(64 * merged.len() + 16);
    for (name, stats) in &merged {
        out.extend_from_slice(name);
        out.push(b';');
        push_tenths(&mut out, stats.min as i64);
        out.push(b';');
        push_tenths(&mut out, mean_tenths(stats.sum, stats.count));
        out.push(b';');
        push_tenths(&mut out, stats.max as i64);
        out.push(b'\n');
    }

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    writer.write_all(&out)?;
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_via_words(line: &[u8]) -> (i16, usize) {
        let mut padded = [0u8; 16];
        padded[..line.len()].copy_from_slice(line);
        let word = u64::from_le_bytes(padded[..8].try_into().unwrap());
        parse_temp(word)
    }

    #[test]
    fn parse_temp_exhaustive() {
        for v in -999i32..=999 {
            let line = format!("{}.{}\n", v / 10, (v % 10).abs());
            // canonical form only: -0.x must render as "-0.x"
            let line = if v < 0 && v > -10 {
                format!("-0.{}\n", (-v) % 10)
            } else {
                line
            };
            let bytes = line.as_bytes();
            let (parsed, len) = parse_via_words(bytes);
            assert_eq!(parsed as i32, v, "value for {:?}", line);
            assert_eq!(len, bytes.len(), "length for {:?}", line);
        }
    }

    #[test]
    fn semicolon_match_finds_first() {
        let w = u64::from_le_bytes(*b"ab;cd;ef");
        let m = semicolon_match(w);
        assert_eq!(m.trailing_zeros() as usize >> 3, 2);
        assert_eq!(semicolon_match(u64::from_le_bytes(*b"abcdefgh")), 0);
        // High-bit (UTF-8 continuation) bytes must not false-positive.
        assert_eq!(semicolon_match(u64::from_le_bytes([0xbb; 8])), 0);
        assert_eq!(semicolon_match(u64::from_le_bytes([0x3a; 8])), 0);
        assert_eq!(semicolon_match(u64::from_le_bytes([0x3c; 8])), 0);
    }

    #[test]
    fn mean_rounding_half_up() {
        assert_eq!(mean_tenths(15, 10), 2); // 1.5 -> 2 (half toward +inf)
        assert_eq!(mean_tenths(-15, 10), -1); // -1.5 -> -1
        assert_eq!(mean_tenths(14, 10), 1); // 1.4 -> 1
        assert_eq!(mean_tenths(-14, 10), -1); // -1.4 -> -1
        assert_eq!(mean_tenths(16, 10), 2);
        assert_eq!(mean_tenths(-16, 10), -2); // -1.6 -> -2
        assert_eq!(mean_tenths(0, 5), 0);
        assert_eq!(mean_tenths(999, 1), 999);
        assert_eq!(mean_tenths(-999, 1), -999);
    }

    #[test]
    fn push_tenths_formats() {
        let cases = [
            (0, "0.0"),
            (1, "0.1"),
            (-1, "-0.1"),
            (105, "10.5"),
            (-999, "-99.9"),
            (999, "99.9"),
            (100, "10.0"),
            (-5, "-0.5"),
        ];
        for (v, expected) in cases {
            let mut out = Vec::new();
            push_tenths(&mut out, v);
            assert_eq!(std::str::from_utf8(&out).unwrap(), expected);
        }
    }

    #[test]
    fn process_chunk_basic() {
        let data = b"Hamburg;12.0\nLisbon;-3.4\nHamburg;5.6\n";
        let mut buf = data.to_vec();
        buf.push(b';');
        buf.extend_from_slice(&[0u8; PAD]);
        let mut table = vec![EMPTY_SLOT; TABLE_LEN + 1]; // +1: over-read sentinel for names_equal
        let mut used = 0;
        process_chunk(&buf, 0, data.len(), &mut table, &mut used);
        assert_eq!(used, 2);
        let mut found = 0;
        for slot in table.iter().filter(|s| s.count != 0) {
            let name = &slot.name[..slot.name_len as usize];
            match name {
                b"Hamburg" => {
                    assert_eq!(slot.count, 2);
                    assert_eq!(slot.min, 56);
                    assert_eq!(slot.max, 120);
                    assert_eq!(slot.sum, 176);
                }
                b"Lisbon" => {
                    assert_eq!(slot.count, 1);
                    assert_eq!(slot.min, -34);
                }
                other => panic!("unexpected station {:?}", other),
            }
            found += 1;
        }
        assert_eq!(found, 2);
    }
}
