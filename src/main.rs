#![feature(portable_simd)]

use std::{
    collections::HashMap,
    fs::File,
    io,
    os::unix::fs::FileExt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

use crossbeam::thread;
use fxhash::FxBuildHasher;
use memchr::memrchr;
use std::simd::prelude::SimdPartialEq;
use std::simd::Simd;

const CHUNK_SIZE: u64 = 16 * 1024 * 1024;
const CHUNK_OVERLAP: u64 = 64;

use jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Debug, Clone, Copy)]
struct Records {
    count: u64,
    min: f64,
    max: f64,
    sum: f64,
}

impl Records {
    fn update(&mut self, temp: f64) {
        self.count += 1;
        self.sum += temp;
        if temp < self.min {
            self.min = temp;
        }
        if temp > self.max {
            self.max = temp;
        }
    }

    fn new(temp: f64) -> Self {
        Self {
            count: 1,
            min: temp,
            max: temp,
            sum: temp,
        }
    }

    fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }

    fn merge(&mut self, other: &Records) {
        self.count += other.count;
        self.sum += other.sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

fn parse_temp(bytes: &[u8]) -> Option<f64> {
    let temp_str = std::str::from_utf8(bytes).ok()?;
    temp_str.trim().parse::<f64>().ok()
}

fn process_chunk<'a>(chunk: &'a [u8]) -> fxhash::FxHashMap<&'a [u8], Records> {
    let mut map: fxhash::FxHashMap<&'a [u8], Records> = fxhash::FxHashMap::default();

    let mut start = 0;
    let len = chunk.len();

    while start < len {
        let end = match find_next_newline_simd(&chunk[start..]) {
            Some(pos) => start + pos,
            None => len,
        };

        let line = &chunk[start..end];
        if let Some(pos) = memchr::memchr(b';', line) {
            let station = &line[..pos];
            let temp_bytes = &line[pos + 1..];

            if let Some(temp) = parse_temp(temp_bytes) {
                map.entry(station)
                    .and_modify(|e| e.update(temp))
                    .or_insert_with(|| Records::new(temp));
            }
        }

        start = end + 1;
    }

    map
}

fn find_next_newline_simd(buffer: &[u8]) -> Option<usize> {
    let mut index = 0;
    let simd_size = 64;

    while index + simd_size <= buffer.len() {
        let bytes = Simd::<u8, 64>::from_slice(&buffer[index..index + simd_size]);
        let mask = bytes.simd_eq(Simd::splat(b'\n'));
        let bits = mask.to_bitmask();

        if bits != 0 {
            let pos = bits.trailing_zeros() as usize;
            return Some(index + pos);
        }

        index += simd_size;
    }

    for i in index..buffer.len() {
        if buffer[i] == b'\n' {
            return Some(i);
        }
    }

    None
}

fn process_file_parallel(filename: &str) -> io::Result<HashMap<String, Records, FxBuildHasher>> {
    let file = File::open(filename)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    let num_threads = num_cpus::get();
    let offset = AtomicU64::new(0);

    let global_map = Arc::new(Mutex::new(HashMap::with_hasher(FxBuildHasher::default())));

    thread::scope(|s| {
        for _ in 0..num_threads {
            let file = file.try_clone().unwrap();
            let global_map = Arc::clone(&global_map);
            let offset = &offset;

            s.spawn(move |_| {
                let mut buffer = vec![0u8; (CHUNK_SIZE + CHUNK_OVERLAP) as usize];

                loop {
                    let chunk_start = offset.fetch_add(CHUNK_SIZE, Ordering::SeqCst);
                    if chunk_start >= file_size {
                        break;
                    }

                    let read_start = if chunk_start == 0 {
                        0
                    } else {
                        chunk_start - CHUNK_OVERLAP
                    };

                    let read_size =
                        std::cmp::min(CHUNK_SIZE + CHUNK_OVERLAP, file_size - read_start);
                    let buffer = &mut buffer[..read_size as usize];

                    if let Err(e) = file.read_exact_at(buffer, read_start) {
                        eprintln!("Error reading file at position {}: {}", read_start, e);
                        break;
                    }

                    let mut chunk = &buffer[..];
                    if chunk_start != 0 {
                        if let Some(pos) = memrchr(b'\n', chunk) {
                            chunk = &chunk[pos + 1..];
                        } else {
                            continue;
                        }
                    }

                    if let Some(pos) = memrchr(b'\n', chunk) {
                        chunk = &chunk[..pos];
                    }

                    let local_map = process_chunk(chunk);

                    let mut global_map = global_map.lock().unwrap();
                    for (station_bytes, records) in local_map {
                        let station = String::from_utf8_lossy(station_bytes).to_string();
                        global_map
                            .entry(station)
                            .and_modify(|e: &mut Records| e.merge(&records))
                            .or_insert(records);
                    }
                }
            });
        }
    })
    .map_err(|_| io::Error::new(io::ErrorKind::Other, "Thread error"))?;

    let global_map = Arc::try_unwrap(global_map)
        .expect("More than one Arc pointer")
        .into_inner()
        .unwrap();

    Ok(global_map)
}

fn main() -> io::Result<()> {
    let filename = "measurements.txt";

    let stats_map = process_file_parallel(filename)?;

    let mut stations: Vec<_> = stats_map.keys().collect();
    stations.sort_unstable();

    for station in stations {
        let stats = &stats_map[station];
        println!(
            "{};{:.1};{:.1};{:.1}",
            station,
            stats.min,
            stats.mean(),
            stats.max
        );
    }

    Ok(())
}
