use std::{
  borrow::Borrow,
  collections::HashMap,
  fs::File,
  hash::{BuildHasherDefault, Hash, Hasher},
  io::{self, BufWriter, Write},
  num::NonZeroUsize,
  os::unix::fs::FileExt,
  thread,
};

const CHUNK_OVERLAP: u64 = 128;
const ESTIMATED_STATION_COUNT: usize = 2048;
const MAX_STATION_LEN: usize = 100;
const FX_HASH_MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Debug, Clone, Copy)]
struct Records {
  count: i64,
  min: i16,
  max: i16,
  sum: i64,
}

impl Records {
  fn update(&mut self, temp: i16) {
    self.count += 1;
    self.sum += i64::from(temp);
    self.min = self.min.min(temp);
    self.max = self.max.max(temp);
  }

  fn new(temp: i16) -> Self {
    Self {
      count: 1,
      min: temp,
      max: temp,
      sum: i64::from(temp),
    }
  }

  const fn mean_tenths(&self) -> i16 {
    let numerator = self.sum * 2 + self.count;
    let denominator = self.count * 2;
    let rounded = numerator.div_euclid(denominator);

    #[allow(clippy::cast_possible_truncation)]
    {
      rounded as i16
    }
  }

  fn mean(&self) -> f64 {
    f64::from(self.mean_tenths()) / 10.0
  }

  fn min_value(&self) -> f64 {
    f64::from(self.min) / 10.0
  }

  fn max_value(&self) -> f64 {
    f64::from(self.max) / 10.0
  }

  fn merge(&mut self, other: Self) {
    self.count += other.count;
    self.sum += other.sum;
    self.min = self.min.min(other.min);
    self.max = self.max.max(other.max);
  }
}

#[derive(Clone, Copy)]
struct StationKey {
  len: u8,
  bytes: [u8; MAX_STATION_LEN],
}

impl StationKey {
  fn from_slice(station: &[u8]) -> Option<Self> {
    if station.len() > MAX_STATION_LEN {
      return None;
    }

    let len = u8::try_from(station.len()).ok()?;
    let mut bytes = [0u8; MAX_STATION_LEN];
    bytes[..station.len()].copy_from_slice(station);

    Some(Self { len, bytes })
  }

  fn as_slice(&self) -> &[u8] {
    &self.bytes[..usize::from(self.len)]
  }
}

impl PartialEq for StationKey {
  fn eq(&self, other: &Self) -> bool {
    self.as_slice() == other.as_slice()
  }
}

impl Eq for StationKey {}

impl Hash for StationKey {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.as_slice().hash(state);
  }
}

impl Borrow<[u8]> for StationKey {
  fn borrow(&self) -> &[u8] {
    self.as_slice()
  }
}

#[derive(Default)]
struct FxHasher64 {
  hash: u64,
}

impl FxHasher64 {
  #[inline]
  const fn hash_word(hash: u64, word: u64) -> u64 {
    hash.rotate_left(5) ^ word
  }

  #[inline]
  const fn add_to_hash(&mut self, word: u64) {
    self.hash =
      Self::hash_word(self.hash, word).wrapping_mul(FX_HASH_MULTIPLIER);
  }
}

impl Hasher for FxHasher64 {
  fn finish(&self) -> u64 {
    self.hash
  }

  fn write(&mut self, bytes: &[u8]) {
    let mut index = 0;

    while index + 8 <= bytes.len() {
      let mut word_bytes = [0u8; 8];
      word_bytes.copy_from_slice(&bytes[index..index + 8]);
      self.add_to_hash(u64::from_ne_bytes(word_bytes));
      index += 8;
    }

    if index + 4 <= bytes.len() {
      let mut word_bytes = [0u8; 4];
      word_bytes.copy_from_slice(&bytes[index..index + 4]);
      self.add_to_hash(u64::from(u32::from_ne_bytes(word_bytes)));
      index += 4;
    }

    if index + 2 <= bytes.len() {
      let mut word_bytes = [0u8; 2];
      word_bytes.copy_from_slice(&bytes[index..index + 2]);
      self.add_to_hash(u64::from(u16::from_ne_bytes(word_bytes)));
      index += 2;
    }

    if index < bytes.len() {
      self.add_to_hash(u64::from(bytes[index]));
    }
  }

  fn write_u8(&mut self, i: u8) {
    self.add_to_hash(u64::from(i));
  }

  fn write_u16(&mut self, i: u16) {
    self.add_to_hash(u64::from(i));
  }

  fn write_u32(&mut self, i: u32) {
    self.add_to_hash(u64::from(i));
  }

  fn write_u64(&mut self, i: u64) {
    self.add_to_hash(i);
  }

  #[allow(clippy::cast_possible_truncation)]
  fn write_usize(&mut self, i: usize) {
    self.add_to_hash(i as u64);
  }
}

type StationMap = HashMap<StationKey, Records, BuildHasherDefault<FxHasher64>>;

#[inline]
fn worker_count() -> usize {
  let default_threads =
    thread::available_parallelism().map_or(1, NonZeroUsize::get);

  std::env::var("FAST_1BRC_THREADS")
    .ok()
    .and_then(|value| value.parse::<usize>().ok())
    .filter(|&value| value > 0)
    .unwrap_or(default_threads)
}

#[inline]
fn chunk_size_bytes() -> u64 {
  std::env::var("FAST_1BRC_CHUNK_MB")
    .ok()
    .and_then(|value| value.parse::<u64>().ok())
    .filter(|&value| value > 0)
    .map_or(4 * 1024 * 1024, |mb| mb * 1024 * 1024)
}

fn update_station(map: &mut StationMap, station: &[u8], temp: i16) {
  if let Some(entry) = map.get_mut(station) {
    entry.update(temp);
    return;
  }

  if let Some(key) = StationKey::from_slice(station) {
    map.insert(key, Records::new(temp));
  }
}

fn process_chunk(chunk: &[u8], map: &mut StationMap) {
  let mut index = 0;

  while index < chunk.len() {
    let station_start = index;
    while index < chunk.len() && chunk[index] != b';' {
      index += 1;
    }

    if index >= chunk.len() {
      break;
    }

    let station = &chunk[station_start..index];
    index += 1;

    let negative = chunk[index] == b'-';
    if negative {
      index += 1;
    }

    let mut value = i16::from(chunk[index] - b'0');
    index += 1;

    if chunk[index] != b'.' {
      value = value * 10 + i16::from(chunk[index] - b'0');
      index += 1;
    }

    index += 1;
    value = value * 10 + i16::from(chunk[index] - b'0');
    index += 1;

    update_station(map, station, if negative { -value } else { value });

    if index < chunk.len() {
      index += 1;
    }
  }
}

#[inline]
fn find_first_byte(bytes: &[u8], needle: u8) -> Option<usize> {
  let mut index = 0;
  while index < bytes.len() {
    if bytes[index] == needle {
      return Some(index);
    }
    index += 1;
  }

  None
}

#[inline]
fn find_last_byte(bytes: &[u8], needle: u8) -> Option<usize> {
  let mut index = bytes.len();
  while index > 0 {
    index -= 1;
    if bytes[index] == needle {
      return Some(index);
    }
  }

  None
}

fn usize_from_u64(value: u64, context: &'static str) -> io::Result<usize> {
  usize::try_from(value).map_err(|_| {
    io::Error::new(
      io::ErrorKind::InvalidInput,
      format!("{context} does not fit into usize on this platform"),
    )
  })
}

fn process_worker(
  thread_file: &File,
  file_size: u64,
  chunk_size: u64,
  chunk_with_overlap_usize: usize,
  start_chunk: u64,
  end_chunk: u64,
) -> io::Result<StationMap> {
  let mut buffer = vec![0u8; chunk_with_overlap_usize];
  let mut local_map = HashMap::with_capacity_and_hasher(
    ESTIMATED_STATION_COUNT,
    BuildHasherDefault::default(),
  );

  for chunk_index in start_chunk..end_chunk {
    let chunk_start = chunk_index * chunk_size;
    if chunk_start >= file_size {
      break;
    }

    let read_start = if chunk_start == 0 {
      0
    } else {
      chunk_start - CHUNK_OVERLAP
    };

    let prefix_len = chunk_start - read_start;
    let read_size =
      std::cmp::min(chunk_size + prefix_len, file_size - read_start);
    let read_size_usize = usize_from_u64(read_size, "read size")?;
    let buffer = &mut buffer[..read_size_usize];

    let bytes_read = thread_file.read_at(buffer, read_start)?;
    if bytes_read == 0 {
      break;
    }

    let mut chunk = &buffer[..bytes_read];
    if prefix_len > 0 {
      let prefix_len_usize = usize_from_u64(prefix_len, "prefix size")?;
      let prefix_end = std::cmp::min(prefix_len_usize, chunk.len());
      let chunk_start_pos = find_last_byte(&chunk[..prefix_end], b'\n')
        .or_else(|| find_first_byte(chunk, b'\n'));

      if let Some(pos) = chunk_start_pos {
        chunk = &chunk[pos + 1..];
      } else {
        continue;
      }
    }

    let chunk_end = read_start
      + u64::try_from(bytes_read).map_err(|_| {
        io::Error::new(
          io::ErrorKind::InvalidInput,
          "bytes read does not fit into u64 on this platform",
        )
      })?;

    if chunk_end < file_size {
      if let Some(pos) = find_last_byte(chunk, b'\n') {
        chunk = &chunk[..pos];
      } else {
        continue;
      }
    }

    if chunk.is_empty() {
      continue;
    }

    process_chunk(chunk, &mut local_map);
  }

  Ok(local_map)
}

fn chunk_range(
  total_chunks: u64,
  worker_count: usize,
  worker_index: usize,
) -> io::Result<(u64, u64)> {
  let worker_count_u64 = u64::try_from(worker_count).map_err(|_| {
    io::Error::new(
      io::ErrorKind::InvalidInput,
      "worker count does not fit into u64 on this platform",
    )
  })?;
  let worker_index_u64 = u64::try_from(worker_index).map_err(|_| {
    io::Error::new(
      io::ErrorKind::InvalidInput,
      "worker index does not fit into u64 on this platform",
    )
  })?;

  let chunks_per_worker = total_chunks / worker_count_u64;
  let remainder = total_chunks % worker_count_u64;

  let start_chunk = chunks_per_worker
    .checked_mul(worker_index_u64)
    .and_then(|base| base.checked_add(worker_index_u64.min(remainder)))
    .ok_or_else(|| {
      io::Error::new(io::ErrorKind::InvalidInput, "chunk range overflow")
    })?;
  let extra_chunk = u64::from(worker_index_u64 < remainder);
  let end_chunk = start_chunk
    .checked_add(chunks_per_worker)
    .and_then(|value| value.checked_add(extra_chunk))
    .ok_or_else(|| {
      io::Error::new(io::ErrorKind::InvalidInput, "chunk range overflow")
    })?;

  Ok((start_chunk, end_chunk))
}

fn process_file_parallel(filename: &str) -> io::Result<StationMap> {
  let file = File::open(filename)?;
  let file_size = file.metadata()?.len();

  let num_threads = worker_count();
  let chunk_size = chunk_size_bytes();
  let chunk_with_overlap =
    chunk_size.checked_add(CHUNK_OVERLAP).ok_or_else(|| {
      io::Error::new(io::ErrorKind::InvalidInput, "chunk size overflow")
    })?;
  let chunk_with_overlap_usize =
    usize_from_u64(chunk_with_overlap, "chunk with overlap size")?;
  let total_chunks = file_size.div_ceil(chunk_size);

  let mut local_maps = Vec::with_capacity(num_threads);

  thread::scope(|scope| -> io::Result<()> {
    let mut workers = Vec::with_capacity(num_threads);
    for thread_index in 0..num_threads {
      let thread_file = file.try_clone()?;
      let (start_chunk, end_chunk) =
        chunk_range(total_chunks, num_threads, thread_index)?;

      workers.push(scope.spawn(move || {
        process_worker(
          &thread_file,
          file_size,
          chunk_size,
          chunk_with_overlap_usize,
          start_chunk,
          end_chunk,
        )
      }));
    }

    for worker in workers {
      let local_map = worker
        .join()
        .map_err(|_| io::Error::other("worker thread panicked"))??;
      local_maps.push(local_map);
    }

    Ok(())
  })?;

  let mut global_map: StationMap = HashMap::with_capacity_and_hasher(
    ESTIMATED_STATION_COUNT,
    BuildHasherDefault::default(),
  );

  for local_map in local_maps {
    for (station, records) in local_map {
      if let Some(entry) = global_map.get_mut(station.as_slice()) {
        entry.merge(records);
      } else {
        global_map.insert(station, records);
      }
    }
  }

  Ok(global_map)
}

fn main() -> io::Result<()> {
  let stats_map = process_file_parallel("measurements.txt")?;

  let mut stations: Vec<_> = stats_map.iter().collect();
  stations.sort_unstable_by(|(station_a, _), (station_b, _)| {
    station_a.as_slice().cmp(station_b.as_slice())
  });

  let stdout = io::stdout();
  let mut writer = BufWriter::new(stdout.lock());

  for (station_key, stats) in stations {
    let station =
      std::str::from_utf8(station_key.as_slice()).map_err(|error| {
        io::Error::new(
          io::ErrorKind::InvalidData,
          format!("invalid UTF-8 station name: {error}"),
        )
      })?;

    writeln!(
      writer,
      "{};{:.1};{:.1};{:.1}",
      station,
      stats.min_value(),
      stats.mean(),
      stats.max_value()
    )?;
  }

  writer.flush()?;

  Ok(())
}
