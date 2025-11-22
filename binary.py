# Minimal, self-contained pipeline: Shannon entropy -> arithmetic coding -> ZEP-compatible binary tokens.
from fractions import Fraction
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple, Any
import math
import json
import itertools
import copy
import numpy as np

# If you already have zep.py on PYTHONPATH, you can import compute_velocity and ZPECharge:

# For this example we keep BASE_CHUNK=4 to match your zep.py
BASE_CHUNK = 4

# -------------------------
# Utilities: Shannon entropy
# -------------------------
def shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy in bits-per-symbol for a byte sequence."""
    if not data:
        return 0.0
    freq = Counter(data)
    length = len(data)
    ent = -sum((count/length) * math.log2(count/length) for count in freq.values())
    return ent


# -------------------------
# Bit packing helpers
# -------------------------
def pack_bits_from_iterator(bit_iter):
    """Pack an iterator of bits (0/1 ints) into bytes (MSB-first) and return (bytes, bit_length)."""
    ba = bytearray()
    cur = 0
    filled = 0
    total = 0
    for b in bit_iter:
        cur = (cur << 1) | (int(b) & 1)
        filled += 1
        total += 1
        if filled == 8:
            ba.append(cur)
            cur = 0
            filled = 0
    if filled:
        cur = cur << (8 - filled)
        ba.append(cur)
    return bytes(ba), total


def bits_from_bytes(packed_bytes: bytes, bit_length: int):
    """Yield bits (0/1) MSB-first from packed_bytes up to bit_length."""
    for i in range(bit_length):
        b = packed_bytes[i // 8]
        shift = 7 - (i % 8)
        yield (b >> shift) & 1


def bytes_to_base_chunks(packed_bytes: bytes, bit_length: int, base_chunk: int):
    """Return list of base_chunk-sized bit strings (pad with '0') from packed bytes."""
    if base_chunk <= 0:
        raise ValueError("base_chunk must be > 0")
    pad_len = (-bit_length) % base_chunk
    total = bit_length + pad_len
    chunks = []
    for start in range(0, total, base_chunk):
        pieces = []
        for j in range(base_chunk):
            bit_index = start + j
            if bit_index < bit_length:
                b = packed_bytes[bit_index // 8]
                shift = 7 - (bit_index % 8)
                pieces.append(str((b >> shift) & 1))
            else:
                pieces.append('0')
        chunks.append(''.join(pieces))
    return chunks

# -------------------------
# Arithmetic coder (rational Interval approach using Fraction)
# -------------------------
def _build_cumulative(freq: Dict[int, int]) -> Tuple[Dict[int, Tuple[int,int]], int]:
    """
    Build cumulative counts table.
    Returns dict symbol -> (cum_low, cum_high) and total_count
    """
    items = sorted(freq.items())
    cum = 0
    table = {}
    for sym, f in items:
        table[sym] = (cum, cum + f)
        cum += f
    # Add EOF symbol with count 1 to make the coding prefix-free (optional)
    return table, cum
    return table, cum


#!/usr/bin/env python3
"""
Cumulative sum objects and Fenwick trees for fast operations
============================================================

Fenwick tree and CumulativeSum classes designed to work with adaptive models.

"""


class FenwickTree:
    """A data structure for maintaining cumulative (prefix) sums.

    All operations are O(log n).
    """

    def __init__(self, frequencies):
        """Initializes n frequencies to zero."""
        self._v = list(frequencies)

        # Initialize in O(n) with specified frequencies.
        for idx in range(1, len(self) + 1):
            parent_idx = idx + (idx & -idx)  # parent in update tree
            if parent_idx <= len(self):
                self._v[parent_idx - 1] += self._v[idx - 1]

    def __len__(self):
        return len(self._v)

    def prefix_sum(self, stop):
        """Returns sum of first elements (sum up to *stop*, exclusive)."""
        if stop <= 0 or stop > len(self):
            raise IndexError("index out of range")
        _sum = 0
        while stop > 0:
            _sum += self._v[stop - 1]
            stop &= stop - 1
        return _sum

    def range_sum(self, start, stop):
        if start < 0 or start >= len(self):
            raise IndexError("index out of range")
        if stop <= start or stop > len(self):
            raise IndexError("index out of range")
        result = self.prefix_sum(stop)
        if start > 0:
            result -= self.prefix_sum(start)
        return result

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = idx % len(self)
            return self.range_sum(idx, idx + 1)
        else:
            raise IndexError(f"Indexing only works with integers, got {idx}")

    def frequencies(self):
        _frequencies = [0] * len(self)
        for idx in range(1, len(self) + 1):
            _frequencies[idx - 1] += self._v[idx - 1]
            parent_idx = idx + (idx & -idx)
            if parent_idx <= len(self):
                _frequencies[parent_idx - 1] -= self._v[idx - 1]
        return _frequencies

    def add(self, idx, k):
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        idx += 1
        while idx <= len(self):
            self._v[idx - 1] += k
            idx += idx & -idx

    def __setitem__(self, idx, value):
        self.add(idx, value - self[idx])

    def bisect_left(self, value):
        # Binary-lifting search using largest power of two <= n
        n = len(self)
        if n == 0:
            return 0
        bitmask = 1 << (n.bit_length() - 1)
        idx = 0
        # 1-based logic adapted to 0-based storage
        while bitmask:
            t = idx + bitmask
            if t <= n and value > self._v[t - 1]:
                value -= self._v[t - 1]
                idx = t
            bitmask >>= 1
        # idx is number of elements with cumulative sum < original value
        return idx

    def __eq__(self, other):
        return isinstance(other, FenwickTree) and self._v == other._v


class NaiveCumulativeSum:
    """Cumulative sum with slow asymptotic performance."""

    def __init__(self, frequencies, update=True):
        self.frequencies = dict(frequencies)
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))
        self.update = update

    def get_low_high(self, symbol):
        return self.ranges[symbol]

    def add_count(self, symbol, value):
        if self.update:
            self.frequencies[symbol] += value
            self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    def total_count(self):
        return sum(self.frequencies.values())

    def reset(self):
        self.frequencies = {frequency: 1 for frequency in self.frequencies}
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    @staticmethod
    def ranges_from_frequencies(frequencies):
        cumsum = 0
        for symbol, frequency in sorted(frequencies.items()):
            yield (symbol, (cumsum, cumsum + frequency))
            cumsum += frequency

    def search_ranges(self, value):
        for symbol, (low, high) in self.ranges.items():
            if low <= value < high:
                return symbol
        raise ValueError("Could not locate value in ranges.")


class CumulativeSum:
    """Cumulative sum with fast asymptotic performance (Fenwick-based)."""

    def __init__(self, frequencies, update=True):
        symbols = sorted(frequencies.keys())
        self.idx_to_symbol = dict(enumerate(symbols))
        self.symbol_to_idx = {s: i for (i, s) in self.idx_to_symbol.items()}
        self.fenwick_tree = FenwickTree([frequencies[s] for s in symbols])
        self.update = update

    def get_low_high(self, symbol):
        idx = self.symbol_to_idx[symbol]
        if idx == 0:
            return (0, self.fenwick_tree[idx])
        sum_upto = self.fenwick_tree.prefix_sum(idx)
        return (sum_upto, sum_upto + self.fenwick_tree[idx])

    def add_count(self, symbol, value):
        if self.update:
            idx = self.symbol_to_idx[symbol]
            self.fenwick_tree.add(idx, value)

    def total_count(self):
        return self.fenwick_tree.prefix_sum(len(self.fenwick_tree))

    def reset(self):
        self.fenwick_tree = FenwickTree([1] * len(self.fenwick_tree))

    def search_ranges(self, value):
        # FenwickTree.bisect_left expects a 1-based cumulative target (1..total_count).
        # The decoder computes a scaled_value in range [0, total_count-1], so add 1.
        idx = self.fenwick_tree.bisect_left(value + 1)
        # Clamp to valid index range to be defensive against off-by-one cases.
        if idx >= len(self.idx_to_symbol):
            idx = len(self.idx_to_symbol) - 1
        return self.idx_to_symbol[idx]

    def clone(self):
        """Create a lightweight clone of this CumulativeSum without deepcopy.

        Reconstructs the frequencies from the fenwick tree and builds a new
        CumulativeSum with the same update flag. This avoids expensive deepcopy.
        """
        freqs = {self.idx_to_symbol[i]: f for i, f in enumerate(self.fenwick_tree.frequencies())}
        return CumulativeSum(freqs, update=self.update)


class BitQueue:
    """A queue to keep track of bits to follow.

    Minimal implementation matching the earlier encoder example.
    """

    def __init__(self):
        self.bits_to_follow = 0

    def __iadd__(self, bits):
        self.bits_to_follow += bits
        return self

    def bit_plus_follow(self, bit):
        yield bit
        for _ in range(self.bits_to_follow):
            yield int(not bit)
        self.bits_to_follow = 0


# ---------------------------------------------------------------------------
# ArithmeticEncoder (wired from provided implementation)
# ---------------------------------------------------------------------------
class ArithmeticEncoder:
    def __init__(self, frequencies, *, bits=32, verbose=0, EOM="<EOM>"):
        self.EOM = EOM
        # frequencies expected as dict mapping symbol->int
        if isinstance(frequencies, dict):
            self.frequencies = dict(frequencies)
            self.cumsum = CumulativeSum(self.frequencies, update=False)
        elif isinstance(frequencies, (list, set)):
            # dynamic model
            self.frequencies = list(frequencies)
            freqs = {s: 1 for s in self.frequencies}
            self.cumsum = CumulativeSum(freqs, update=True)
        else:
            raise TypeError("frequencies must be dict or list/set")

        self.bits = bits
        self.verbose = verbose
        self.TOP_VALUE = (1 << self.bits) - 1
        self.FIRST_QUARTER = (self.TOP_VALUE >> 2) + 1
        self.HALF = self.FIRST_QUARTER * 2
        self.THIRD_QUARTER = self.FIRST_QUARTER * 3

        if self.cumsum.total_count() > int((self.TOP_VALUE + 1) / 4) + 1:
            raise Exception("Insufficient precision to encode low-probability symbols. Increase bits.")

    def encode(self, iterable):
        # Lightweight clone of cumulative model to avoid deepcopy overhead
        cumsum = self.cumsum.clone() if hasattr(self.cumsum, 'clone') else copy.deepcopy(self.cumsum)
        bit_queue = BitQueue()
        low = 0
        high = self.TOP_VALUE

        for symbol in iterable:
            range_ = high - low + 1
            if range_ < cumsum.total_count():
                raise Exception("Insufficient precision to encode low-probability symbols.")
            symbol_low, symbol_high = cumsum.get_low_high(symbol)
            total_count = cumsum.total_count()
            high = low + int(range_ * symbol_high / total_count) - 1
            low = low + int(range_ * symbol_low / total_count)

            while True:
                if high < self.HALF:
                    yield from bit_queue.bit_plus_follow(bit=0)
                elif low >= self.HALF:
                    yield from bit_queue.bit_plus_follow(bit=1)
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                    bit_queue += 1
                else:
                    break
                low = 2 * low
                high = 2 * high + 1

            cumsum.add_count(symbol, 1)

        # finish
        bit_queue += 1
        yield from bit_queue.bit_plus_follow(int(low >= self.FIRST_QUARTER))

    def decode(self, iterable):
        # Lightweight clone to avoid deepcopy overhead
        cumsum = self.cumsum.clone() if hasattr(self.cumsum, 'clone') else copy.deepcopy(self.cumsum)
        low = 0
        value = 0
        high = self.TOP_VALUE

        iterable = enumerate(itertools.chain(iter(iterable), itertools.repeat(0)), 1)
        first_bits = itertools.islice(iterable, self.bits)
        i = 0
        for i, input_bit in first_bits:
            value = (value << 1) + input_bit

        while True:
            range_ = high - low + 1
            total_count = cumsum.total_count()
            # Use integer scaled value to match encoder arithmetic
            scaled_value = int(((value - low + 1) * total_count - 1) // range_)
            symbol = cumsum.search_ranges(scaled_value)
            yield symbol

            symbol_low, symbol_high = cumsum.get_low_high(symbol)
            high = low + int(range_ * symbol_high / total_count) - 1
            low = low + int(range_ * symbol_low / total_count)
            cumsum.add_count(symbol, 1)

            if symbol == self.EOM:
                break

            while True:
                if high < self.HALF:
                    pass
                elif low >= self.HALF:
                    value -= self.HALF
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    value -= self.FIRST_QUARTER
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break

                low = 2 * low
                high = 2 * high + 1
                i, input_bit = next(iterable, (i + 1, 0))
                value = 2 * value + input_bit

# -------------------------
# Chunking pipeline
# -------------------------
def segment_bytes(data: bytes, chunk_size: int = 2048, overlap: int = 0) -> List[Tuple[int, bytes]]:
    """Split data into chunks with optional overlap (overlap in bytes).

    Returns a list of (start_offset, chunk_bytes) so callers can reconstruct absolute
    positions when overlap > 0.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    chunks = []
    i = 0
    n = len(data)
    step = chunk_size - overlap
    while i < n:
        end = min(n, i + chunk_size)
        chunks.append((i, data[i:end]))
        if end == n:
            break
        i += step
    return chunks

# -------------------------
# Map compressed bits -> ZEP-compatible tokens
# -------------------------
def bits_to_base_chunks(bitstr: str, base_chunk: int = BASE_CHUNK) -> List[str]:
    """Split bitstring into base_chunk-sized pieces (pad with zeros at end)."""
    if base_chunk <= 0:
        raise ValueError("base_chunk must be > 0")
    # pad to multiple of base_chunk
    pad_len = (-len(bitstr)) % base_chunk
    if pad_len:
        bitstr = bitstr + ('0' * pad_len)
    return [bitstr[i:i+base_chunk] for i in range(0, len(bitstr), base_chunk)]

def chunk_bits_to_zep_tokens(chunks_bits: List[str]) -> List[Dict[str,Any]]:
    """
    Convert base_chunk-sized bit groups into a list of ZEP-token-like dicts.
    We produce a small dict describing how to realize the token as a ZPECharge:
      { 'canonical_symbol': '+', 'canonical_binary': '0000', 'overlays': [ ... ] }
    We'll use canonical '+' for base and set canonical_binary to the actual chunk bits.
    """
    tokens = []
    for b in chunks_bits:
        token = {
            'canonical_symbol': '+',   # use a base symbol that exists in CARDINAL_CHARGES
            'canonical_binary': b,     # override the canonical to represent arbitrary pattern
            'overlays': []             # overlays can be used to represent subcharge overlaps later
        }
        tokens.append(token)
    return tokens

# -------------------------
# High-level API
# -------------------------
def binarify_data(data: bytes, chunk_size: int = 2048, overlap: int = 0, base_chunk: int = BASE_CHUNK):
    """
    Full pipeline:
      - split data
      - compute entropy per chunk
      - arithmetic encode each chunk and keep its freq table and symbol count
      - convert compressed bitstring into base_chunk pieces and produce zep-token metadata
    Returns a manifest dict with everything needed to reconstruct exactly.
    """
    chunks = segment_bytes(data, chunk_size=chunk_size, overlap=overlap)
    manifest = {
        'chunk_size': chunk_size,
        'overlap': overlap,
        'base_chunk': base_chunk,
        'chunks': []
    }

    for idx, (start, c) in enumerate(chunks):
        ent = shannon_entropy(c)
        # Prepare symbols as integers and add an explicit EOM marker (-1)
        EOM = -1
        symbols = [int(b) for b in c] + [EOM]
        # Build frequency table (include EOM with count 1)
        freq = Counter(symbols)
        freq_table = {int(k): int(v) for k, v in freq.items()}

        # Choose bits width conservatively based on total counts
        total_count = sum(freq_table.values())
        bits_width = max(16, math.ceil(math.log2(max(2, total_count * 4))))
        # Create encoder and encode, packing bits into bytes for efficiency
        encoder = ArithmeticEncoder(frequencies=freq_table, bits=bits_width, EOM=EOM)
        packed_bytes, bit_length = pack_bits_from_iterator(encoder.encode(symbols))
        base_pieces = bytes_to_base_chunks(packed_bytes, bit_length, base_chunk)
        tokens = chunk_bits_to_zep_tokens(base_pieces)
        
        # Create element-level records for each byte in this chunk
        elements = []
        for element_offset in range(len(c)):
            absolute_offset = start + element_offset
            byte_value = int(c[element_offset])
            # Generate deterministic element signature: dataset_id will be added by caller
            # For now use chunk_index:element_offset format
            element_signature = f"chunk_{idx}:elem_{element_offset}"
            elements.append({
                'element_index': element_offset,
                'absolute_offset': absolute_offset,
                'value': byte_value,
                'element_signature': element_signature,
                'chunk_index': idx
            })
        
        chunk_record = {
            'index': idx,
            'start': start,
            'length_bytes': len(c),
            'entropy': ent,
            'freq_table': freq_table,
            # number of original payload symbols (bytes) in this chunk
            'symbol_count': len(c),
            'compressed_bit_length': bit_length,
            'compressed_bits': packed_bytes,  # packed bytes (MSB-first), length in compressed_bit_length
            'base_chunks': base_pieces,
            'zep_tokens': tokens,
            'elements': elements  # Add element-level records
        }
        manifest['chunks'].append(chunk_record)
    return manifest

def reconstruct_from_manifest(manifest: Dict[str,Any]) -> bytes:
    """Reconstruct original data from the manifest produced by binarify_data.

    This implementation uses the recorded 'start' offsets per chunk and writes each
    decoded payload into its original absolute position. It detects conflicting
    overlapping writes (raises ValueError) to help catch encoding/decoding bugs.
    """
    if not manifest.get('chunks'):
        return b''

    # Determine final output size
    final_size = 0
    for chunk in manifest['chunks']:
        start = int(chunk.get('start', 0))
        length = int(chunk.get('length_bytes', 0))
        final_size = max(final_size, start + length)

    out = bytearray(b'\x00' * final_size)
    written = bytearray(b'\x00' * final_size)  # flags for written positions

    for chunk in manifest['chunks']:
        packed = chunk['compressed_bits']
        bit_length = int(chunk.get('compressed_bit_length', 0))
        freq_table = {int(k):int(v) for k,v in chunk['freq_table'].items()}
        start = int(chunk.get('start', 0))
        expected_len = int(chunk.get('length_bytes', 0))

        # Use the same EOM marker we encoded with (-1)
        EOM = -1
        # Choose bits width similar to encoding
        total_count = sum(freq_table.values())
        bits_width = max(16, math.ceil(math.log2(max(2, total_count * 4))))
        encoder = ArithmeticEncoder(frequencies=freq_table, bits=bits_width, EOM=EOM)
        # Create bit iterator directly from packed bytes (fast, no string parsing)
        bit_iter = bits_from_bytes(packed, bit_length)

        # Stream decode directly into output buffer to avoid materializing a large list
        i_local = 0
        for symbol in encoder.decode(bit_iter):
            if symbol == EOM:
                break
            # write into out directly
            pos = start + i_local
            if pos < 0 or pos >= final_size:
                raise IndexError("Decoded write out of bounds")
            if written[pos]:
                if out[pos] != symbol:
                    raise ValueError(f"Overlap conflict at pos {pos}: existing=0x{out[pos]:02x} new=0x{symbol:02x}")
            else:
                out[pos] = symbol
                written[pos] = 1
            i_local += 1

        payload_len = i_local

        if payload_len != expected_len:
            raise ValueError(f"Decoded payload length {payload_len} != expected {expected_len} for chunk starting at {start}")

    return bytes(out)

def extract_4d_features(manifest, data):
    """Extract 4D features with content-aware metadata linking."""
    features = []
    chunk_metadata = []
    
    for i, chunk in enumerate(manifest['chunks']):
        # Extract original 4D features
        entropy = chunk['entropy']
        compression_ratio = chunk['compressed_bit_length'] / (chunk['length_bytes'] * 8)
        symbol_diversity = len(chunk['freq_table'])
        
        if chunk['freq_table']:
            most_frequent_symbol = max(chunk['freq_table'], key=chunk['freq_table'].get)
        else:
            most_frequent_symbol = 0
            
        # Add content-aware features
        start_pos = chunk['start']
        chunk_length = chunk['length_bytes']
        chunk_data = data[start_pos:start_pos + chunk_length]
        
        # Content fingerprints
        if chunk_length > 0:
            text_ratio = sum(1 for b in chunk_data if 32 <= b <= 126) / len(chunk_data)
            newline_count = chunk_data.count(10)  # \n
            uppercase_ratio = sum(1 for b in chunk_data if 65 <= b <= 90) / len(chunk_data)
        else:
            text_ratio = 0.0
            newline_count = 0
            uppercase_ratio = 0.0
        
        features.append([entropy, compression_ratio, symbol_diversity, text_ratio])
        
        # Store reconstruction metadata with element-level information
        chunk_metadata.append({
            'chunk_index': i,
            'chunk_id': f"chunk_{i}",
            'start_position': start_pos,
            'length': chunk_length,
            'byte_range': (start_pos, start_pos + chunk_length),
            'freq_table': chunk['freq_table'],
            'compressed_bits': chunk['compressed_bits'],
            'compressed_bit_length': chunk['compressed_bit_length'],
            'content': chunk_data,  
            'most_frequent_symbol': most_frequent_symbol,
            'newline_count': newline_count,
            'uppercase_ratio': uppercase_ratio,
            'elements': chunk.get('elements', [])  # Include element-level records
        })
    
    return np.array(features), chunk_metadata


def _decode_chunk_from_metadata(meta):
    """Decode a single chunk using stored arithmetic coding metadata."""
    if not meta:
        raise ValueError("Chunk metadata is required for decoding")

    freq_table = {int(k): int(v) for k, v in meta['freq_table'].items()}
    total_count = sum(freq_table.values())
    if total_count <= 0:
        return b''

    bits_width = max(16, math.ceil(math.log2(max(2, total_count * 4))))
    encoder = ArithmeticEncoder(frequencies=freq_table, bits=bits_width, EOM=-1)
    bit_iter = bits_from_bytes(meta['compressed_bits'], meta['compressed_bit_length'])

    decoded = bytearray()
    for symbol in encoder.decode(bit_iter):
        if symbol == -1:
            break
        if symbol < 0 or symbol > 255:
            raise ValueError(f"Decoded symbol {symbol} out of byte range")
        decoded.append(symbol)

    return bytes(decoded)


def create_reconstruction_registry():
    """Return an empty registry used to track datasets, chunks, and byte ranges."""
    return {}


def register_dataset(registry, dataset_id, manifest, chunk_metadata):
    """Register a dataset with its manifest and chunk metadata for hierarchical lookup."""
    if dataset_id in registry:
        raise ValueError(f"Dataset '{dataset_id}' already registered")

    chunk_lookup = {meta['chunk_index']: dict(meta) for meta in chunk_metadata}
    total_length = 0
    for meta in chunk_metadata:
        start = meta['start_position']
        end = start + meta['length']
        total_length = max(total_length, end)

    registry[dataset_id] = {
        'dataset_id': dataset_id,
        'manifest': manifest,
        'chunk_metadata': chunk_lookup,
        'chunk_order': sorted(chunk_lookup.keys()),
        'total_chunks': len(chunk_lookup),
        'total_length': total_length
    }
    return registry[dataset_id]


def get_dataset_entry(registry, dataset_id):
    """Retrieve the registered dataset entry."""
    if dataset_id not in registry:
        raise KeyError(f"Dataset '{dataset_id}' is not registered")
    return registry[dataset_id]


def reconstruct_dataset_from_registry(registry, dataset_id):
    """Reconstruct the entire dataset identified by dataset_id."""
    entry = get_dataset_entry(registry, dataset_id)
    return reconstruct_from_manifest(entry['manifest'])


def reconstruct_chunk_from_registry(registry, dataset_id, chunk_index):
    """Reconstruct a specific chunk from the registry."""
    entry = get_dataset_entry(registry, dataset_id)
    if chunk_index not in entry['chunk_metadata']:
        raise KeyError(f"Chunk {chunk_index} not found in dataset '{dataset_id}'")
    return _decode_chunk_from_metadata(entry['chunk_metadata'][chunk_index])


def locate_chunk_for_byte(registry, dataset_id, byte_offset):
    """Return (chunk_index, metadata) for the chunk covering a specific byte offset."""
    entry = get_dataset_entry(registry, dataset_id)
    if byte_offset < 0 or byte_offset >= entry['total_length']:
        raise IndexError(f"Byte offset {byte_offset} out of bounds for dataset '{dataset_id}'")

    for chunk_index in entry['chunk_order']:
        meta = entry['chunk_metadata'][chunk_index]
        start = meta['start_position']
        end = start + meta['length']
        if start <= byte_offset < end:
            return chunk_index, meta

    raise IndexError(f"No chunk covers byte offset {byte_offset} in dataset '{dataset_id}'")


def reconstruct_byte_range(registry, dataset_id, start_byte, end_byte):
    """Reconstruct a byte slice [start_byte, end_byte) from the dataset."""
    if start_byte < 0:
        raise ValueError("start_byte must be >= 0")
    if end_byte is not None and end_byte < start_byte:
        raise ValueError("end_byte must be >= start_byte")

    entry = get_dataset_entry(registry, dataset_id)
    dataset_length = entry['total_length']
    if end_byte is None or end_byte > dataset_length:
        end_byte = dataset_length

    if start_byte >= dataset_length:
        return b''

    result = bytearray()
    for chunk_index in entry['chunk_order']:
        meta = entry['chunk_metadata'][chunk_index]
        chunk_start = meta['start_position']
        chunk_end = chunk_start + meta['length']

        if chunk_end <= start_byte:
            continue
        if chunk_start >= end_byte:
            break

        chunk_bytes = _decode_chunk_from_metadata(meta)
        slice_start = max(0, start_byte - chunk_start)
        slice_end = min(meta['length'], end_byte - chunk_start)
        if slice_start < slice_end:
            result.extend(chunk_bytes[slice_start:slice_end])

    return bytes(result)


def get_chunk_metadata(registry, dataset_id, chunk_index):
    """Return metadata for a specific chunk."""
    entry = get_dataset_entry(registry, dataset_id)
    if chunk_index not in entry['chunk_metadata']:
        raise KeyError(f"Chunk {chunk_index} not found in dataset '{dataset_id}'")
    return entry['chunk_metadata'][chunk_index]


def get_byte_metadata(registry, dataset_id, byte_offset):
    """Return hierarchical metadata describing a specific byte location."""
    chunk_index, meta = locate_chunk_for_byte(registry, dataset_id, byte_offset)
    return {
        'dataset_id': dataset_id,
        'chunk_index': chunk_index,
        'chunk_start': meta['start_position'],
        'chunk_length': meta['length'],
        'relative_offset': byte_offset - meta['start_position'],
        'compressed_bit_length': meta['compressed_bit_length']
    }


def find_chunks_by_content_type(registry, dataset_id, content_type):
    """Return chunk indices whose previews resemble the requested content type."""
    entry = get_dataset_entry(registry, dataset_id)
    matching_chunks = []

    for chunk_idx in entry['chunk_order']:
        meta = entry['chunk_metadata'][chunk_idx]
        preview = meta.get('content_preview', b'') or b''
        preview_len = len(preview) if len(preview) > 0 else 1
        printable = sum(1 for b in preview if 32 <= b <= 126)
        text_ratio = printable / preview_len

        if content_type == "text" and text_ratio > 0.8:
            matching_chunks.append(chunk_idx)
        elif content_type == "binary" and text_ratio < 0.3:
            matching_chunks.append(chunk_idx)
        elif content_type == "mixed" and 0.3 <= text_ratio <= 0.8:
            matching_chunks.append(chunk_idx)

    return matching_chunks


def save_registry_to_file(registry, filepath):
    """Persist the registry to disk (without original payload data)."""
    import pickle
    with open(filepath, 'wb') as file_handle:
        pickle.dump(registry, file_handle)


def load_registry_from_file(filepath):
    """Reload a registry that was previously saved to disk."""
    import pickle
    with open(filepath, 'rb') as file_handle:
        return pickle.load(file_handle)


def stream_byte_range(registry, dataset_id, start_byte, chunk_size=8192):
    """Yield successive byte blocks for memory-efficient range reconstruction."""
    entry = get_dataset_entry(registry, dataset_id)
    current_pos = start_byte

    while current_pos < entry['total_length']:
        end_pos = min(current_pos + chunk_size, entry['total_length'])
        segment = reconstruct_byte_range(registry, dataset_id, current_pos, end_pos)
        if not segment:
            break
        yield segment
        current_pos = end_pos


