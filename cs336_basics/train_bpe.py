import regex as re
import os
from typing import BinaryIO
from collections import defaultdict
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        num_processes = 12
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        init_freq_table = defaultdict(int)
        tables = []
        chunks = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

        with Pool(processes=num_processes) as pool:
            for freq_table in pool.imap_unordered(pretokenization, [(chunk, special_tokens) for chunk in chunks]):
                tables.append(freq_table)

        for table in tables:
            for k, v in table.items():
                init_freq_table[k] += v

        vocab, merges = run_bpe(init_freq_table, vocab_size, special_tokens)
        return vocab, merges


def pretokenization(args):
    chunk, special_tokens = args
    corpus = split_with_special_tokens(chunk, special_tokens)
    freq_table = defaultdict(int)
    for c in corpus:
        for p in re.finditer(PAT, c):
            byte_tuple = tuple(bytes([b]) for b in p.group().encode("utf-8"))
            freq_table[byte_tuple] += 1
    return freq_table


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_with_special_tokens(corpus: str, special_tokens: list[str]):
    result = [corpus]
    for tok in special_tokens:
        length = len(result)
        for _ in range(length):
            s = result.pop(0)
            result += s.split(tok)
    return result


def run_bpe(freq_table: dict[tuple[bytes], int], vocab_size: int,  special_tokens: list[str]):
    vocab = {i: special_tokens[i].encode("utf-8") for i in range(len(special_tokens))}
    for i in range(256):
        vocab[i+len(special_tokens)] = bytes([i])
    merges = []
    freq_table = dict(sorted(freq_table.items(), key=lambda x: (x[1], *x[0]), reverse=True))

    i = len(vocab.keys())
    while i < vocab_size:
        successive_pairs = defaultdict(int)
        for k in freq_table:
            if len(k) <= 1:
                continue
            for first, second in zip(k, k[1:]):
                successive_pairs[(first, second)] += freq_table[k]
        successive_pairs = sorted(successive_pairs.items(), key=lambda x: (x[1], *x[0]), reverse=True)

        if len(successive_pairs) < 1:
            break

        win_pair = successive_pairs[0][0]
        win_byte = b''.join(win_pair)
        vocab[i] = win_byte
        merges.append(win_pair)
        freq_table = merge(freq_table, win_pair, win_byte)

        i += 1

    return vocab, merges


def merge(freq_table, win_pair, win_byte):
    freq_table_new = defaultdict(int)
    for k in freq_table:
        k_new = []
        j = 0
        while j < len(k) - 1:
            if (k[j], k[j+1]) == win_pair:
                k_new.append(win_byte)
                j += 2
            else:
                k_new.append(k[j])
                j += 1
        if j == len(k) - 1:
            k_new.append(k[j])
        freq_table_new[tuple(k_new)] = freq_table[k]
    return freq_table_new
