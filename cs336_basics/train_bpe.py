import regex as re
import os
from typing import BinaryIO
from collections import defaultdict
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        num_processes = cpu_count()
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

        del tables
        del chunks

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
    successive_pairs = defaultdict(int)
    for k in freq_table:
        if len(k) <= 1:
            continue
        for first, second in zip(k, k[1:]):
            successive_pairs[(first, second)] += freq_table[k]

    i = len(vocab.keys())
    while i < vocab_size:
        win_pair_tuple = max(successive_pairs.items(), key=lambda x: (x[1], *x[0]))
        win_pair = win_pair_tuple[0]
        merges.append(win_pair)
        win_byte = b''.join(win_pair)
        vocab[i] = win_byte
        freq_table, successive_pairs = merge(freq_table, successive_pairs, win_pair, win_byte)
        i += 1

    return vocab, merges


def merge(freq_table, successive_pairs, win_pair, win_byte):
    freq_table_new = defaultdict(int)
    for k in freq_table:
        if win_pair in zip(k, k[1:]):
            k_new = []
            k_length = len(k)
            k_freq = freq_table[k]
            processed = [False] * k_length
            i = 0
            while i < k_length - 1:
                if (k[i], k[i+1]) == win_pair:
                    k_new.append(win_byte)
                    if i > 0 and not processed[i-1] and not processed[i]:
                        successive_pairs[(k[i-1], k[i])] -= k_freq

                    if i + 1 < k_length - 1 and not processed[i+1] and not processed[i+2]:
                        successive_pairs[(k[i+1], k[i+2])] -= k_freq
                    processed[i] = True
                    processed[i+1] = True
                    i += 2
                else:
                    k_new.append(k[i])
                    i += 1
            if i == k_length - 1:
                k_new.append(k[i])

            k_new = tuple(k_new)
            freq_table_new[k_new] = freq_table[k]
            k_new_length = len(k_new)
            processed = [False] * k_new_length
            i = 0
            for i in range(k_new_length):
                if k_new[i] == win_byte:
                    if i > 0 and not processed[i-1] and not processed[i]:
                        successive_pairs[(k_new[i-1], k_new[i])] += k_freq
                    if i < k_new_length - 1 and not processed[i] and not processed[i+1]:
                        successive_pairs[(k_new[i], k_new[i+1])] += k_freq
                    processed[i] = True
        else:
            freq_table_new[k] = freq_table[k]
    del successive_pairs[win_pair]
    return freq_table_new, successive_pairs
