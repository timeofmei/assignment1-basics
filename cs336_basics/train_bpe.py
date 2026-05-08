import regex as re
from collections import Counter
from multiprocessing import Pool, cpu_count
from util import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def bpe(input_path: str, vocab_size: int, special_tokens: list[str], multi: bool = True):
    with open(input_path, "rb") as f:
        num_processes = cpu_count()
        boundaries = find_chunk_boundaries(f, 60, b"<|endoftext|>")

        init_freq_table = Counter()

        if multi:
            tables = []
            chunk_positions = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_positions.append((start, end - start))

            with Pool(processes=num_processes) as pool:
                for freq_table in pool.imap_unordered(pretokenization, [(chunk_position, special_tokens, input_path, multi) for chunk_position in chunk_positions]):
                    init_freq_table.update(freq_table)

            del tables
        else:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                freq_table = pretokenization((chunk, special_tokens, input_path, multi))
                init_freq_table.update(freq_table)

        vocab, merges = run_bpe(init_freq_table, vocab_size, special_tokens)
        return vocab, merges


def pretokenization(args):
    chunk, special_tokens, input_path, multi = args
    if multi:
        with open(input_path, "rb") as f:
            f.seek(chunk[0])
            chunk = f.read(chunk[1]).decode("utf-8", errors="ignore")
    special_pat = rf"""(?:{"|".join(map(re.escape, special_tokens))})"""
    corpus = re.split(special_pat, chunk)
    freq_table = Counter()
    for c in corpus:
        for p in re.finditer(PAT, c):
            byte_tuple = tuple(bytes([b]) for b in p.group().encode("utf-8"))
            freq_table[byte_tuple] += 1
    return freq_table


def run_bpe(freq_table: Counter[tuple[bytes], int], vocab_size: int,  special_tokens: list[str]):
    vocab = {i: special_tokens[i].encode("utf-8") for i in range(len(special_tokens))}
    for i in range(256):
        vocab[i+len(special_tokens)] = bytes([i])
    merges = []
    successive_pairs = Counter()
    for k in freq_table:
        if len(k) <= 1:
            continue
        for first, second in zip(k, k[1:]):
            successive_pairs[(first, second)] += freq_table[k]

    i = len(vocab.keys())
    while i < vocab_size:
        win_count = successive_pairs.most_common(1)[0][1]
        win_pair = max(k for k, v in successive_pairs.items() if v == win_count)
        merges.append(win_pair)
        win_byte = b''.join(win_pair)
        vocab[i] = win_byte
        freq_table, successive_pairs = merge(freq_table, successive_pairs, win_pair, win_byte)
        i += 1

    return vocab, merges


def merge(freq_table: Counter[tuple[bytes], int], successive_pairs: Counter[tuple[bytes], int], win_pair: tuple[bytes], win_byte: bytes):
    freq_table_new = Counter()
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
    successive_pairs.pop(win_pair, None)
    return freq_table_new, successive_pairs
