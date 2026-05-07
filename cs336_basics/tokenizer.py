from collections.abc import Iterable, Iterator
import json
import ast
import regex as re
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_BYTE = r"""(b(?:'[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*")) (b(?:'[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*"))"""


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_priority = {v: i for i, v in enumerate(merges)}
        self.special_tokens = special_tokens
        self.cached_result = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f, object_hook=lambda d: {int(k): ast.literal_eval(v) for k, v in d.items()})
        with open(merges_filepath, encoding="utf-8") as f:
            merges = []
            line = f.readline()
            while line != "":
                token1, token2 = re.findall(PAT_BYTE, line)[0]
                merges.append((ast.literal_eval(token1), ast.literal_eval(token2)))
                line = f.readline()
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pieces = self._pretokenization(text)
        token_ids = []
        for piece in pieces:
            if isinstance(piece, int):
                token_ids.append(piece)
            else:
                token_ids += self._parse(piece)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        result = b''
        for id in ids:
            result += self.vocab[id]
        return result.decode("utf-8", errors="ignore")

    def _pretokenization(self, text: str) -> list[tuple[bytes] | int]:
        has_special_tokens = self.special_tokens is not None

        if has_special_tokens:
            special_pat = rf"""({"|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True)))})"""
            result = re.split(special_pat, text)
        else:
            result = [text]
        new_result = []
        for r in result:
            if has_special_tokens and r in self.special_tokens:
                new_result.append(self.reverse_vocab[r.encode("utf-8")])
            else:
                for p in re.finditer(PAT, r):
                    byte_tuple = tuple(bytes([b]) for b in p.group().encode("utf-8"))
                    new_result.append(byte_tuple)
        return new_result

    def _parse(self, result: tuple[bytes]) -> list[int]:
        init_result = result
        token_ids = self.cached_result.get(init_result)
        if token_ids is not None:
            return token_ids
        pair_priority = []
        heapq.heapify(pair_priority)
        recorded = set()
        stop = len(result) <= 1
        pairs = list(zip(result[:-1], result[1:]))
        next_scan = list(range(len(pairs)))
        while not stop:
            changing = False
            for i in next_scan:
                pair = pairs[i]
                if pair in recorded:
                    continue
                recorded.add(pair)
                priority = self.merges_priority.get(pair)
                if priority is not None:
                    heapq.heappush(pair_priority, (priority, pair))
                    changing = True
                else:
                    continue
            if not changing and len(pair_priority) == 0:
                break
            result, next_scan = self._merge(result, heapq.heappop(pair_priority)[1])
            pairs = list(zip(result[:-1], result[1:]))
            stop = len(result) <= 1

        token_ids = [self.reverse_vocab[token] for token in result]
        self.cached_result[init_result] = token_ids
        return token_ids

    def _merge(self, result, win_pair):
        new_result = []
        i = 0
        next_scan = []
        win_byte = b''.join(win_pair)
        while i < len(result) - 1:
            if win_pair == (result[i], result[i+1]):
                new_result.append(win_byte)
                i += 2
            else:
                new_result.append(result[i])
                i += 1
        if i == len(result) - 1:
            new_result.append(result[i])
        
        new_result_length = len(new_result)
        for i in range(new_result_length):
            if new_result[i] == win_byte:
                if i > 0:
                    next_scan.append(i-1)
                if i < new_result_length - 1:
                    next_scan.append(i)
        return new_result, next_scan


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files("data/vocab/vocab-TinyStoriesV2-GPT4-valid.json", "data/merges/merges-TinyStoriesV2-GPT4-valid.txt", ["<|endoftext|>"])
    print(tokenizer.merges_priority)
