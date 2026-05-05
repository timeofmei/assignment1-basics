from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        return Tokenizer()

    def encode(self, text: str) -> list[int]:
        return []

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return iter()

    def decode(self, ids: list[int]) -> str:
        return ""