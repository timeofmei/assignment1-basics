from tokenizer import Tokenizer
import numpy as np
import pickle
from util import find_chunk_boundaries


def experiment_d():
    prefix = ["TinyStoriesV2-GPT4-", "owt_"]
    suffix = ["train", "valid"]
    for p in prefix:
        for s in suffix:
            filename = p + s
            text_path = f"data/{filename}.txt"
            merges_path = f"data/merges/merges-{filename}.txt"
            vocab_path = f"data/vocab/vocab-{filename}.json"
            token_ids_path = f"data/token_ids/token_ids-{filename}.pkl"
            try:
                tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
            except FileNotFoundError as e:
                print(e)
                continue

            token_ids = []
            with open(text_path, "rb") as f_text:
                boundaries = find_chunk_boundaries(f_text, 20, b"<|endoftext|>")
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f_text.seek(start)
                    chunk = f_text.read(end - start).decode("utf-8", errors="ignore")
                    token_ids += tokenizer.encode(chunk)
            token_ids = np.array(token_ids, dtype="uint16")

            with open(token_ids_path, "wb") as f_token:
                pickle.dump(token_ids, f_token)


if __name__ == "__main__":
    experiment_d()
