from tokenizer import Tokenizer
import numpy as np
import pickle
from util import find_chunk_boundaries
from multiprocessing import Pool, cpu_count


def experiment_d(multi: bool = True, chunk_store: bool = True):
    prefix = ["TinyStoriesV2-GPT4-", "owt_"]
    suffix = ["train", "valid"]
    for p in prefix:
        for s in suffix:
            filename = p + s
            text_path = f"data/{filename}.txt"
            merges_path = f"data/merges/merges-{filename}.txt"
            vocab_path = f"data/vocab/vocab-{filename}.json"
            token_ids_path = f"data/token_ids/token_ids-{filename}"
            try:
                tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
            except FileNotFoundError as e:
                print(e)
                continue

            with open(text_path, "rb") as f_text:
                num_processes = cpu_count()
                boundaries = find_chunk_boundaries(f_text, 200, b"<|endoftext|>")
                chunk_positions = []
                token_ids = np.array([], dtype="uint16")
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    chunk_positions.append((start, end - start))
                chunk_worker_args = [(tokenizer, text_path, chunk_index, chunk_position) for chunk_index, chunk_position in enumerate(chunk_positions)]
                if multi:
                    with Pool(processes=num_processes) as pool:
                        if chunk_store:
                            multi_func = pool.imap_unordered
                        else:
                            multi_func = pool.imap
                        for chunk_index, chunk_token_ids in multi_func(chunk_worker, chunk_worker_args):
                            if chunk_store:
                                with open(token_ids_path + f"-{chunk_index}.pkl", "wb") as f_chunk_token:
                                    pickle.dump(chunk_token_ids, f_chunk_token)
                            else:
                                token_ids = np.concatenate((token_ids, chunk_token_ids))
                else:
                    for chunk_worker_arg in chunk_worker_args:
                        _, chunk_token_ids = chunk_worker(chunk_worker_arg)
                        token_ids = np.concatenate((token_ids, chunk_token_ids))
            if not chunk_store:
                with open(token_ids_path + ".pkl", "wb") as f_token:
                    pickle.dump(token_ids, f_token)


def chunk_worker(args):
    tokenizer, text_path, chunk_index, chunk_position = args
    with open(text_path, "rb") as f_text:
        f_text.seek(chunk_position[0])
        chunk = f_text.read(chunk_position[1]).decode("utf-8", errors="ignore")
        chunk_token_ids = np.array(tokenizer.encode(chunk), dtype="uint16")
    return chunk_index, chunk_token_ids


if __name__ == "__main__":
    experiment_d()
