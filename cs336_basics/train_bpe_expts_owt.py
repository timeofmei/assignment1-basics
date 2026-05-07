from train_bpe import bpe
import json

FILENAME = "owt_valid"

if __name__ == "__main__":
    vocab, merges = bpe(f"data/{FILENAME}.txt", 32000, ["<|endoftext|>"], False)

    with open(f"data/vocab/vocab-{FILENAME}.json", "w", encoding="utf-8") as f:
        for k in vocab:
            vocab[k] = str(vocab[k])
        json.dump(vocab, f)

    with open(f"data/merges/merges-{FILENAME}.txt", "w", encoding="utf-8") as f:
        for line in merges:
            f.write(f"{line[0]} {line[1]}\n")
