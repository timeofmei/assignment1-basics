from train_bpe import bpe
import json

FILENAME = "TinyStoriesV2-GPT4-train"

if __name__ == "__main__":
    vocab, merges = bpe(f"data/{FILENAME}.txt", 10000, ["<|endoftext|>"])

    with open(f"data/vocab-{FILENAME}.json", "w") as f:
        for k in vocab:
            vocab[k] = str(vocab[k])
        json.dump(vocab, f)

    with open(f"data/merges-{FILENAME}.txt", "w") as f:
        for line in merges:
            f.write(f"{line[0]} {line[1]}\n")
