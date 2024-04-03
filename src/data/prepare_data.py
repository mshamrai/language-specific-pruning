import os
import argparse
import json


def read_chunk(corpora_path, min_len):
    with open(corpora_path, "r") as f:
        chunk = []
        count_n = 0
        for line in f:
            if line == "\n":
                count_n += 1
                if count_n == 3:
                    entity = "".join(chunk)
                    count_n = 0
                    del chunk
                    chunk = []
                    if len(entity) <= min_len:
                        continue
                    yield entity
            else:
                chunk.append(line)


def split_data(corpora_path, output_dir, data_name, min_len=512, split_ratio=None, train_size=None, test_size=None):
    print(f"Splitting {data_name}...")

    train_count = 0
    train_entities = []
    test_count = 0
    test_entities = []

    train_output_path = os.path.join(output_dir, f"train_{data_name}.jsonl")
    if os.path.isfile(train_output_path):
        os.remove(train_output_path)

    test_output_path = os.path.join(output_dir, f"test_{data_name}.jsonl")
    if os.path.isfile(test_output_path):
        os.remove(test_output_path)

    for i, entity in enumerate(read_chunk(corpora_path, min_len)):
        if split_ratio:
            if i % 10000 < split_ratio * 10000: # goes into train set
                train_count += 1
                train_entities.append(entity)
            else: # goes into test set
                test_count += 1
                test_entities.append(entity)
            
            if i > 0 and i % 10000 == 0: # write to disk
                train_corpora = "".join([json.dumps({"text": " ".join(text.split())}, ensure_ascii=False) + '\n' for text in train_entities])
                with open(train_output_path, "a") as f:
                    f.write(train_corpora)

                del train_entities
                train_entities = []

                test_corpora = "".join([json.dumps({"text": " ".join(text.split())}, ensure_ascii=False) + '\n' for text in test_entities])
                with open(test_output_path, "a") as f:
                    f.write(test_corpora)

                del test_entities
                test_entities = []
        elif train_size and test_size:
            if train_count < train_size: # goes into train set
                train_count += 1
                train_entities.append(entity)
            elif test_count < test_size: # goes into test set
                test_count += 1
                test_entities.append(entity)
            else:
                train_corpora = "".join([json.dumps({"text": " ".join(text.split())}, ensure_ascii=False) + '\n' for text in train_entities])
                with open(train_output_path, "a") as f:
                    f.write(train_corpora)

                del train_entities

                test_corpora = "".join([json.dumps({"text": " ".join(text.split())}, ensure_ascii=False) + '\n' for text in test_entities])
                with open(test_output_path, "a") as f:
                    f.write(test_corpora)

                del test_entities
                
                break
        else:
            NotImplementedError("split_ratio or train_size and test_size should be passed")

    print("Texts count: " + str(train_count + test_count))
    print("Train count: " + str(train_count))
    print("Test count: " + str(test_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpora-path', type=str)
    parser.add_argument('--data-name', type=str)
    parser.add_argument('--output-dir', type=str, default="splitted_data")
    args = parser.parse_args()
    split_data(**vars(args))