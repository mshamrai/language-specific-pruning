from datasets import load_dataset


data_files = {
    "train": "splitted_data/train_*.jsonl",
    "test": "splitted_data/test_*.jsonl"
}

dataset = load_dataset('json', data_files=data_files)

dataset.push_to_hub("mshamrai/lang-pruning-uk-uber-text-2")
