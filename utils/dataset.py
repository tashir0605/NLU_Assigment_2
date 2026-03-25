from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from utils.vocab import NameVocab


def load_names(file_path: str) -> List[str]:


    names = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip().lower()
            if text:
                names.append(text)
    return names


class NameSequenceDataset(Dataset):

    def __init__(self, names: List[str], vocab: NameVocab):
        self.vocab = vocab
        self.samples: List[Tuple[List[int], List[int]]] = []

        for name in names:
            encoded = vocab.encode_name(name)
            input_ids = encoded[:-1]
            target_ids = encoded[1:]
            self.samples.append((input_ids, target_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]


def collate_batch(batch: List[Tuple[List[int], List[int]]], pad_idx: int):

    batch_size = len(batch)
    lengths = [len(item[0]) for item in batch]
    max_len = max(lengths)

    inputs = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    targets = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)

    for i, (input_ids, target_ids) in enumerate(batch):
        seq_len = len(input_ids)
        inputs[i, :seq_len] = torch.tensor(input_ids, dtype=torch.long)
        targets[i, :seq_len] = torch.tensor(target_ids, dtype=torch.long)

    return inputs, targets, torch.tensor(lengths, dtype=torch.long)
