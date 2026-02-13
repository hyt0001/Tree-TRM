import torch
from torch.utils.data import Dataset
import os

class LongBenchDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        """
        LongBench Dataset
        加载由 build_longbench_dataset.py 生成的 .pt 文件
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load tensors
        self.inputs = torch.load(os.path.join(data_dir, f"{split}_inputs.pt"))   # [N, SEQ_LEN] (Question)
        self.context = torch.load(os.path.join(data_dir, f"{split}_context.pt")) # [N, MAX_CHUNKS, SEQ_LEN] (Chunks)
        self.labels = torch.load(os.path.join(data_dir, f"{split}_labels.pt"))   # [N, SEQ_LEN] (Answer)
        
        print(f"Loaded {split} dataset: {len(self.inputs)} examples")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tree-TRM 期望的格式是一个 dict
        return {
            "inputs": self.inputs[idx],             # Query / Question
            "puzzle_identifiers": self.context[idx], # Context Chunks (作为外部记忆)
            "labels": self.labels[idx]              # Answer
        }

    @staticmethod
    def collate_fn(batch):
        """
        简单的 collate function，堆叠 tensor
        """
        inputs = torch.stack([item["inputs"] for item in batch])
        puzzle_identifiers = torch.stack([item["puzzle_identifiers"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        return {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_identifiers,
            "labels": labels
        }

