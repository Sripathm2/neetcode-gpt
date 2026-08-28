import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        dataset = raw_dataset.split(" ")
        X = []
        Y = []
        for i in range(batch_size):
            starting_position = torch.randint(0, len(dataset) - context_length, (1,)).item()
            X.append(dataset[starting_position:starting_position+context_length])
            Y.append(dataset[starting_position+1:starting_position+context_length+1])
        return (X, Y)