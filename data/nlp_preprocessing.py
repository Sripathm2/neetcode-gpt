import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        words = []
        for strs in positive:
            for word in strs.split(" "):
                words.append(word)
        for strs in negative:
            for word in strs.split(" "):
                words.append(word)

        unique_words = sorted(set(words))
        dictionary = {word: idx for idx, word in enumerate(unique_words, start=1)}

        output = []
        for strs in positive:
            encode = [dictionary[word] for word in strs.split(" ")]
            output.append(torch.tensor(encode))
        for strs in negative:
            encode = [dictionary[word] for word in strs.split(" ")]
            output.append(torch.tensor(encode))

        return torch.nn.utils.rnn.pad_sequence(output, padding_value=0, batch_first=True)