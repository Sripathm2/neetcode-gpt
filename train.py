import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            X = []
            Y = []
            for i in range(batch_size):
                starting_position = torch.randint(0, len(data) - context_length, (1,)).item()
                X.append(data[starting_position:starting_position+context_length])
                Y.append(data[starting_position+1:starting_position+context_length+1])
            X = torch.stack(X)
            Y = torch.stack(Y)
            logits = model(X)
            logits_flat = logits.view(-1, logits.shape[-1])
            targets_flat = Y.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.round(loss.item(),4)
