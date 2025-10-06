import torch
import torch.nn as nn

class BasicEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device="cpu"):
        super(BasicEmbedding, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        # Implement Positional Encoding

    def forward(self, x):
        return self.embedding(x)


