import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as used in models like LLaMA, GPT-NeoX.
    
    RoPE encodes position information by rotating token embeddings in a way that
    naturally encodes relative positions between tokens.
    """
    def __init__(self, dim, max_seq_len=2048, base=10000, device="cpu"):
        """
        Args:
            dim: Dimension of the embeddings (should be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for the frequency calculation (default: 10000)
            device: Device to store tensors on
        """
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = device
        
        # Precompute the frequency tensor (inverse frequencies)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for efficiency
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """Precompute cos and sin values for positions up to seq_len."""
        # Create position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies: outer product of positions and inverse frequencies
        # Shape: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Concatenate frequencies to match embedding dimension
        # Shape: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Precompute cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x, seq_len=None):
        """
        Apply rotary positional embedding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            seq_len: Sequence length (if None, inferred from x)
            
        Returns:
            Tensor with rotary positional encoding applied
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # If sequence is longer than cached, rebuild cache
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        # Get cos and sin for current sequence length
        # Shape: [seq_len, dim]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Apply rotation
        return self._apply_rotary_emb(x, cos, sin)
    
    def _apply_rotary_emb(self, x, cos, sin):
        """
        Apply rotary embedding using the rotation matrix.
        
        The rotation is applied by splitting the embedding into pairs and rotating them.
        """
        # Split x into two halves for rotation
        # Shape: [batch_size, seq_len, dim/2]
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        
        # Apply rotation
        # cos and sin shapes: [seq_len, dim]
        cos = cos.unsqueeze(0)  # [1, seq_len, dim]
        sin = sin.unsqueeze(0)  # [1, seq_len, dim]
        
        # Rotate: [x1*cos - x2*sin, x1*sin + x2*cos]
        cos1, cos2 = cos[..., : self.dim // 2], cos[..., self.dim // 2 :]
        sin1, sin2 = sin[..., : self.dim // 2], sin[..., self.dim // 2 :]
        
        rotated_x1 = x1 * cos1 - x2 * sin1
        rotated_x2 = x1 * sin2 + x2 * cos2
        
        # Concatenate back
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


class EmbeddingWithRoPE(nn.Module):
    """
    Complete embedding layer with token embeddings and Rotary Positional Encoding.
    """
    def __init__(self, vocab_size, embedding_dim, max_seq_len=2048, base=10000, device="cpu"):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings (should be even for RoPE)
            max_seq_len: Maximum sequence length
            base: Base for RoPE frequency calculation
            device: Device to store tensors on
        """
        super(EmbeddingWithRoPE, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        
        # Rotary Positional Embedding
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=embedding_dim,
            max_seq_len=max_seq_len,
            base=base,
            device=device
        )
    
    def forward(self, x):
        """
        Args:
            x: Token indices of shape [batch_size, seq_len]
            
        Returns:
            Embeddings with positional encoding of shape [batch_size, seq_len, embedding_dim]
        """
        # Get token embeddings
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Apply rotary positional encoding
        embeddings = self.rotary_emb(token_emb)
        
        return embeddings


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    vocab_size = 50000
    embedding_dim = 512  # Must be even for RoPE
    max_seq_len = 1024
    batch_size = 4
    seq_len = 128
    
    # Create model
    embedding_model = EmbeddingWithRoPE(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        device="cpu"
    )
    
    # Create sample input (token indices)
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    embeddings = embedding_model(input_tokens)
    
    print(f"Input shape: {input_tokens.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {embedding_dim}]")
    
    # Test with different sequence lengths
    print("\n" + "="*50)
    print("Testing with different sequence lengths:")
    for test_len in [64, 128, 256, 512]:
        test_input = torch.randint(0, vocab_size, (2, test_len))
        test_output = embedding_model(test_input)
        print(f"Seq len {test_len}: Input {test_input.shape} -> Output {test_output.shape}")
    
    print("\n" + "="*50)
    print("RoPE Implementation Complete! ✓")
    print("="*50)