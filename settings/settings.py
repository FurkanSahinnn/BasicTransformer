from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Model:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_length: int
    dropout: float

@dataclass
class Training:
    batch_size: int
    learning_rate: float
    epochs: int
    context_length: int
    stride: int

@dataclass
class TestParameters:
    context_length: int
    stride: int
    batch_size: int

@dataclass
class TestTexts:
    english_long: str
    english_short: str
    turkish_long: str
    turkish_short: str