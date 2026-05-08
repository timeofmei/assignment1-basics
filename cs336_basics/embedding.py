import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        emb = torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)
        mean = 0
        std = 1
        nn.init.trunc_normal_(emb, mean=mean, std=std, a=-3, b=3)
        self.emb = nn.Parameter(emb, requires_grad=True)

    def load_weights(self, weights: Float[Tensor, " vocab_size d_model"]):
        self.load_state_dict({"emb": weights}, strict=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]
