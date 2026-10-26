import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, mask):
        attn_output, _ = self.attn(
            self.ln1(x),
            self.ln1(x),
            self.ln1(x),
            attn_mask=mask
        )

        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=128,
        n_head=4,
        n_layer=4,
        block_size=128
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.token_embedding(idx) + self.position_embedding(positions)

        mask = torch.triu(
            torch.ones(T, T, device=idx.device) * float("-inf"),
            diagonal=1
        )

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)