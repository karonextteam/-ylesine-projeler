from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
if device == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))


@dataclass
class InferConfig:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float


class CharTokenizer:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: ch for ch, i in stoi.items()}

    def encode(self, s: str) -> list[int]:
        unknown = [c for c in s if c not in self.stoi]
        if unknown:
            raise ValueError(f"Prompt içinde vocab dışı karakter var: {unknown}")
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: InferConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int | None = 20,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


def main():
    base = Path(__file__).resolve().parent

    ckpt_name = "char_gpt_best.pt"
    ckpt_path = base / "out" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    cfg_dict = checkpoint["config"]
    stoi = checkpoint["vocab"]

    cfg = InferConfig(
        block_size=cfg_dict["block_size"],
        n_layer=cfg_dict["n_layer"],
        n_head=cfg_dict["n_head"],
        n_embd=cfg_dict["n_embd"],
        dropout=cfg_dict["dropout"],
    )

    tok = CharTokenizer(stoi)
    model = MiniGPT(tok.vocab_size, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model yüklendi.")
    print("Çıkmak için boş prompt gir.")

    while True:
        prompt = input("\nPrompt: ").strip()
        if not prompt:
            break

        temperature = input("Temperature (örn 0.7, boşsa 0.7): ").strip()
        temperature = float(temperature) if temperature else 0.7

        top_k = input("Top-k (örn 40, boşsa 40, kapatmak için 0): ").strip()
        top_k = int(top_k) if top_k else 40
        if top_k == 0:
            top_k = None

        max_new_tokens = input("Kaç yeni karakter? (örn 400, boşsa 400): ").strip()
        max_new_tokens = int(max_new_tokens) if max_new_tokens else 400

        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()

        text = tok.decode(out)

        print("\n--- ÜRETİLEN METİN ---\n")
        print(text)
        print("\n----------------------")


if __name__ == "__main__":
    main()