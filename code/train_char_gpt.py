from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
if device == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))



@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "cuda"
    # Data
    data_dir: str = "data"
    train_file: str = "train.txt"
    val_file: str = "val.txt"

    # Model
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.10

    # Train
    batch_size: int = 16
    max_iters: int = 15000
    eval_interval: int = 100
    eval_iters: int = 50
    lr: float = 2e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Sampling
    sample_every: int = 200
    sample_len: int = 300
    temperature: float = 0.7
    top_k: int = 40

    # Output
    out_dir: str = "out"
    save_last_name: str = "char_gpt_last.pt"
    save_best_name: str = "char_gpt_best.pt"


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def make_dataset(text: str, tok: CharTokenizer) -> torch.Tensor:
    return torch.tensor(tok.encode(text), dtype=torch.long)


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

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hs)
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
    def __init__(self, vocab_size: int, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout) for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.cfg.block_size}")

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
    self,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int | None = 40,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


def get_batch(data: torch.Tensor, cfg: TrainConfig, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: MiniGPT, train_data: torch.Tensor, val_data: torch.Tensor, cfg: TrainConfig):
    model.eval()
    out: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg, cfg.device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    cfg = TrainConfig(
        data_dir=str(Path(os.environ.get("DATA_DIR", "data"))),
        out_dir=str(Path(os.environ.get("OUT_DIR", "out"))),
    )

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)

    base = Path(__file__).resolve().parent
    data_dir = base / cfg.data_dir
    train_text = load_text(data_dir / cfg.train_file)
    val_text = load_text(data_dir / cfg.val_file)

    tok = CharTokenizer(train_text + val_text)
    train_data = make_dataset(train_text, tok)
    val_data = make_dataset(val_text, tok)

    print(f"[INFO] device={cfg.device}")
    print(f"[INFO] vocab_size={tok.vocab_size}")
    print(f"[INFO] train_chars={len(train_data)} val_chars={len(val_data)}")

    model = MiniGPT(tok.vocab_size, cfg).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")

    t0 = time.time()
    for it in range(1, cfg.max_iters + 1):
        if it % 10 == 0 or it == 1:
            print(f"[STEP] iter={it}")

        xb, yb = get_batch(train_data, cfg, cfg.device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if it % cfg.eval_interval == 0 or it == 1:
            losses = estimate_loss(model, train_data, val_data, cfg)
            ppl_train = math.exp(losses["train"]) if losses["train"] < 20 else float("inf")
            ppl_val = math.exp(losses["val"]) if losses["val"] < 20 else float("inf")
            dt = time.time() - t0

            print(
                f"iter {it:6d} | train_loss {losses['train']:.4f} val_loss {losses['val']:.4f} "
                f"| ppl_train {ppl_train:.2f} ppl_val {ppl_val:.2f} | {dt:.1f}s"
            )

            out_dir = base / cfg.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            # last checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "vocab": tok.stoi,
                },
                out_dir / cfg.save_last_name,
            )

            # best checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg.__dict__,
                        "vocab": tok.stoi,
                    },
                    out_dir / cfg.save_best_name,
                )
                print(f"[OK] New best checkpoint saved: {cfg.save_best_name} (val_loss={best_val_loss:.4f})")

        if it % cfg.sample_every == 0 or it == 1:
            prompt = "Bir gün"
            idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=cfg.device)
            out = model.generate(idx,max_new_tokens=cfg.sample_len,temperature=cfg.temperature,top_k=cfg.top_k,)[0].tolist()
            print("\n[SAMPLE]\n" + tok.decode(out) + "\n")

    # Save
    out_dir = base / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "vocab": tok.stoi,
        },
        out_dir / "char_gpt_final.pt",
    )
    print("[OK] Saved out/char_gpt_final.pt")


if __name__ == "__main__":
    main()
