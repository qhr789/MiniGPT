import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minigpt import MiniGPT  # noqa: E402


def main():
    torch.manual_seed(42)

    vocab_size = 100
    max_seq_len = 16
    model = MiniGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embedding_dim=64,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )

    x = torch.randint(0, vocab_size, (2, 8))
    input_ids = x[:, :-1]
    targets = x[:, 1:]

    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )

    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"targets shape:   {tuple(targets.shape)}")
    print(f"logits shape:    {tuple(logits.shape)}")
    print(f"loss:            {loss.item():.4f}")

    assert logits.shape == (2, 7, vocab_size)
    assert torch.isfinite(loss)
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
