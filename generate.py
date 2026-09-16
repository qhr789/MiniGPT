import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from minigpt.checkpoint import load_model_from_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "minigpt_best.pth"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_text(text, stoi):
    unknown = sorted(set(text) - set(stoi))
    if unknown:
        raise ValueError(f"Prompt contains unknown characters: {unknown}")
    return [stoi[char] for char in text]


def decode_tokens(indices, itos):
    return "".join(itos[int(index)] for index in indices)


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, greedy=False):
    """Return one sampled token id for each row in logits [B, vocab_size]."""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        top_k_values = torch.topk(logits, k=k, dim=-1).values
        threshold = top_k_values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -torch.inf)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        indices_to_remove = cumulative_probs > top_p
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(indices_to_remove, -torch.inf)
        filtered_probs = F.softmax(sorted_logits, dim=-1)
        sorted_next_token = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_indices.gather(-1, sorted_next_token)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model,
    start_text,
    stoi,
    itos,
    max_seq_len,
    device,
    max_new_tokens=200,
    temperature=1.0,
    top_k=None,
    top_p=None,
    greedy=False,
):
    token_ids = encode_text(start_text, stoi)
    if not token_ids:
        raise ValueError("Prompt must contain at least one known character.")

    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()

    for _ in range(max_new_tokens):
        x_cond = x[:, -max_seq_len:]
        logits = model(x_cond)[:, -1, :]
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        x = torch.cat([x, next_token], dim=1)

    return decode_tokens(x[0].tolist(), itos)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", type=str, default="First Citizen:")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device()
    model, checkpoint, config = load_model_from_checkpoint(args.checkpoint, device)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(
        "Decoding: "
        f"greedy={args.greedy}, temperature={args.temperature}, "
        f"top_k={args.top_k}, top_p={args.top_p}"
    )
    print("-" * 60)

    result = generate(
        model=model,
        start_text=args.prompt,
        stoi=stoi,
        itos=itos,
        max_seq_len=config["max_seq_len"],
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )
    print(result)


if __name__ == "__main__":
    main()
