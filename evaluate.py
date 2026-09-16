import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from minigpt.checkpoint import load_model_from_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "minigpt_best.pth"
DATA_PATH = PROJECT_ROOT / "data" / "input.txt"


class TextDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, index):
        x = self.data[index:index + self.sequence_length]
        y = self.data[index + 1:index + self.sequence_length + 1]
        return x, y


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MiniGPT loss and perplexity."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Only evaluate the first N batches; useful for a quick check.",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()
    model, checkpoint, config = load_model_from_checkpoint(args.checkpoint, device)

    text = DATA_PATH.read_text(encoding="utf-8")
    stoi = checkpoint["stoi"]
    unknown = sorted(set(text) - set(stoi))
    if unknown:
        raise ValueError(f"Dataset contains characters outside the checkpoint vocab: {unknown}")

    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    split_index = int(0.9 * len(data))
    selected_data = data[:split_index] if args.split == "train" else data[split_index:]

    dataset = TextDataset(selected_data, config["max_seq_len"])
    if len(dataset) == 0:
        raise ValueError("The selected split is too short for the checkpoint max_seq_len.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(
            logits.reshape(-1, config["vocab_size"]),
            y.reshape(-1),
        )
        total_loss += loss.item()
        total_batches += 1

        if args.max_batches is not None and total_batches >= args.max_batches:
            break

    average_loss = total_loss / total_batches
    perplexity = math.exp(average_loss)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Tokens: {len(selected_data)}")
    print(f"Batches evaluated: {total_batches}")
    print(f"Cross-entropy loss: {average_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
