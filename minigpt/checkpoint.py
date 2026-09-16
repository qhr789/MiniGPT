from pathlib import Path

import torch

from .model import MiniGPT


def load_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )


def read_config(checkpoint):
    config = checkpoint.get("config", {})
    keys = (
        "embedding_dim",
        "hidden_dim",
        "num_heads",
        "vocab_size",
        "max_seq_len",
        "num_layers",
    )

    values = {}
    for key in keys:
        value = config.get(key, checkpoint.get(key))
        if value is None:
            raise KeyError(f"Missing '{key}' in checkpoint metadata.")
        values[key] = value
    return values


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = read_config(checkpoint)

    model = MiniGPT(
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        num_layers=config["num_layers"],
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, config
