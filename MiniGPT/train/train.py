import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.minigpt import MiniGPT


# ============================================================
# 1. 超参数
# ============================================================

batch_size = 32
max_seq_len = 64

embedding_dim = 128
hidden_dim = 512
num_heads = 4
num_layers = 4
dropout = 0.1

learning_rate = 3e-4
weight_decay = 0.01
epochs = 20
early_stopping_patience = 5
seed = 42

DATA_PATH = PROJECT_ROOT / "data" / "input.txt"
BEST_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "minigpt_best.pth"
FINAL_CHECKPOINT_PATH = PROJECT_ROOT / "train" / "minigpt_final.pth"
HISTORY_PATH = PROJECT_ROOT / "train" / "history.json"


# ============================================================
# 2. 工具函数
# ============================================================

def set_seed(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# 3. 读取文本并建立字符词表
# ============================================================

with open(DATA_PATH, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for char, index in stoi.items()}


def encode(text_value):
    unknown = sorted(set(text_value) - set(stoi))
    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")
    return [stoi[char] for char in text_value]


def decode(indices):
    return "".join(itos[index] for index in indices)


data = torch.tensor(encode(text), dtype=torch.long)


# ============================================================
# 4. 划分训练集和验证集
# ============================================================

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]


# ============================================================
# 5. Dataset 和 DataLoader
# ============================================================

class TextDataset(Dataset):
    def __init__(self, data_tensor, sequence_length):
        self.data = data_tensor
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, index):
        x = self.data[index:index + self.sequence_length]
        y = self.data[index + 1:index + self.sequence_length + 1]
        return x, y


train_dataset = TextDataset(train_data, max_seq_len)
val_dataset = TextDataset(val_data, max_seq_len)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 6. 创建模型、损失函数和优化器
# ============================================================

def create_model():
    return MiniGPT(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        dropout=dropout,
    )


def checkpoint_payload(model, checkpoint_type, epoch, validation_loss, history):
    return {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "checkpoint_type": checkpoint_type,
        "checkpoint_epoch": epoch,
        "checkpoint_val_loss": validation_loss,
        "history": history,
        "config": {
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
        },
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Validation loader is empty. Check max_seq_len and data size.")
    return total_loss / total_batches


def main():
    set_seed(seed)
    device = get_device()

    print(f"Device: {device}")
    print(f"Text length: {len(text)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train tokens: {len(train_data)}")
    print(f"Validation tokens: {len(val_data)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    model = create_model().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_batches += 1

        train_loss = total_train_loss / total_batches
        val_loss = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "perplexity": torch.exp(torch.tensor(val_loss)).item(),
            }
        )

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val PPL: {history[-1]['perplexity']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                checkpoint_payload(model, "best", best_epoch, best_val_loss, history),
                BEST_CHECKPOINT_PATH,
            )
            print(f"Saved best checkpoint at epoch {best_epoch}.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping: validation loss has not improved for "
                f"{early_stopping_patience} epochs."
            )
            break

    torch.save(
        checkpoint_payload(model, "final", epoch, val_loss, history),
        FINAL_CHECKPOINT_PATH,
    )
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
