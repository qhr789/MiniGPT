import torch

from minigpt import MiniGPT


def make_model():
    return MiniGPT(
        embedding_dim=32,
        hidden_dim=64,
        num_heads=4,
        vocab_size=50,
        max_seq_len=16,
        num_layers=2,
        dropout=0.0,
    )


def test_model_forward_shape():
    model = make_model()
    input_ids = torch.randint(0, 50, (2, 8))

    logits = model(input_ids)

    assert logits.shape == (2, 8, 50)


def test_causal_mask_does_not_leak_future_tokens():
    torch.manual_seed(42)
    model = make_model()
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    changed_input_ids = input_ids.clone()
    changed_input_ids[:, -1] = 6

    with torch.no_grad():
        original_logits = model(input_ids)
        changed_logits = model(changed_input_ids)

    prefix_length = input_ids.size(1) - 1
    assert torch.allclose(
        original_logits[:, :prefix_length],
        changed_logits[:, :prefix_length],
    )
