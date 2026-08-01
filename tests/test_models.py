"""Phase 5 tests: capsule layer shapes, all-model forward passes, losses,
parameter counts, and gradient-flow sanity (README roadmap steps 59-63)."""

from __future__ import annotations

import torch

from src.models import AeroMindCapsNet, AeroMindCNNLSTM, AeroMindEEGNet, build_model
from src.models.layers import DigitCapsuleRouting, PrimaryCapsule1D, squash
from src.models.losses import margin_loss, multi_task_loss
from src.utils.config import DataConfig, ModelConfig, TrainConfig

BATCH, SEQ_LEN, CHANNELS, SAMPLES = 2, 15, 7, 512


def _batch():
    x = torch.randn(BATCH, SEQ_LEN, CHANNELS, SAMPLES)
    workload = torch.tensor([0, 2])
    fatigue = torch.tensor([1, 0])
    return x, workload, fatigue


# --- capsule layers -----------------------------------------------------


def test_squash_preserves_direction_and_bounds_length():
    x = torch.randn(4, 5, 8)
    out = squash(x, dim=-1)
    assert out.shape == x.shape
    lengths = out.norm(dim=-1)
    assert torch.all(lengths < 1.0)
    cos_sim = torch.nn.functional.cosine_similarity(x, out, dim=-1)
    assert torch.all(cos_sim > 0.99)


def test_primary_capsule_output_shape():
    layer = PrimaryCapsule1D(
        in_channels=64, num_types=32, capsule_dim=8, kernel_size=9, stride=16, padding=4
    )
    x = torch.randn(3, 64, 256)
    out = layer(x)
    # T_out = floor((256 + 8 - 9)/16) + 1 = 16 -> n_in = 32*16 = 512
    assert out.shape == (3, 512, 8)


def test_digit_capsule_routing_shape_and_coupling_sums_to_one():
    layer = DigitCapsuleRouting(n_in=512, dim_in=8, n_out=3, dim_out=16, n_iterations=3)
    u = torch.randn(2, 512, 8)
    v, c = layer(u)
    assert v.shape == (2, 3, 16)
    assert c.shape == (2, 512, 3)
    # softmax over the n_out axis -> each primary capsule's outgoing coupling sums to 1
    assert torch.allclose(c.sum(dim=2), torch.ones(2, 512), atol=1e-5)


# --- forward pass, all models --------------------------------------------


def test_capsnet_forward_shapes():
    x, workload, _ = _batch()
    model = AeroMindCapsNet()
    out = model(x, workload_target=workload)
    assert out["workload_logits"].shape == (BATCH, 3)
    assert out["fatigue_logits"].shape == (BATCH, 2)
    assert out["capsule_lengths"].shape == (BATCH, 3)
    assert out["reconstruction"].shape == out["reconstruction_target"].shape


def test_cnn_lstm_forward_shapes():
    x, _, _ = _batch()
    model = AeroMindCNNLSTM()
    out = model(x)
    assert out["workload_logits"].shape == (BATCH, 3)
    assert out["fatigue_logits"].shape == (BATCH, 2)
    assert "reconstruction" not in out


def test_eegnet_forward_shapes():
    x, _, _ = _batch()
    model = AeroMindEEGNet()
    out = model(x)
    assert out["workload_logits"].shape == (BATCH, 3)
    assert out["fatigue_logits"].shape == (BATCH, 2)
    assert "reconstruction" not in out


def test_registry_builds_all_models():
    data_cfg = DataConfig()
    for name in ["aeromind_capsnet", "aeromind_cnn_lstm", "aeromind_eegnet"]:
        model = build_model(ModelConfig(name=name), data_cfg)
        x, workload, _ = _batch()
        out = model(x, workload) if name == "aeromind_capsnet" else model(x)
        assert out["workload_logits"].shape == (BATCH, 3)


# --- parameter counts (measured, see src/models/README.md for the ~720k
# README-target vs ~460k measured discrepancy explanation) --------------


def test_capsnet_parameter_count_matches_measured_value():
    n_params = sum(p.numel() for p in AeroMindCapsNet().parameters())
    assert 400_000 <= n_params <= 500_000


def test_cnn_lstm_parameter_count_is_reasonable():
    n_params = sum(p.numel() for p in AeroMindCNNLSTM().parameters())
    assert 100_000 <= n_params <= 400_000


def test_eegnet_parameter_count_is_compact():
    n_params = sum(p.numel() for p in AeroMindEEGNet().parameters())
    assert 10_000 <= n_params <= 150_000


# --- losses ---------------------------------------------------------------


def test_margin_loss_lower_for_correct_high_confidence_prediction():
    targets = torch.tensor([0, 1])
    confident_correct = torch.tensor([[0.95, 0.05, 0.05], [0.05, 0.95, 0.05]])
    confident_wrong = torch.tensor([[0.05, 0.95, 0.05], [0.95, 0.05, 0.05]])
    loss_correct = margin_loss(confident_correct, targets)
    loss_wrong = margin_loss(confident_wrong, targets)
    assert loss_correct.item() < loss_wrong.item()


def test_multi_task_loss_capsnet_includes_reconstruction_component():
    x, workload, fatigue = _batch()
    model = AeroMindCapsNet()
    out = model(x, workload_target=workload)
    total, components = multi_task_loss(out, workload, fatigue, TrainConfig())
    assert total.item() > 0
    assert "reconstruction_loss" in components
    assert set(components) == {"workload_loss", "fatigue_loss", "reconstruction_loss", "total_loss"}


def test_multi_task_loss_baseline_has_no_reconstruction_component():
    x, workload, fatigue = _batch()
    model = AeroMindCNNLSTM()
    out = model(x)
    total, components = multi_task_loss(out, workload, fatigue, TrainConfig())
    assert total.item() > 0
    assert "reconstruction_loss" not in components


# --- gradient flow ----------------------------------------------------------


def test_gradient_flow_no_nans_all_models():
    x, workload, fatigue = _batch()
    for name, model in [
        ("capsnet", AeroMindCapsNet()),
        ("cnn_lstm", AeroMindCNNLSTM()),
        ("eegnet", AeroMindEEGNet()),
    ]:
        model.zero_grad()
        out = model(x, workload_target=workload) if name == "capsnet" else model(x)
        loss, _ = multi_task_loss(out, workload, fatigue, TrainConfig())
        loss.backward()
        for pname, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{name}.{pname} has non-finite gradient"
