import copy
import math

import pytest
import torch

from mtace.model import MambaACE
from mtace.optim import (
    DEFAULT_NS_COEFFICIENTS,
    MuonWithAuxAdamW,
    adjusted_muon_learning_rate,
    build_optimizer,
    get_muon_param_groups,
    muon_update,
    zeropower_via_newton_schulz5,
)


def _reference_zeropower(matrix, steps=5, eps=1.0e-7):
    update = matrix.clone()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.T
    update = update / max(float(torch.linalg.vector_norm(update)), eps)
    a, b, c = DEFAULT_NS_COEFFICIENTS
    for _ in range(steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * gram @ gram) @ update
    return update.T if transposed else update


def _official_bfloat16_zeropower(matrix, steps=5, eps=1.0e-7):
    update = matrix.bfloat16()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.T
    update.div_(update.norm().clamp(min=eps))
    a, b, c = DEFAULT_NS_COEFFICIENTS
    for _ in range(steps):
        gram = update @ update.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        update = torch.addmm(update, gram_update, update, beta=a)
    return update.T if transposed else update


def _official_reference_step(
    parameter,
    gradient,
    momentum_buffer,
    *,
    lr,
    weight_decay,
    momentum,
    nesterov,
    ns_steps,
    adjust_lr_fn,
):
    momentum_buffer.lerp_(gradient, 1.0 - momentum)
    update = (
        gradient.lerp(momentum_buffer, momentum)
        if nesterov
        else momentum_buffer
    )
    update = _official_bfloat16_zeropower(update, steps=ns_steps)
    rows, columns = parameter.shape
    if adjust_lr_fn == "original":
        scale = math.sqrt(max(1.0, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        scale = 0.2 * math.sqrt(max(rows, columns))
    elif adjust_lr_fn == "spectral_unclamped":
        scale = math.sqrt(rows / columns)
    else:
        raise AssertionError("unsupported reference scaling")
    parameter.mul_(1.0 - lr * weight_decay)
    parameter.add_(update, alpha=-lr * scale)


def _small_model():
    torch.manual_seed(17)
    return MambaACE(
        r_max=4.0,
        l_max=1,
        num_radial=3,
        hidden_dim=6,
        num_layers=1,
        correlation_order=3,
        correlation_channels=3,
        mamba_dim=8,
        mamba_d_state=4,
        mamba_headdim=4,
        mamba_backend="torch",
        ffn_hidden=12,
    )


def _small_data():
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8]]
    )
    senders, receivers = [], []
    for receiver in range(3):
        for sender in range(3):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    return {
        "z": torch.tensor([8, 1, 1], dtype=torch.long),
        "pos": positions,
        "cell": torch.eye(3) * 8.0,
        "edge_index": edge_index,
        "edge_shift": torch.zeros((edge_index.shape[1], 3)),
    }


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(4, 3)
        self.scale = torch.nn.Parameter(torch.ones(3))

    def forward(self, inputs):
        return (torch.tanh(self.hidden(inputs)) * self.scale).square().mean()


def test_newton_schulz_matches_independent_reference_and_transpose():
    torch.manual_seed(1)
    matrix = torch.randn(7, 5, dtype=torch.float64)
    actual = zeropower_via_newton_schulz5(matrix, precision="parameter")
    expected = _reference_zeropower(matrix)
    torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(
        zeropower_via_newton_schulz5(matrix.T, precision="parameter"),
        actual.T,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_zero_matrix_is_finite_and_stays_zero():
    result = zeropower_via_newton_schulz5(torch.zeros(4, 3))
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_muon_momentum_matches_canonical_normalized_ema_convention():
    torch.manual_seed(2)
    gradient = torch.randn(5, 4)
    momentum = torch.randn_like(gradient) * 0.1
    expected_momentum = 0.95 * momentum + 0.05 * gradient
    expected_direction = 0.05 * gradient + 0.95 * expected_momentum
    expected = zeropower_via_newton_schulz5(expected_direction)
    actual = muon_update(gradient, momentum, momentum=0.95, nesterov=True)
    torch.testing.assert_close(momentum, expected_momentum)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "adjust_lr_fn", ["original", "match_rms_adamw", "spectral_unclamped"]
)
def test_bfloat16_mode_matches_official_pytorch_equations(adjust_lr_fn):
    torch.manual_seed(3)
    initial = torch.randn(7, 5)
    ours = torch.nn.Parameter(initial.clone())
    reference = torch.nn.Parameter(initial.clone())
    ours_optimizer = MuonWithAuxAdamW(
        [
            {
                "params": [ours],
                "use_muon": True,
                "lr": 0.02,
                "weight_decay": 0.01,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "ns_precision": "bfloat16",
                "adjust_lr_fn": adjust_lr_fn,
            }
        ]
    )
    reference_momentum = torch.zeros_like(reference)
    for _ in range(2):
        gradient = torch.randn_like(initial)
        ours.grad = gradient.clone()
        ours_optimizer.step()
        with torch.no_grad():
            _official_reference_step(
                reference,
                gradient,
                reference_momentum,
                lr=0.02,
                weight_decay=0.01,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                adjust_lr_fn=adjust_lr_fn,
            )
    torch.testing.assert_close(ours, reference, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        ours_optimizer.state[ours]["momentum_buffer"],
        reference_momentum,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch Muon unavailable")
def test_bfloat16_mode_matches_installed_pytorch_muon():
    torch.manual_seed(31)
    initial = torch.randn(7, 5)
    ours = torch.nn.Parameter(initial.clone())
    official = torch.nn.Parameter(initial.clone())
    ours_optimizer = MuonWithAuxAdamW(
        [{"params": [ours], "use_muon": True, "ns_precision": "bfloat16"}]
    )
    official_optimizer = torch.optim.Muon(
        [official],
        lr=1.0e-3,
        weight_decay=0.0,
        adjust_lr_fn="match_rms_adamw",
    )
    gradient = torch.randn_like(initial)
    ours.grad = gradient.clone()
    official.grad = gradient.clone()
    ours_optimizer.step()
    official_optimizer.step()
    torch.testing.assert_close(ours, official, atol=0.0, rtol=0.0)


def test_auxiliary_group_matches_torch_adamw():
    torch.manual_seed(4)
    initial = torch.randn(6)
    ours = torch.nn.Parameter(initial.clone())
    reference = torch.nn.Parameter(initial.clone())
    ours_optimizer = MuonWithAuxAdamW(
        [
            {
                "params": [ours],
                "use_muon": False,
                "lr": 3.0e-4,
                "betas": (0.9, 0.95),
                "eps": 1.0e-10,
                "weight_decay": 1.0e-2,
            }
        ]
    )
    reference_optimizer = torch.optim.AdamW(
        [reference],
        lr=3.0e-4,
        betas=(0.9, 0.95),
        eps=1.0e-10,
        weight_decay=1.0e-2,
    )
    for _ in range(3):
        gradient = torch.randn_like(initial)
        ours.grad = gradient.clone()
        reference.grad = gradient.clone()
        ours_optimizer.step()
        reference_optimizer.step()
    torch.testing.assert_close(ours, reference, atol=2.0e-7, rtol=2.0e-7)


@pytest.mark.parametrize("use_muon", [True, False])
def test_none_gradient_skips_update_weight_decay_and_state(use_muon):
    parameter = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = MuonWithAuxAdamW(
        [
            {
                "params": [parameter],
                "use_muon": use_muon,
                "lr": 0.02,
                "weight_decay": 0.1,
            }
        ]
    )
    before = parameter.detach().clone()
    optimizer.step()
    torch.testing.assert_close(parameter, before)
    assert parameter not in optimizer.state


def test_hidden_grouping_targets_all_internal_matrix_operators_by_role():
    model = _small_model()
    groups = get_muon_param_groups(model, 1.0e-3, 1.0e-5)
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    muon_group = next(group for group in groups if group["use_muon"])
    names = {by_id[id(parameter)] for parameter in muon_group["params"]}
    assert "layers.0.mixer.forward_direction.in_proj.weight" in names
    assert "layers.0.mixer.forward_direction.out_proj.weight" in names
    assert "layers.0.mixer.backward_direction.in_proj.weight" in names
    assert "layers.0.mixer.backward_direction.out_proj.weight" in names
    assert "layers.0.scalar_ffn_in.weight" in names
    assert "layers.0.scalar_ffn_out.weight" in names
    assert "ace.radial_net.layer0.weight" in names
    assert "ace.radial_net.layer1.weight" in names
    assert "ace.center_proj.weight" in names
    assert "layers.0.node_context.weight" in names
    assert "layers.0.token_input.weight" in names
    assert "layers.0.gate_projection.weight" in names
    assert "readout.network.0.weight" in names
    assert "species_embedding.weight" not in names
    assert "layers.0.kind_embedding.weight" not in names
    assert "layers.0.coordinate_projection.weight" not in names
    assert "readout.network.2.weight" not in names
    assert not any(name.endswith("B_bias") or name.endswith("C_bias") for name in names)

    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_mamba_dynamics_and_biases_are_in_no_decay_auxiliary_group():
    model = _small_model()
    groups = get_muon_param_groups(model, 1.0e-3, 1.0e-4)
    no_decay = {
        id(parameter)
        for group in groups
        if not group["use_muon"] and group["weight_decay"] == 0.0
        for parameter in group["params"]
    }
    for direction in (
        model.layers[0].mixer.forward_direction,
        model.layers[0].mixer.backward_direction,
    ):
        assert id(direction.dt_bias) in no_decay
        assert id(direction.D) in no_decay
        assert id(direction.B_bias) in no_decay
        assert id(direction.C_bias) in no_decay


def test_all_supported_rectangular_learning_rate_rules():
    expected = 0.02 * math.sqrt(7.0 / 5.0)
    assert adjusted_muon_learning_rate(0.02, (7, 5), "original") == pytest.approx(
        expected
    )
    assert adjusted_muon_learning_rate(0.02, (5, 7), "original") == 0.02
    assert adjusted_muon_learning_rate(
        0.02, (7, 5), "match_rms_adamw"
    ) == pytest.approx(0.004 * math.sqrt(7.0))
    assert adjusted_muon_learning_rate(
        0.02, (7, 5), "spectral_unclamped"
    ) == pytest.approx(0.02 * math.sqrt(7.0 / 5.0))


def test_match_rms_scaling_gives_shape_independent_orthogonal_update_rms():
    torch.manual_seed(9)
    for rows, columns in ((7, 5), (5, 7), (8, 8)):
        matrix = torch.randn(rows, columns, dtype=torch.float64)
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        polar = u @ vh
        scaled = adjusted_muon_learning_rate(
            1.0, (rows, columns), "match_rms_adamw"
        ) * polar
        assert scaled.square().mean().sqrt().item() == pytest.approx(
            0.2, abs=1.0e-12
        )


def test_auto_precision_is_scientific_on_cpu():
    single = zeropower_via_newton_schulz5(torch.randn(4, 3))
    double = zeropower_via_newton_schulz5(
        torch.randn(4, 3, dtype=torch.float64)
    )
    assert single.dtype == torch.float32
    assert double.dtype == torch.float64


def test_hidden_routing_overrides_are_explicit_and_exclusion_wins():
    model = _small_model()
    groups = get_muon_param_groups(
        model,
        1.0e-3,
        1.0e-5,
        include_patterns="readout.network.2.weight",
        exclude_patterns="ace.radial_net.layer0.weight",
    )
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    names = {
        by_id[id(parameter)]
        for group in groups
        if group["use_muon"]
        for parameter in group["params"]
    }
    assert "readout.network.2.weight" in names
    assert "ace.radial_net.layer0.weight" not in names


def test_configuration_uses_independent_scalable_defaults():
    optimizer = build_optimizer(
        _small_model(),
        {"optimizer": "muon", "learning_rate": 1.0e-3, "weight_decay": 1.0e-5},
    )
    muon_group = next(group for group in optimizer.param_groups if group["use_muon"])
    assert muon_group["adjust_lr_fn"] == "match_rms_adamw"
    assert muon_group["ns_precision"] == "auto"


def test_optimizer_state_round_trip_is_exact():
    torch.manual_seed(8)
    model_a = TinyNetwork()
    optimizer_a = build_optimizer(
        model_a,
        {
            "optimizer": "muon",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "muon_parameter_mode": "all_matrices",
        },
    )
    inputs = torch.randn(5, 4)
    model_a(inputs).backward()
    optimizer_a.step()

    model_b = TinyNetwork()
    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))
    optimizer_b = build_optimizer(
        model_b,
        {
            "optimizer": "muon",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "muon_parameter_mode": "all_matrices",
        },
    )
    optimizer_b.load_state_dict(copy.deepcopy(optimizer_a.state_dict()))
    optimizer_a.zero_grad(set_to_none=True)
    optimizer_b.zero_grad(set_to_none=True)
    model_a(inputs).backward()
    model_b(inputs).backward()
    optimizer_a.step()
    optimizer_b.step()
    for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(parameter_a, parameter_b, atol=0.0, rtol=0.0)


def test_muon_steps_a_force_training_graph_with_finite_parameters():
    model = _small_model().train()
    optimizer = build_optimizer(
        model,
        {
            "optimizer": "muon",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-6,
            "muon_parameter_mode": "hidden",
        },
    )
    target = model.layers[0].mixer.forward_direction.in_proj.weight
    before = target.detach().clone()
    energy, forces, _, _ = model(
        _small_data(), training=True, compute_stress=False
    )
    loss = energy.square() + forces.square().mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert torch.isfinite(target).all()
    assert not torch.equal(target, before)
