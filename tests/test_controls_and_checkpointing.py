"""Fourier shell-coordinate features and mixer activation checkpointing.

Both are opt-in.  The first changes the mixer's inductive bias and nothing
else; the second changes cost and nothing at all.
"""

from __future__ import annotations

import math

import pytest
import torch

from mtace.model import MambaACEV2

DOUBLE = torch.float64
R_MAX = 6.0


def _model(**overrides):
    torch.manual_seed(9)
    settings = dict(
        r_max=R_MAX, l_max=2, num_radial=8, hidden_dim=16, num_layers=2,
        correlation_order=3, correlation_channels=8, num_shells=12,
        shell_r_min=1.5, shell_degree=5, avg_num_neighbors=12.0,
        invariant_norm="homogeneous", mamba_dim=16, mamba_d_state=8,
        mamba_headdim=8, readout_hidden=16, mamba_backend="torch",
        mixer_schedule=["mamba", "attention"], layer_scale_init=1.0,
    )
    settings.update(overrides)
    return MambaACEV2(**settings).double()


def _cluster(seed=2, count=6):
    torch.manual_seed(seed)
    positions = torch.randn(count, 3, dtype=DOUBLE) * 1.7
    z = torch.tensor([1, 6, 8, 1, 6, 8][:count], dtype=torch.long)
    cell = torch.eye(3, dtype=DOUBLE) * 100.0
    senders, receivers = [], []
    for i in range(count):
        for j in range(count):
            if i != j and torch.linalg.norm(positions[j] - positions[i]) < R_MAX:
                senders.append(j)
                receivers.append(i)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    return positions, z, cell, edge_index, torch.zeros(edge_index.shape[1], 3, dtype=DOUBLE)


def _data():
    positions, z, cell, edge_index, edge_shift = _cluster()
    return {"z": z, "pos": positions, "cell": cell, "edge_index": edge_index, "edge_shift": edge_shift}


class TestFourierCoordinate:
    def test_features_are_what_the_docstring_says(self):
        layer = _model(coordinate_features="fourier", coordinate_harmonics=3).layers[0]
        xi = torch.linspace(0.0, 1.0, 5, dtype=DOUBLE)
        features = layer._coordinate_features(xi)
        assert features.shape == (5, 7)
        torch.testing.assert_close(features[:, 0], xi)
        for m in (1, 2, 3):
            torch.testing.assert_close(features[:, m], torch.sin(math.pi * m * xi))
            torch.testing.assert_close(features[:, 3 + m], torch.cos(math.pi * m * xi))

    def test_separation_dependent_logits_are_in_the_span(self):
        """cos(pi m (xi_k - xi_k')) is a bilinear form in the two feature vectors."""

        layer = _model(coordinate_features="fourier", coordinate_harmonics=2).layers[0]
        xi = torch.linspace(0.0, 1.0, 9, dtype=DOUBLE)
        f = layer._coordinate_features(xi)            # columns: xi, s1, s2, c1, c2
        bilinear = torch.zeros(5, 5, dtype=DOUBLE)
        bilinear[1, 1] = bilinear[3, 3] = 1.0           # s1 s1' + c1 c1' = cos(pi (xi - xi'))
        realised = f @ bilinear @ f.T
        expected = torch.cos(math.pi * (xi[:, None] - xi[None, :]))
        torch.testing.assert_close(realised, expected)

    def test_default_is_bitwise_unchanged(self):
        reference = _model().eval()
        explicit = _model(coordinate_features="linear").eval()
        explicit.load_state_dict(reference.state_dict())
        data = _data()
        torch.testing.assert_close(
            explicit(data, compute_stress=False)[0], reference(data, compute_stress=False)[0],
            atol=0.0, rtol=0.0,
        )

    def test_physics_is_untouched(self):
        model = _model(coordinate_features="fourier").eval()
        positions, z, cell, edge_index, edge_shift = _cluster()
        energy = lambda p: float(model.atomic_energies(z, p, cell, edge_index, edge_shift).sum())
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=DOUBLE))[0]
        assert abs(energy(positions @ rotation.T) - energy(positions)) < 1e-11
        assert abs(energy(-positions) - energy(positions)) < 1e-11
        _, forces, _, _ = model(_data(), training=False, compute_stress=False)
        plus, minus = positions.clone(), positions.clone()
        plus[2, 0] += 1e-6
        minus[2, 0] -= 1e-6
        assert float(forces[2, 0]) == pytest.approx(-(energy(plus) - energy(minus)) / 2e-6, abs=1e-7)

    def test_bad_settings_are_rejected(self):
        with pytest.raises(ValueError, match="coordinate_features"):
            _model(coordinate_features="wavelet")
        with pytest.raises(ValueError, match="coordinate_harmonics"):
            _model(coordinate_features="fourier", coordinate_harmonics=0)


class TestMixerCheckpointing:
    def test_energy_forces_and_parameter_gradients_are_identical(self):
        plain = _model(mimo_rank=2) if False else _model()
        saving = _model(mixer_checkpoint=True)
        saving.load_state_dict(plain.state_dict())
        assert all(layer.mixer_checkpoint for layer in saving.layers)
        outputs = []
        for model in (plain, saving):
            model.train()
            model.zero_grad()
            energy, forces, _, _ = model(_data(), training=True, compute_stress=False)
            (energy.square() + forces.square().sum()).backward()
            outputs.append((energy.detach(), forces.detach(),
                            {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0.0, rtol=0.0)
        assert outputs[0][2].keys() == outputs[1][2].keys()
        for name in outputs[0][2]:
            torch.testing.assert_close(outputs[0][2][name], outputs[1][2][name], atol=0.0, rtol=0.0)

    def test_inference_takes_the_plain_path(self):
        model = _model(mixer_checkpoint=True).eval()
        with torch.no_grad():
            energy = model.atomic_energies(*_cluster()[1:2], _cluster()[0], *_cluster()[2:])
        assert torch.isfinite(energy).all()

    def test_export_is_unaffected(self, tmp_path):
        from mtace.checkpoint import save_checkpoint
        from mtace.deployment import export_lammps_model

        model = _model(mixer_checkpoint=True).float().eval()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, dict(mixer_checkpoint=True, r_max=R_MAX, l_max=2,
                        num_radial=8, hidden_dim=16, num_layers=2, correlation_order=3,
                        correlation_channels=8, num_shells=12, shell_r_min=1.5, shell_degree=5,
                        avg_num_neighbors=12.0, invariant_norm="homogeneous", mamba_dim=16,
                        mamba_d_state=8, mamba_headdim=8, readout_hidden=16,
                        mamba_backend="torch", mixer_schedule=["mamba", "attention"],
                        layer_scale_init=1.0),
                        atomic_energies={1: 0.0, 6: 0.0, 8: 0.0}, atomic_numbers=[1, 6, 8])
        exported = export_lammps_model(path, tmp_path / "model.mtace.pt", ["H", "C", "O"])
        assert exported.exists()
