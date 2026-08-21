"""Symmetry-exact ACE features for the MTACE potential."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value > 20.0, value, value + torch.log(-torch.expm1(-value)))


def _regular_spherical_bessel_channels(
    distances: torch.Tensor,
    frequencies: torch.Tensor,
    r_max: float,
    normalization: float,
) -> torch.Tensor:
    """Evaluate normalized ``sin(k r) / r`` channels at their regular limit."""

    distance_column = distances.unsqueeze(-1)
    scaled = distance_column * frequencies / r_max
    # Preserve TRACE-v2 arithmetic away from the removable singularity. The
    # even Taylor branch is the analytic continuation of sin(z)/z and, unlike
    # torch.sinc on some supported Torch releases, has finite second derivatives
    # exactly at zero for force/stress training.
    near_zero = distance_column.abs() <= 1.0e-8 * r_max
    safe_distance = torch.where(near_zero, torch.ones_like(distance_column), distance_column)
    direct = normalization * (torch.sin(scaled) / safe_distance)
    squared = scaled.square()
    sine_over_argument = 1.0 - squared / 6.0 + squared.square() / 120.0
    sine_over_argument = sine_over_argument - squared.pow(3) / 5040.0
    regular = normalization * (frequencies / r_max) * sine_over_argument
    return torch.where(near_zero, regular, direct)


class SmoothPolynomialCutoff(nn.Module):
    """Compact C2 quintic envelope with value and two derivatives zero at r_max."""

    def __init__(self, r_max: float):
        super().__init__()
        if r_max <= 0.0:
            raise ValueError("r_max must be positive")
        self.r_max = float(r_max)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(distances / self.r_max, min=0.0, max=1.0)
        envelope = (1.0 - x).pow(3) * (1.0 + 3.0 * x + 6.0 * x.pow(2))
        return torch.where(distances < self.r_max, envelope, torch.zeros_like(envelope))


class GaussianRadialBasis(nn.Module):
    """Ordered Gaussian shells, multiplied by a compact C2 cutoff."""

    def __init__(
        self,
        r_max: float,
        num_radial: int,
        width_factor: float = 0.7,
        trainable: bool = False,
    ):
        super().__init__()
        if num_radial < 1:
            raise ValueError("num_radial must be positive")
        if width_factor <= 0.0:
            raise ValueError("width_factor must be positive")
        self.r_max = float(r_max)
        self.cutoff = SmoothPolynomialCutoff(r_max)
        centers = torch.linspace(0.0, r_max, num_radial + 2)[1:-1]
        spacing = r_max / float(num_radial + 1)
        raw_widths = torch.full_like(centers, math.log(math.expm1(width_factor * spacing)))
        if trainable:
            self.centers = nn.Parameter(centers)
            self.raw_widths = nn.Parameter(raw_widths)
            self.centers._no_weight_decay = True
            self.raw_widths._no_weight_decay = True
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("raw_widths", raw_widths)

    @property
    def widths(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_widths).clamp_min(1.0e-4)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances[:, None] - self.centers[None, :]
        basis = torch.exp(-0.5 * (diff / self.widths[None, :]).pow(2))
        return self.cutoff(distances)[:, None] * basis


class BesselRadialBasis(nn.Module):
    """Frequency-ordered spherical Bessel channels with a compact C2 cutoff."""

    def __init__(self, r_max: float, num_radial: int, trainable: bool = False):
        super().__init__()
        if num_radial < 1:
            raise ValueError("num_radial must be positive")
        self.r_max = float(r_max)
        self.cutoff = SmoothPolynomialCutoff(r_max)
        frequencies = torch.arange(1, num_radial + 1, dtype=torch.get_default_dtype()) * math.pi
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
            self.frequencies._no_weight_decay = True
        else:
            self.register_buffer("frequencies", frequencies)
        self.normalization = math.sqrt(2.0 / r_max)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        values = _regular_spherical_bessel_channels(
            distances, self.frequencies, self.r_max, self.normalization
        )
        return self.cutoff(distances)[:, None] * values


def make_radial_basis(
    basis_type: str,
    r_max: float,
    num_radial: int,
    trainable: bool,
    gaussian_width: float,
) -> nn.Module:
    basis_type = basis_type.lower()
    if basis_type == "gaussian":
        return GaussianRadialBasis(r_max, num_radial, gaussian_width, trainable)
    if basis_type == "bessel":
        return BesselRadialBasis(r_max, num_radial, trainable)
    raise ValueError(f"Unsupported radial basis type: {basis_type}")


class V2BesselBasis(nn.Module):
    """Bessel basis with parameter names and equations matching TRACE-v2."""

    def __init__(self, r_max: float, num_radial: int, trainable: bool = False):
        super().__init__()
        if r_max <= 0.0 or num_radial < 1:
            raise ValueError("r_max and num_radial must be positive")
        self.r_max = float(r_max)
        frequency = torch.arange(1, num_radial + 1, dtype=torch.get_default_dtype()) * math.pi
        if trainable:
            gaps = torch.full_like(frequency, math.pi)
            self.raw_frequency_gaps = nn.Parameter(_inverse_softplus(gaps))
            self.raw_frequency_gaps._no_weight_decay = True
        else:
            self.register_buffer("freq", frequency)
        self.norm = math.sqrt(2.0 / self.r_max)

    @property
    def frequencies(self) -> torch.Tensor:
        if hasattr(self, "raw_frequency_gaps"):
            gaps = torch.nn.functional.softplus(self.raw_frequency_gaps).clamp_min(1.0e-6)
            return torch.cumsum(gaps, dim=0)
        return self.freq

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = torch.clamp(distances, min=0.0)
        return _regular_spherical_bessel_channels(
            distances,
            self.frequencies,
            self.r_max,
            self.norm,
        )


class V2GaussianBasis(nn.Module):
    """Gaussian basis with parameter names and equations matching TRACE-v2."""

    def __init__(
        self,
        r_max: float,
        num_radial: int,
        width_factor: float = 0.5,
        trainable: bool = False,
    ):
        super().__init__()
        if r_max <= 0.0 or num_radial < 1 or width_factor <= 0.0:
            raise ValueError("r_max, num_radial, and width_factor must be positive")
        self.r_max = float(r_max)
        centers = torch.linspace(
            0.0, self.r_max, num_radial + 2, dtype=torch.get_default_dtype()
        )[1:-1]
        spacing = centers[1] - centers[0] if num_radial > 1 else self.r_max / 2.0
        widths = torch.full_like(centers, width_factor * spacing)
        if trainable:
            center_gaps = torch.full(
                (num_radial + 1,),
                self.r_max / float(num_radial + 1),
                dtype=centers.dtype,
            )
            self.raw_center_gaps = nn.Parameter(_inverse_softplus(center_gaps))
            self.raw_widths = nn.Parameter(_inverse_softplus(widths))
            self.raw_center_gaps._no_weight_decay = True
            self.raw_widths._no_weight_decay = True
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    @property
    def positive_widths(self) -> torch.Tensor:
        if hasattr(self, "raw_widths"):
            return torch.nn.functional.softplus(self.raw_widths).clamp_min(1.0e-6)
        return self.widths

    @property
    def radial_centers(self) -> torch.Tensor:
        if hasattr(self, "raw_center_gaps"):
            gaps = torch.nn.functional.softplus(self.raw_center_gaps).clamp_min(1.0e-6)
            return self.r_max * torch.cumsum(gaps, dim=0)[:-1] / gaps.sum()
        return self.centers

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = torch.clamp(distances, min=0.0)
        difference = distances.unsqueeze(-1) - self.radial_centers
        return torch.exp(-0.5 * (difference / self.positive_widths).pow(2))


class V2SmoothRadialBasis(nn.Module):
    """TRACE-v2 radial basis multiplied by its compact C2 envelope."""

    def __init__(
        self,
        r_max: float,
        num_radial: int,
        basis_type: str = "bessel",
        trainable: bool = False,
        gaussian_width: float = 0.5,
    ):
        super().__init__()
        self.cutoff = SmoothPolynomialCutoff(r_max)
        basis_type = basis_type.lower()
        if basis_type == "bessel":
            self.basis = V2BesselBasis(r_max, num_radial, trainable=trainable)
        elif basis_type == "gaussian":
            self.basis = V2GaussianBasis(
                r_max,
                num_radial,
                width_factor=gaussian_width,
                trainable=trainable,
            )
        else:
            raise ValueError(f"Unsupported radial basis type: {basis_type}")

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return self.cutoff(distances).unsqueeze(-1) * self.basis(distances)


class ACEV2Descriptor(nn.Module):
    """Exact standalone implementation of the TRACE-v2 ACE frontend."""

    def __init__(
        self,
        r_max: float,
        l_max: int,
        num_radial: int,
        hidden_dim: int,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        gaussian_width: float = 0.5,
        radial_mlp_hidden: int = 32,
        radial_mlp_layers: int = 2,
        avg_num_neighbors: float = 1.0,
    ):
        super().__init__()
        if r_max <= 0.0:
            raise ValueError("r_max must be positive")
        if l_max < 0 or num_radial < 1 or hidden_dim < 1 or correlation_channels < 1:
            raise ValueError(
                "l_max must be non-negative and radial, hidden, and correlation dimensions positive"
            )
        if correlation_order < 2 or correlation_order > 6:
            raise ValueError("correlation_order must be between 2 and 6")
        avg_num_neighbors = float(avg_num_neighbors)
        if not math.isfinite(avg_num_neighbors) or avg_num_neighbors <= 0.0:
            raise ValueError("avg_num_neighbors must be positive and finite")
        self.r_max = float(r_max)
        self.hidden_dim = int(hidden_dim)
        self.correlation_order = int(correlation_order)
        # Density normalization.  Without it the one-particle density A_i grows
        # linearly with coordination and the order-nu correlation grows as
        # (n_neigh)^(nu-1), so activations depend strongly on local density and
        # transfer poorly between phases.  Dividing the edge feature once, before
        # any pooling, rescales A_i and every shell token T_ik by the same
        # constant and therefore preserves ``sum_k T_ik = A_i`` exactly.
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_normalization = 1.0 / math.sqrt(avg_num_neighbors)

        output_irreps = []
        correlation_irreps = []
        for ell in range(l_max + 1):
            output_mul = hidden_dim if ell == 0 else hidden_dim // (2 if ell == 1 else 4)
            corr_mul = (
                correlation_channels
                if ell == 0
                else correlation_channels // (2 if ell == 1 else 4)
            )
            output_irreps.append((max(1, output_mul), (ell, (-1) ** ell)))
            correlation_irreps.append((max(1, corr_mul), (ell, (-1) ** ell)))

        self.irreps_out = o3.Irreps(output_irreps)
        self.irreps_correlation = o3.Irreps(correlation_irreps)
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        self.irreps_node = o3.Irreps(f"{hidden_dim}x0e")
        self.irreps_out_dim = int(self.irreps_out.dim)
        self.irreps_correlation_dim = int(self.irreps_correlation.dim)

        self.cutoff = SmoothPolynomialCutoff(r_max)
        self.radial_basis = V2SmoothRadialBasis(
            r_max,
            num_radial,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_sh,
            normalize=True,
            normalization="component",
        )
        self.tp_density = o3.FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_correlation,
            internal_weights=False,
            shared_weights=False,
        )
        radial_mlp_hidden = max(1, int(radial_mlp_hidden))
        radial_mlp_layers = max(1, int(radial_mlp_layers))
        mlp_sizes = (
            [num_radial]
            + [radial_mlp_hidden] * (radial_mlp_layers - 1)
            + [self.tp_density.weight_numel]
        )
        self.radial_net = FullyConnectedNet(mlp_sizes, torch.nn.functional.silu)
        self.contractions = nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    self.irreps_correlation,
                    self.irreps_correlation,
                    self.irreps_correlation,
                    internal_weights=True,
                    shared_weights=True,
                )
                for _ in range(self.correlation_order - 2)
            ]
        )
        self.order_mix = nn.ModuleList(
            [
                o3.Linear(self.irreps_correlation, self.irreps_out)
                for _ in range(self.correlation_order - 1)
            ]
        )
        self.center_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _density(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
    ):
        sender, receiver = edge_index[0], edge_index[1]
        radial = self.radial_basis(edge_len)
        harmonics = self.sh(edge_vec)
        weights = self.radial_net(radial)
        edge_features = self.tp_density(node_attrs[sender], harmonics, weights)
        if self.edge_normalization != 1.0:
            edge_features = edge_features * self.edge_normalization
        density = torch.zeros(
            node_attrs.shape[0],
            self.irreps_correlation_dim,
            device=node_attrs.device,
            dtype=edge_features.dtype,
        )
        density.index_add_(0, receiver, edge_features)
        return density, edge_features, self.cutoff(edge_len)

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
        return_edge_features: bool = False,
    ):
        density, edge_features, cutoff = self._density(
            node_attrs, edge_index, edge_vec, edge_len
        )
        non_scalar_dim = self.irreps_out_dim - self.hidden_dim
        center = torch.cat(
            (
                self.center_proj(node_attrs),
                node_attrs.new_zeros((node_attrs.shape[0], non_scalar_dim)),
            ),
            dim=-1,
        )
        correlation = density
        output = center + self.order_mix[0](correlation)
        for contraction, mix in zip(self.contractions, self.order_mix[1:]):
            correlation = contraction(correlation, density)
            output = output + mix(correlation)
        if return_edge_features:
            return output, edge_features, cutoff
        return output


_BSPLINE_TRUNCATED_POWER_COEFFICIENTS = {
    # degree -> ((shift, coefficient), ...) for a >= 0, from
    #   B_d(a) = (1/d!) sum_{k=0}^{m-1} (-1)^k C(d+1, k) (m - k - a)_+^d,  m = (d+1)/2.
    # Only k < m contributes for a >= 0 because (m - k - a)_+ vanishes otherwise.
    # Verified against the exact rational values B3 = 2/3, 1/6 and
    # B5 = 11/20, 13/60, 1/120 at a = 0, 1, 2.
    3: ((2.0, 1.0), (1.0, -4.0)),
    5: ((3.0, 1.0), (2.0, -6.0), (1.0, 15.0)),
}
_BSPLINE_FACTORIAL = {3: 6.0, 5: 120.0}


def cardinal_bspline(argument: torch.Tensor, degree: int) -> torch.Tensor:
    """Centered cardinal B-spline of odd ``degree``, evaluated by truncated powers.

    The spline is piecewise polynomial of degree ``d`` and therefore exactly
    :math:`C^{d-1}`: cubic shells give continuous second derivatives (energies,
    forces, harmonic force constants) but a *discontinuous third derivative*, and
    quintic shells give continuous fourth derivatives.

    The third-derivative jump of the cubic spline at a knot is exactly ``6`` in
    the reduced coordinate, hence

        [d^3 W / dr^3] = 6 ((L - 1) / (r_max - r_min))^3

    in physical units, which *grows as* ``L^3`` under mesh refinement.  Third-order
    interatomic force constants -- and therefore three-phonon scattering, lattice
    thermal conductivity, Grueneisen parameters and anharmonic phonon
    renormalization -- inherit that discontinuity.  Use ``degree=5`` for any
    anharmonic property; see ``docs/DERIVATIVE_CONTRACT.md``.

    Truncated powers keep the evaluation branch-free and numerically benign: the
    largest intermediate is ``3^5 = 243`` against a result of order one, so the
    cancellation is mild enough for float32.
    """

    try:
        terms = _BSPLINE_TRUNCATED_POWER_COEFFICIENTS[degree]
        normalization = _BSPLINE_FACTORIAL[degree]
    except KeyError as exception:
        raise ValueError("B-spline degree must be 3 or 5") from exception
    absolute = argument.abs()
    total = torch.zeros_like(absolute)
    for shift, coefficient in terms:
        total = total + coefficient * (shift - absolute).clamp_min(0.0).pow(degree)
    return total / normalization


class CompactRadialShellBasis(nn.Module):
    """Compact cardinal B-splines on a physical radial coordinate.

    Shell centers are uniformly spaced in the reduced coordinate
    ``xi = (r - r_min) / (r_max - r_min)``.  Every distance contributes to at
    most ``degree + 1`` adjacent centers, so sparse tokenization is linear in the
    edge count and independent of the number of shells.

    ``degree`` selects the smoothness of the radial pooling and therefore the
    highest energy derivative that is continuous:

    ``degree=3`` (cubic, four-point support)
        Energies, forces and harmonic force constants are continuous.  The third
        derivative jumps by ``6 ((L-1)/(r_max-r_min))^3`` at every shell radius,
        so third-order force constants are *not* usable.
    ``degree=5`` (quintic, six-point support)
        Continuous up to the fourth derivative at a 1.5x tokenizer cost.  Required
        for anharmonic properties.

    Three boundary treatments are supported.

    ``fold`` (default from architecture v9)
        The cardinal cubic B-spline satisfies ``sum_{k in Z} B3(q - k) = 1``
        exactly, and only four terms are nonzero for any ``q``.  Weights whose
        index falls outside ``[0, L-1]`` are therefore *folded* onto the nearest
        retained shell instead of being discarded.  This gives an exact
        partition of unity with no division, is :math:`C^2` for every ``r``
        including ``r < r_min`` and ``r > r_max``, and makes a nonzero
        ``r_min`` safe.
    ``renormalize`` (architecture v8)
        Out-of-range weights are dropped and the remainder is rescaled to sum
        to one.  Also :math:`C^2`, but it requires a division and cannot be
        used with ``r_min > 0`` because the reduced coordinate must be clamped.
    ``legacy``
        The v8 ``renormalize`` weights multiplied by a second compact cutoff,
        as used by architecture versions 6 and 7.

    In every mode ``sum_k W_k(r) = 1``, so the reconstruction identity
    ``sum_k T_ik = A_i`` of the manuscript holds exactly (the ``legacy`` mode
    reproduces it only up to the extra cutoff factor, by design).
    """

    def __init__(
        self,
        r_max: float,
        num_shells: int,
        shell_coupling_mode: str = "conservative",
        r_min: float = 0.0,
        boundary_mode: str = "fold",
        degree: int = 3,
    ):
        super().__init__()
        if r_max <= 0.0:
            raise ValueError("r_max must be positive")
        if num_shells < 2:
            raise ValueError("num_shells must be at least two")
        shell_coupling_mode = shell_coupling_mode.lower()
        if shell_coupling_mode not in {"conservative", "legacy"}:
            raise ValueError(
                "shell_coupling_mode must be 'conservative' or 'legacy'"
            )
        boundary_mode = boundary_mode.lower()
        if boundary_mode not in {"fold", "renormalize"}:
            raise ValueError("boundary_mode must be 'fold' or 'renormalize'")
        r_min = float(r_min)
        if not math.isfinite(r_min) or r_min < 0.0:
            raise ValueError("shell_r_min must be finite and nonnegative")
        if r_min >= float(r_max):
            raise ValueError("shell_r_min must be smaller than r_max")
        if r_min > 0.0 and boundary_mode != "fold":
            raise ValueError(
                "shell_r_min > 0 requires boundary_mode='fold'; the "
                "renormalizing boundary is not C1 at the inner edge"
            )
        degree = int(degree)
        if degree not in _BSPLINE_TRUNCATED_POWER_COEFFICIENTS:
            raise ValueError("shell_degree must be 3 (cubic) or 5 (quintic)")
        self.degree = degree
        # Support half-width m = (degree + 1) / 2; the contributing shell indices
        # for floor(q) = base are base - m + 1 ... base + m, i.e. degree + 1 of them.
        self.support_half_width = (degree + 1) // 2
        self.support_size = degree + 1
        if num_shells < self.support_size:
            raise ValueError(
                f"num_shells must be at least {self.support_size} for a "
                f"degree-{degree} shell basis"
            )
        self.r_max = float(r_max)
        self.r_min = r_min
        self.num_shells = int(num_shells)
        self.shell_coupling_mode = shell_coupling_mode
        self.boundary_mode = boundary_mode
        self.cutoff = SmoothPolynomialCutoff(r_max)
        # Reduced shell coordinate xi_k = k / (L - 1); the physical radius of
        # shell k is r_min + (r_max - r_min) * xi_k.
        self.register_buffer(
            "centers",
            torch.linspace(0.0, 1.0, num_shells, dtype=torch.get_default_dtype()),
        )

    @property
    def shell_radii(self) -> torch.Tensor:
        """Physical radius of every shell center, for diagnostics and plots."""

        return self.r_min + (self.r_max - self.r_min) * self.centers

    @property
    def reduced_spacing(self) -> float:
        """``Delta xi`` between adjacent shells in the reduced coordinate."""

        return 1.0 / float(self.num_shells - 1)

    def forward(self, distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shell indices and compact interpolation coefficients."""

        if distances.ndim != 1:
            raise ValueError("distances must have shape (num_edges,)")
        span = self.r_max - self.r_min
        steps = float(self.num_shells - 1)
        if self.boundary_mode == "fold":
            # No clamping: the folded partition of unity is exact and C2 for
            # every real coordinate, including outside [r_min, r_max].
            scaled = (distances - self.r_min) * (steps / span)
        else:
            scaled = distances.clamp(min=0.0, max=self.r_max) * (steps / self.r_max)
        base = torch.floor(scaled).to(torch.long)
        half = self.support_half_width
        offsets = torch.arange(1 - half, half + 1, device=distances.device)
        raw_indices = base[:, None] + offsets[None, :]
        arguments = scaled[:, None] - raw_indices.to(scaled.dtype)
        coefficients = cardinal_bspline(arguments, self.degree)
        if self.boundary_mode == "renormalize":
            valid = (raw_indices >= 0) & (raw_indices < self.num_shells)
            coefficients = coefficients * valid.to(scaled.dtype)
            coefficients = coefficients / coefficients.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(coefficients.dtype).tiny)
        if self.shell_coupling_mode == "legacy":
            coefficients = coefficients * self.cutoff(distances)[:, None]
        # Clamping the index folds out-of-range weights onto the nearest
        # retained shell.  Combined with the exact four-point partition of
        # unity this keeps ``sum_k W_k = 1`` without any renormalization.
        return raw_indices.clamp(0, self.num_shells - 1), coefficients

    def dense(self, distances: torch.Tensor) -> torch.Tensor:
        """Materialize shell weights for diagnostics; tokenization stays sparse."""

        indices, coefficients = self(distances)
        weights = coefficients.new_zeros((distances.shape[0], self.num_shells))
        weights.scatter_add_(1, indices, coefficients)
        return weights


class ACEV2MambaTokenizer(ACEV2Descriptor):
    """Exact v2 descriptor plus a smooth permutation-invariant tokenization."""

    def __init__(
        self,
        r_max: float,
        l_max: int,
        num_radial: int,
        hidden_dim: int,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        gaussian_width: float = 0.5,
        radial_mlp_hidden: int = 32,
        radial_mlp_layers: int = 2,
        tokenizer_type: str = "physical_shells",
        num_shells: int | None = None,
        shell_coupling_mode: str | None = None,
        shell_r_min: float = 0.0,
        shell_boundary_mode: str = "fold",
        shell_degree: int = 3,
        shell_scales: int = 1,
        avg_num_neighbors: float = 1.0,
    ):
        super().__init__(
            r_max=r_max,
            l_max=l_max,
            num_radial=num_radial,
            hidden_dim=hidden_dim,
            correlation_order=correlation_order,
            correlation_channels=correlation_channels,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            gaussian_width=gaussian_width,
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
            avg_num_neighbors=avg_num_neighbors,
        )
        tokenizer_type = tokenizer_type.lower()
        if tokenizer_type not in {"physical_shells", "legacy_basis"}:
            raise ValueError(
                "tokenizer_type must be 'physical_shells' or 'legacy_basis'"
            )
        self.tokenizer_type = tokenizer_type
        if shell_coupling_mode is None:
            shell_coupling_mode = (
                "legacy" if tokenizer_type == "legacy_basis" else "conservative"
            )
        shell_coupling_mode = shell_coupling_mode.lower()
        if shell_coupling_mode not in {"conservative", "legacy"}:
            raise ValueError(
                "shell_coupling_mode must be 'conservative' or 'legacy'"
            )
        self.shell_coupling_mode = shell_coupling_mode
        self.shell_r_min = float(shell_r_min)
        self.shell_boundary_mode = str(shell_boundary_mode).lower()
        self.shell_degree = int(shell_degree)
        # ---- dyadic multiresolution ------------------------------------------
        # Scale s uses L_s = (L_0 - 1) 2^s + 1 shells over the same physical
        # interval, so each scale is its own exact partition of unity and the
        # reconstruction identity holds *per scale*:
        #
        #     sum_{k in scale s} T^(s)_ik = A_i    for every s.
        #
        # The sequence is then ordered coarse -> fine, which is a canonical
        # physical ordering with a renormalisation reading: the recurrence
        # carries information from a coarse radial description into successively
        # finer ones.  Unlike the single-scale axis, this ordering is not a basis
        # convention -- refining resolution is a directed operation.
        #
        # It also gives ``token_kind`` a job.  In the single-scale model there is
        # one kind and the embedding is a constant; here the kind *is* the scale,
        # so the mixer can tell coarse tokens from fine ones.
        shell_scales = int(shell_scales)
        if shell_scales < 1:
            raise ValueError("shell_scales must be at least one")
        self.shell_scales = shell_scales
        self.num_radial = int(num_radial)
        self.num_shells = int(num_shells if num_shells is not None else num_radial)
        if self.tokenizer_type == "legacy_basis" and self.num_shells != self.num_radial:
            raise ValueError("legacy_basis requires num_shells == num_radial")
        if self.tokenizer_type == "physical_shells":
            self.shell_counts = [
                (self.num_shells - 1) * (2**scale) + 1
                for scale in range(self.shell_scales)
            ]
            self.shell_bases = nn.ModuleList(
                [self._build_shell_basis(count) for count in self.shell_counts]
            )
            self.shell_basis = self.shell_bases[0]
        else:
            self.shell_counts = [self.num_shells]
            self.shell_bases = None
            self.shell_basis = None
        self.sequence_length = sum(self.shell_counts)
        self.num_token_kinds = len(self.shell_counts)
        self.register_buffer(
            "token_kind",
            torch.cat([
                torch.full((count,), scale, dtype=torch.long)
                for scale, count in enumerate(self.shell_counts)
            ]),
        )
        if self.tokenizer_type == "physical_shells":
            coordinate = torch.cat([b.centers.clone() for b in self.shell_bases])
        else:
            coordinate = torch.arange(1, num_radial + 1, dtype=torch.get_default_dtype())
            denominator = (
                num_radial + 1
                if radial_basis_type.lower() == "gaussian"
                else num_radial
            )
            coordinate = coordinate / float(denominator)
        self.register_buffer("token_coordinate", coordinate)

    def _build_shell_basis(self, num_shells: int) -> CompactRadialShellBasis:
        return CompactRadialShellBasis(
            self.r_max,
            num_shells,
            shell_coupling_mode=self.shell_coupling_mode,
            r_min=self.shell_r_min,
            boundary_mode=self.shell_boundary_mode,
            degree=self.shell_degree,
        )

    def set_num_shells(self, num_shells: int) -> None:
        """Re-grid the physical shell axis without touching any parameter.

        Every learned tensor in the physical-shell tokenizer and in the
        Mamba/attention/DeepSets mixers is independent of the shell count, so a
        trained model can be evaluated on a finer or coarser radial mesh.  This
        is what makes the resolution-transfer experiment of the manuscript
        possible; see :meth:`MambaACEV2.set_num_shells` for the model-level
        entry point, which also rescales the continuum-mode quadrature.
        """

        if self.tokenizer_type != "physical_shells":
            raise ValueError("only the physical-shell tokenizer can be re-gridded")
        num_shells = int(num_shells)
        if num_shells < 2:
            raise ValueError("num_shells must be at least two")
        device = self.token_coordinate.device
        dtype = self.token_coordinate.dtype
        self.num_shells = num_shells
        self.shell_counts = [
            (num_shells - 1) * (2**scale) + 1 for scale in range(self.shell_scales)
        ]
        self.shell_bases = nn.ModuleList(
            [self._build_shell_basis(count).to(device=device) for count in self.shell_counts]
        )
        self.shell_basis = self.shell_bases[0]
        self.sequence_length = sum(self.shell_counts)
        self.token_kind = torch.cat([
            torch.full((count,), scale, dtype=torch.long, device=device)
            for scale, count in enumerate(self.shell_counts)
        ])
        self.token_coordinate = torch.cat(
            [b.centers for b in self.shell_bases]
        ).to(device=device, dtype=dtype)

    def _current_token_coordinate(self) -> torch.Tensor:
        if self.tokenizer_type == "physical_shells":
            return self.token_coordinate
        basis = self.radial_basis.basis
        if isinstance(basis, V2GaussianBasis):
            return basis.radial_centers / self.r_max
        frequency = basis.frequencies
        return frequency / frequency[-1]

    def pool_edge_features(
        self,
        edge_features: torch.Tensor,
        receivers: torch.Tensor,
        edge_len: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Pool equivariant edges onto the configured radial token axis."""

        if edge_features.ndim != 2 or edge_features.shape[0] != edge_len.shape[0]:
            raise ValueError("edge_features and edge_len must share the edge axis")
        if receivers.shape != edge_len.shape or receivers.dtype != torch.long:
            raise ValueError("receivers must be a long tensor with shape (num_edges,)")
        if self.tokenizer_type == "legacy_basis":
            radial = self.radial_basis(edge_len)
            edge_shells = radial[:, :, None] * edge_features[:, None, :]
            shell = torch.arange(self.num_shells, device=receivers.device)
            pooled_index = receivers[:, None] * self.num_shells + shell[None, :]
        else:
            # One partition of unity per scale, concatenated coarse to fine.
            blocks = []
            for basis, count in zip(self.shell_bases, self.shell_counts):
                shell, coefficients = basis(edge_len)
                edge_shells = coefficients[:, :, None] * edge_features[:, None, :]
                pooled_index = receivers[:, None] * count + shell
                pooled = edge_features.new_zeros(
                    (num_nodes * count, edge_features.shape[-1])
                )
                pooled.index_add_(
                    0,
                    pooled_index.reshape(-1),
                    edge_shells.reshape(-1, edge_features.shape[-1]),
                )
                blocks.append(
                    pooled.reshape(num_nodes, count, edge_features.shape[-1])
                )
            return torch.cat(blocks, dim=1)
        pooled = edge_features.new_zeros(
            (num_nodes * self.num_shells, edge_features.shape[-1])
        )
        pooled.index_add_(
            0,
            pooled_index.reshape(-1),
            edge_shells.reshape(-1, edge_features.shape[-1]),
        )
        return pooled.reshape(num_nodes, self.num_shells, edge_features.shape[-1])

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features, edge_features, _ = super().forward(
            node_attrs,
            edge_index,
            edge_vec,
            edge_len,
            return_edge_features=True,
        )
        tokens = self.pool_edge_features(
            edge_features,
            edge_index[1],
            edge_len,
            node_attrs.shape[0],
        )
        return node_features, tokens, self.token_kind, self._current_token_coordinate()


class CanonicalACETokenizer(nn.Module):
    """Map an unordered local neighborhood to a smooth canonical ACE sequence.

    The sequence contains one permutation-invariant density token per radial
    channel followed by one token per ACE correlation degree. No neighbor is
    serialized and no coordinate-based sorting is used.
    """

    def __init__(
        self,
        r_max: float,
        l_max: int,
        num_radial: int,
        hidden_dim: int,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        radial_basis_type: str = "gaussian",
        radial_trainable: bool = False,
        gaussian_width: float = 0.7,
        remove_pair_self_contractions: bool = True,
    ):
        super().__init__()
        if not 2 <= correlation_order <= 6:
            raise ValueError("correlation_order must lie in [2, 6]")
        if l_max < 0:
            raise ValueError("l_max must be non-negative")
        self.r_max = float(r_max)
        self.l_max = int(l_max)
        self.num_radial = int(num_radial)
        self.hidden_dim = int(hidden_dim)
        self.correlation_order = int(correlation_order)
        self.remove_pair_self_contractions = bool(remove_pair_self_contractions)
        self.num_body_tokens = self.correlation_order - 1
        self.sequence_length = self.num_radial + self.num_body_tokens

        output_irreps = []
        correlation_irreps = []
        for ell in range(l_max + 1):
            output_mul = hidden_dim if ell == 0 else hidden_dim // (2 if ell == 1 else 4)
            corr_mul = correlation_channels if ell == 0 else correlation_channels // (2 if ell == 1 else 4)
            parity = (-1) ** ell
            output_irreps.append((max(1, output_mul), (ell, parity)))
            correlation_irreps.append((max(1, corr_mul), (ell, parity)))

        self.irreps_out = o3.Irreps(output_irreps)
        self.irreps_token = o3.Irreps(correlation_irreps)
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        self.irreps_species = o3.Irreps(f"{hidden_dim}x0e")
        self.cutoff = SmoothPolynomialCutoff(r_max)
        self.radial_basis = make_radial_basis(
            radial_basis_type,
            r_max,
            num_radial,
            radial_trainable,
            gaussian_width,
        )
        self.spherical_harmonics = o3.SphericalHarmonics(
            self.irreps_sh,
            normalize=True,
            normalization="component",
        )
        self.density_product = o3.FullyConnectedTensorProduct(
            self.irreps_species,
            self.irreps_sh,
            self.irreps_token,
            internal_weights=True,
            shared_weights=True,
        )
        self._token_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.irreps_token
        )
        self.shell_mix = nn.ParameterList(
            [
                nn.Parameter(
                    torch.eye(multiplicity)[None, :, :].repeat(num_radial, 1, 1)
                    / math.sqrt(num_radial)
                )
                for multiplicity, _ in self._token_blocks
            ]
        )

        self.contractions = nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    self.irreps_token,
                    self.irreps_token,
                    self.irreps_token,
                    internal_weights=True,
                    shared_weights=True,
                )
                for _ in range(correlation_order - 2)
            ]
        )
        self.order_to_node = nn.ModuleList(
            [o3.Linear(self.irreps_token, self.irreps_out) for _ in range(self.num_body_tokens)]
        )
        self.center_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

        token_kind = torch.cat(
            (
                torch.zeros(num_radial, dtype=torch.long),
                torch.arange(1, self.num_body_tokens + 1, dtype=torch.long),
            )
        )
        if radial_basis_type.lower() == "gaussian":
            radial_coordinate = torch.linspace(0.0, 1.0, num_radial + 2)[1:-1]
        else:
            radial_coordinate = torch.arange(1, num_radial + 1, dtype=torch.get_default_dtype())
            radial_coordinate = radial_coordinate / float(num_radial)
        body_coordinate = 1.0 + torch.arange(
            self.num_body_tokens, dtype=torch.get_default_dtype()
        ) / max(1, self.num_body_tokens)
        self.register_buffer("token_kind", token_kind)
        self.register_buffer("token_coordinate", torch.cat((radial_coordinate, body_coordinate)))

    def _radial_density_tokens(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender, receiver = edge_index[0], edge_index[1]
        n_nodes = node_attrs.shape[0]
        harmonics = self.spherical_harmonics(edge_vec)
        radial = self.radial_basis(edge_len)
        angular = self.density_product(node_attrs[sender], harmonics)
        shell_blocks = []
        offset = 0
        for (multiplicity, irrep_dim), shell_matrix in zip(
            self._token_blocks, self.shell_mix
        ):
            width = multiplicity * irrep_dim
            angular_block = angular[:, offset : offset + width].reshape(
                angular.shape[0], multiplicity, irrep_dim
            )
            mixed = torch.einsum("noi,eim->enom", shell_matrix, angular_block)
            shell_blocks.append(mixed.reshape(angular.shape[0], self.num_radial, width))
            offset += width
        edge_shells = radial[:, :, None] * torch.cat(shell_blocks, dim=-1)
        flat_density = edge_shells.new_zeros(
            (n_nodes * self.num_radial, self.irreps_token.dim)
        )
        shell_index = torch.arange(self.num_radial, device=receiver.device)
        flat_receiver = (receiver[:, None] * self.num_radial + shell_index[None, :]).reshape(-1)
        flat_density.index_add_(0, flat_receiver, edge_shells.reshape(-1, self.irreps_token.dim))
        radial_tokens = flat_density.reshape(n_nodes, self.num_radial, self.irreps_token.dim)
        return radial_tokens, edge_shells.sum(dim=1)

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        radial_tokens, edge_features = self._radial_density_tokens(
            node_attrs, edge_index, edge_vec, edge_len
        )
        density = radial_tokens.sum(dim=1)
        correlations = [density]
        correlation = density

        if self.correlation_order >= 3:
            correlation = self.contractions[0](density, density)
            if self.remove_pair_self_contractions and edge_features.shape[0] > 0:
                edge_self = self.contractions[0](edge_features, edge_features)
                self_sum = torch.zeros_like(correlation)
                self_sum.index_add_(0, edge_index[1], edge_self)
                correlation = correlation - self_sum
            correlations.append(correlation)

        contraction_start = 1 if self.correlation_order >= 3 else 0
        for contraction in self.contractions[contraction_start:]:
            correlation = contraction(correlation, density)
            correlations.append(correlation)

        non_scalar_dim = self.irreps_out.dim - self.hidden_dim
        center = torch.cat(
            (
                self.center_projection(node_attrs),
                node_attrs.new_zeros((node_attrs.shape[0], non_scalar_dim)),
            ),
            dim=-1,
        )
        node_features = center
        for projection, correlation_token in zip(self.order_to_node, correlations):
            node_features = node_features + projection(correlation_token)

        body_tokens = torch.stack(correlations, dim=1)
        tokens = torch.cat((radial_tokens, body_tokens), dim=1)
        return node_features, tokens, self.token_kind, self.token_coordinate
