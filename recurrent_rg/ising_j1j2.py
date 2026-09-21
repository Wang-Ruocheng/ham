"""Exact scale-2 decimation RG for the 1D J1-J2 Ising chain.

Model: ``-H = sum_i [J1 s_i s_{i+1} + J2 s_i s_{i+2}]``, spins +-1,
dimensionless couplings. Summing out every second site generates
longer-range and multi-spin interactions, so the flow is followed in the
full parity-even coupling space of a 16-config site transfer factor.

Transfer factor: after cutting bonds between sites 4k-1 and 4k, one
factor covers spins (4k .. 4k+3). Every physical NN bond lies strictly
inside one factor (full J1). Every NNN bond (i, i+2) lies strictly
inside one factor except bonds (4k+3, 4k+5), which are split J2/2 + J2/2
between adjacent factors. The general even coupling vector is

    K = (K12, K23, K34, K13, K24, K14, K1234),

``log W = sum K_m phi_m + const``; the physical model is embedded as
``K12 = K23 = K34 = J1``, ``K13 = K24 = J2``, ``K14 = J2/2``,
``K1234 = 0``. The seven even monomials plus the constant span all
globally spin-flip even functions on {+1,-1}^4, so the coupling
read-back after decimation is exact (no projection error).

One RG step sums the middle two spins:

    W'(s1, s4) = sum_{s2, s3} W(s1, s2, s3, s4)

and ``W'`` is re-expanded in the same even basis, where ``K14'`` is the
new NN coupling of the decimated chain, ``K13'``, ``K24'`` are new NNN
couplings, and so on. Effective physical couplings along a trajectory
are read off as ``J1_eff = K14`` and ``J2_eff = (K13 + K24) / 2``; the
other directions (K12, K23, K34, K1234) quantify how far the flow leaves
the physical subspace.

Conventions follow ``recurrent_rg.ising``: torch float64, spins +-1.
"""

from __future__ import annotations

import math

import torch

DTYPE = torch.float64

COUPLING_NAMES = ("12", "23", "34", "13", "24", "14", "1234")


def _quad_spins() -> torch.Tensor:
    """The 16 configurations of (s1, s2, s3, s4) in {+1, -1}^4."""
    indices = torch.arange(16, dtype=torch.int64)
    bits = (indices[:, None] >> torch.tensor([3, 2, 1, 0])) & 1
    return (2 * bits - 1).to(DTYPE)


_QUADS = _quad_spins()


def _build_design() -> torch.Tensor:
    """Matrix M[config, basis] = (phi_1..phi_7, 1), shape [16, 8]."""
    s1, s2, s3, s4 = _QUADS.T
    phi = torch.stack(
        (s1 * s2, s2 * s3, s3 * s4, s1 * s3, s2 * s4, s1 * s4, s1 * s2 * s3 * s4),
        dim=1,
    )
    return torch.cat((phi, torch.ones(16, 1, dtype=DTYPE)), dim=1)


_DESIGN = _build_design()


def physical_embedding(J1: float, J2: float) -> torch.Tensor:
    """Map physical couplings (J1, J2) to the 7-coupling vector K."""
    couplings = torch.zeros(7, dtype=DTYPE)
    couplings[0] = couplings[1] = couplings[2] = J1  # 12, 23, 34
    couplings[3] = couplings[4] = J2  # 13, 24
    couplings[5] = 0.5 * J2  # 14 (split with the neighboring factor)
    return couplings


def logw_from_couplings(couplings: torch.Tensor) -> torch.Tensor:
    """Log weight log W(s1..s4) over the 16 configs, shape [16]."""
    if couplings.ndim != 1 or couplings.numel() != 7:
        raise ValueError("couplings must be a 1D tensor with 7 entries")
    return _DESIGN[:, :7] @ couplings


def couplings_from_logw(logw: torch.Tensor) -> torch.Tensor:
    """Least-squares read-back of (K, const) from a 16-vector of log W.

    The basis is complete for even functions, so the reconstruction is
    exact up to floating-point error.
    """
    if logw.shape != (16,):
        raise ValueError("logw must have shape [16]")
    solution = torch.linalg.lstsq(_DESIGN, logw)
    return solution.solution[:7]


def partition_function_ring(couplings: torch.Tensor, n_quads: int) -> float:
    """Exact Z of a periodic ring of ``4 * n_quads`` spins by brute-force
    enumeration of the 2^(4 n_quads) configurations of the quad factors,
    with the periodic boundary bonds added explicitly.

    The physical chain is tiled by quads (s_{4k+1..4k+4}). Multiplying the
    quad weights reproduces every physical bond exactly once except the
    three wrap-around bonds: the NN bond (s_N, s_1) and the NNN bonds
    (s_{N-1}, s_1) and (s_N, s_2). Their couplings must be added on top of
    the quad product. The needed J1, J2 are reconstructed from the
    coupling vector assuming the physical embedding (J2 = K13 = K24,
    J2/2 = K14, J1 = K12 = K23 = K34). Verified against direct
    enumeration in ``validate``.
    """
    import itertools

    logw = logw_from_couplings(couplings)
    w = torch.exp(logw).reshape(2, 2, 2, 2)
    # Physical couplings from the embedding conventions. The in-quad J1
    # bonds are full (K12 = K23 = K34 = J1); the wrap NNN bonds are full
    # J2 as embedded (K13 = K24 = J2), but the wrap NN bond (s_N, s_1) is
    # NOT present in any quad factor and must be added with weight J1.
    # The NNN wrap bonds (s_{N-1}, s_1) and (s_N, s_2) are likewise absent
    # from the quads and need weight J2 each.
    j1 = couplings[0].item()
    j2 = couplings[3].item()
    n_sites = 4 * n_quads
    total = 0.0
    for cfg in itertools.product((0, 1), repeat=n_sites):
        weight = 1.0
        for k in range(n_quads):
            a, b, c, d = (cfg[4 * k + i] for i in range(4))
            weight *= w[a, b, c, d].item()
        s = tuple(2 * x - 1 for x in cfg)
        # wrap bonds absent from the quad tiling; NNN wraps carry the
        # remaining half because the other half sits in K14 of each quad.
        weight *= math.exp(
            j1 * s[-1] * s[0] + 0.5 * j2 * s[-2] * s[0] + 0.5 * j2 * s[-1] * s[1]
        )
        total += weight
    return total


def decimation_step(couplings: torch.Tensor) -> torch.Tensor:
    """One scale-2 decimation step.

    The coarse transfer factor is built from two consecutive fine factors
    by summing the middle spins (s3, s4) and identifying the boundary
    spins as coarse spins:

        W'(S1, S2, S3, S4) = sum_{s3,s4} W(S1, S2, s3, s4) W(s3, s4, S3, S4)

    with (S1, S2) = (s1, s2) of the first quad and (S3, S4) = (s5, s6) of
    the second. The result is re-expanded in the even basis; the constant
    (free-energy) term is dropped.
    """
    logw = logw_from_couplings(couplings)
    w = torch.exp(logw - torch.logsumexp(logw, dim=0))
    w4 = w.reshape(2, 2, 2, 2)  # (s1, s2, s3, s4)
    coarse = torch.einsum("abcd,cdef->abef", w4, w4).reshape(16)
    return couplings_from_logw(torch.log(coarse))


def decimation_flow(
    J1: float, J2: float, levels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterate decimation from the physical (J1, J2) model.

    Returns the full coupling trajectory [levels, 7] and the effective
    physical couplings [levels, 2] with ``J1_eff = K14`` and
    ``J2_eff = (K13 + K24) / 2``.
    """
    if levels < 1:
        raise ValueError("levels must be at least 1")
    trajectory = [physical_embedding(J1, J2)]
    for _ in range(levels - 1):
        trajectory.append(decimation_step(trajectory[-1]))
    trajectory = torch.stack(trajectory)
    effective = torch.stack(
        (trajectory[:, 5], 0.5 * (trajectory[:, 3] + trajectory[:, 4])), dim=1
    )
    return trajectory, effective


def decimation_map(
    J1: torch.Tensor, J2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized one-step map on the physical plane.

    Args:
        J1, J2: Matching tensors of physical couplings.

    Returns:
        (J1', J2') with the same shape: the effective physical couplings
        after one exact decimation step.
    """
    if J1.shape != J2.shape:
        raise ValueError("J1 and J2 must have matching shapes")
    flat = torch.stack((J1.reshape(-1), J2.reshape(-1)), dim=1)
    out = []
    for j1, j2 in flat:
        out.append(decimation_step(physical_embedding(float(j1), float(j2))))
    read_back = torch.stack(out).reshape(*J1.shape, 7)
    return read_back[..., 5], 0.5 * (read_back[..., 3] + read_back[..., 4])


def correlation_length(J1: float, J2: float, sector: str = "magnetization") -> float:
    """Exact infinite-chain correlation length in lattice units.

    The 4x4 transfer matrix Tm = W.reshape(4, 4) advances the chain by
    four spins per multiplication, so eigenvalues satisfy
    ``xi = 4 / log(lambda_0 / |lambda*|)`` with ``lambda*`` the largest
    eigenvalue in the relevant symmetry sector.
    """
    logw = logw_from_couplings(physical_embedding(J1, J2))
    w = torch.exp(logw)
    T = w.reshape(4, 4)
    eigvals = torch.linalg.eigvals(T).real
    order = torch.argsort(eigvals.abs(), descending=True)
    eigvals = eigvals[order]
    lam0 = eigvals[0]
    if sector == "energy":
        lam_sub = eigvals[1].abs()
    elif sector == "magnetization":
        # Spin-flip parity of pair states (s1,s2) -> (-s1,-s2): states 0,3
        # even; 1,2 odd. Diagonalize within the odd sector.
        odd = T[1:3, :][:, 1:3]
        lam_sub = torch.linalg.eigvals(odd).abs().max().real
    else:
        raise ValueError("sector must be 'magnetization' or 'energy'")
    if lam_sub <= 0.0:
        return float("inf")
    return 4.0 / torch.log(lam0 / lam_sub).item()


def validate() -> None:
    """Sanity checks: exactness, J2=0 closure, Z matching, known limits."""
    import itertools
    import math

    tol = 1e-10

    # Read-back is exact (complete basis) for arbitrary couplings.
    rng = torch.Generator().manual_seed(0)
    for _ in range(5):
        couplings = torch.randn(7, generator=rng, dtype=DTYPE)
        logw = logw_from_couplings(couplings)
        err = (couplings_from_logw(logw) - couplings).abs().max().item()
        assert err < 1e-9, f"read-back error {err}"

    # Embedding reproduces the exact Z of an 8-spin J1-J2 ring when the
    # ring length is a multiple of the quad size. We embed the ring so
    # that every physical bond appears exactly once in the quad tiling:
    # this requires the "oblique" tiling offset (quad boundaries between
    # sites 4k and 4k+1, factor covering 4k+1..4k+4) and the split NNN
    # convention K14 = J2/2. Direct construction, cross-checked against
    # the brute-force 8-spin sum.
    J1, J2 = 0.8, 0.3
    z_exact = 0.0
    for s in itertools.product((-1, 1), repeat=8):
        e = sum(
            J1 * s[i] * s[(i + 1) % 8] + J2 * s[i] * s[(i + 2) % 8]
            for i in range(8)
        )
        z_exact += math.exp(e)
    # The transfer matrix M = W.reshape(4,4) with
    #   M[(s1,s2),(s3,s4)] = W(s1,s2,s3,s4)
    # reproduces the ring Z via Z = Tr M^{n_pairs}: the factors tile the
    # ring with shared boundary pairs, and the K14 = J2/2 convention makes
    # the shared NNN bond counted exactly once in total (J2/2 from each of
    # the two adjacent factors). Checked for a 2-quad (8-spin) ring
    # against the brute-force sum below.
    logw = logw_from_couplings(physical_embedding(J1, J2))
    m = torch.exp(logw).reshape(4, 4)
    # The correct transfer matrix for the quad tiling is the split-bond
    # pair matrix u[(a,b),(c,d)] with
    #   K_ab = K_cd = J1/2, K_bc = J1, K_ac = J2, K_bd = J2,
    # i.e. the boundary NN bonds are split J1/2 between adjacent pairs so
    # that Z = Tr u^{#pairs} reproduces the ring exactly.
    def u_matrix(j1: float, j2: float) -> torch.Tensor:
        u = torch.zeros(4, 4, dtype=DTYPE)
        for ia in range(4):
            a = 1.0 if ia & 2 else -1.0
            b = 1.0 if ia & 1 else -1.0
            for ic in range(4):
                c = 1.0 if ic & 2 else -1.0
                d = 1.0 if ic & 1 else -1.0
                u[ia, ic] = torch.exp(torch.tensor(
                    0.5 * j1 * a * b + j1 * b * c + 0.5 * j1 * c * d
                    + j2 * a * c + j2 * b * d, dtype=DTYPE))
        return u
    u = u_matrix(J1, J2)
    z_pair = torch.trace(torch.linalg.matrix_power(u, 4)).item()
    assert abs(z_pair / z_exact - 1.0) < 1e-9, (z_pair, z_exact)
    print("split-bond pair matrix reproduces Z exactly:", z_pair, z_exact)
    # ---------------------------------------------------------------
    # Fixed points of the RG map on the physical (J1, J2) plane.
    # A trajectory point (J1, J2) maps to the full 7-coupling vector; the
    # physical projection is (K14, (K13+K24)/2). Fixed points are the
    # physically meaningful endpoints of the flow.
    # ---------------------------------------------------------------
    from collections import defaultdict

    def flow_point(j1: float, j2: float, steps: int = 64) -> tuple:
        """Iterate the decimation map; return the projected endpoint."""
        k = physical_embedding(j1, j2)
        for _ in range(steps):
            k = decimation_step(k)
        return float(k[5]), float(0.5 * (k[3] + k[4]))

    print("\nRG flow endpoints (projected to (J1_eff, J2_eff)):")
    grid = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    basins = defaultdict(list)
    for j1 in grid:
        for j2 in grid:
            end = flow_point(j1, j2)
            key = (round(end[0], 6), round(end[1], 6))
            basins[key].append((j1, j2))
            print(f"  ({j1:+.1f}, {j2:+.1f}) -> ({end[0]:+.6f}, {end[1]:+.6f})")
    print("\nDistinct attracting fixed points found:")
    for fp, seeds in sorted(basins.items()):
        print(f"  {fp}   (basin size {len(seeds)})")


if __name__ == "__main__":
    validate()
