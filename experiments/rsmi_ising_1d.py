#!/usr/bin/env python3
"""Exact small-system reproduction of RSMI for the clean 1D Ising chain.

The visible block V=(v1,v2) is compressed to one hidden Ising spin h with

    P(h|V) = sigmoid(2 h (lambda_1 v1 + lambda_2 v2)).

For a short periodic chain we enumerate the Boltzmann distribution exactly,
marginalize it to P(V,E), and optimize the exact mutual information I(H:E).
This removes mutual-information-estimator and sampling errors, exposing the
RG-kernel optimization itself.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recurrent_rg.ising import sample_periodic_ensembles


DTYPE = torch.float64
VISIBLE_STATES = torch.tensor(
    [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
    dtype=DTYPE,
)


@dataclass
class RunResult:
    seed: int
    weights: list[float]
    mutual_information_bits: float
    selected_visible: int
    dominance_ratio: float


def enumerate_joint(chain_length: int, coupling: float, buffer: int) -> torch.Tensor:
    """Return exact P(V,E), with V=(s0,s1) and buffers on both sides."""
    if chain_length < 2 * buffer + 4:
        raise ValueError("chain_length must leave at least two environment spins")
    if chain_length > 18:
        raise ValueError("exact enumeration is intentionally limited to N <= 18")

    states = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=chain_length)), dtype=DTYPE
    )
    log_weight = coupling * (states * torch.roll(states, -1, dims=1)).sum(dim=1)
    probability = torch.softmax(log_weight, dim=0)

    visible = states[:, :2]
    # On the periodic chain, sites immediately outside V are the buffer.
    environment_indices = list(range(2 + buffer, chain_length - buffer))
    environment = states[:, environment_indices]

    visible_index = ((visible > 0).long() * torch.tensor([1, 2])).sum(dim=1)
    powers = 2 ** torch.arange(len(environment_indices), dtype=torch.long)
    environment_index = ((environment > 0).long() * powers).sum(dim=1)

    joint = torch.zeros(4, 2 ** len(environment_indices), dtype=DTYPE)
    joint.index_put_((visible_index, environment_index), probability, accumulate=True)
    return joint


def empirical_joint(
    chain_length: int,
    coupling: float,
    buffer: int,
    samples: int,
    seed: int,
) -> torch.Tensor:
    """Estimate P(V,E) from configurations, without using probabilities in the loss."""
    configurations, _ = sample_periodic_ensembles(
        np.asarray([coupling]),
        replicas=1,
        samples=samples,
        sites=chain_length,
        seed=seed,
    )
    states = configurations[0, :, :, 0]
    visible = states[:, :2]
    environment_indices = list(range(2 + buffer, chain_length - buffer))
    environment = states[:, environment_indices]

    visible_index = ((visible > 0).long() * torch.tensor([1, 2])).sum(dim=1)
    powers = 2 ** torch.arange(len(environment_indices), dtype=torch.long)
    environment_index = ((environment > 0).long() * powers).sum(dim=1)
    flat_index = visible_index * (2 ** len(environment_indices)) + environment_index
    counts = torch.bincount(
        flat_index, minlength=4 * (2 ** len(environment_indices))
    ).to(DTYPE)
    return counts.reshape(4, -1) / counts.sum()


def mutual_information_bits(weights: torch.Tensor, joint_ve: torch.Tensor) -> torch.Tensor:
    """Compute exact I(H:E) for the binary stochastic coarse-graining map."""
    p_h_plus_given_v = torch.sigmoid(2.0 * (VISIBLE_STATES @ weights))
    p_h_given_v = torch.stack((1.0 - p_h_plus_given_v, p_h_plus_given_v), dim=1)
    joint_he = torch.einsum("ve,vh->he", joint_ve, p_h_given_v)
    p_h = joint_he.sum(dim=1, keepdim=True)
    p_e = joint_he.sum(dim=0, keepdim=True)
    eps = torch.finfo(DTYPE).tiny
    mi_nats = (
        joint_he
        * (
            torch.log(joint_he + eps)
            - torch.log(p_h + eps)
            - torch.log(p_e + eps)
        )
    ).sum()
    return mi_nats / math.log(2.0)


def optimize_kernel(
    joint_ve: torch.Tensor, seed: int, steps: int, learning_rate: float
) -> RunResult:
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(0.2 * torch.randn(2, dtype=DTYPE))
    optimizer = torch.optim.Adam([weights], lr=learning_rate)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = -mutual_information_bits(weights, joint_ve)
        loss.backward()
        optimizer.step()

    final = weights.detach()
    absolute = final.abs()
    selected = int(torch.argmax(absolute).item())
    dominance = float((absolute.max() / absolute.min().clamp_min(1e-12)).item())
    return RunResult(
        seed=seed,
        weights=[float(x) for x in final],
        mutual_information_bits=float(mutual_information_bits(final, joint_ve)),
        selected_visible=selected + 1,
        dominance_ratio=dominance,
    )


def deterministic_candidate_mi(joint_ve: torch.Tensor, rule: str) -> float:
    strength = 14.0
    candidates = {
        "decimate_left": torch.tensor([strength, 0.0], dtype=DTYPE),
        "decimate_right": torch.tensor([0.0, strength], dtype=DTYPE),
        "majority": torch.tensor([strength, strength], dtype=DTYPE),
        "difference": torch.tensor([strength, -strength], dtype=DTYPE),
    }
    return float(mutual_information_bits(candidates[rule], joint_ve))


def aligned_truth_table(result: RunResult) -> dict[str, float]:
    """Align the arbitrary sign of h and order the selected spin first."""
    weights = torch.tensor(result.weights, dtype=DTYPE)
    selected = result.selected_visible - 1
    if weights[selected] < 0:
        weights = -weights
    if selected == 1:
        weights = weights.flip(0)
    probabilities = torch.sigmoid(2.0 * (VISIBLE_STATES @ weights))
    labels = ["(-1,-1)", "(+1,-1)", "(-1,+1)", "(+1,+1)"]
    return {label: float(value) for label, value in zip(labels, probabilities)}


def make_figure(
    output: Path,
    joint_ve: torch.Tensor,
    results: list[RunResult],
    representative: RunResult,
    chain_length: int,
    coupling: float,
    buffer: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    # Fixed-norm angular slice through the RSMI objective.
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    radius = 8.0
    angular_mi = []
    for angle in theta:
        w = torch.tensor([radius * np.cos(angle), radius * np.sin(angle)], dtype=DTYPE)
        angular_mi.append(float(mutual_information_bits(w, joint_ve)))
    ax = axes[0, 0]
    ax.plot(theta / np.pi, angular_mi, color="#275DAD", lw=2)
    for location in (0.0, 0.5, 1.0, 1.5, 2.0):
        ax.axvline(location, color="0.82", lw=0.8, zorder=0)
    ax.set(xlabel=r"kernel angle $\theta/\pi$", ylabel=r"$I(H:E)$ [bits]",
           title="Exact RSMI objective at fixed weight norm")

    # Learned weights from independent random initializations.
    ax = axes[0, 1]
    weights = np.asarray([item.weights for item in results])
    ax.scatter(weights[:, 0], weights[:, 1], c=[r.selected_visible for r in results],
               cmap="coolwarm", s=42, edgecolor="black", linewidth=0.4)
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for item, point in zip(results, weights):
        selected = item.selected_visible - 1
        sign = 1 if point[selected] >= 0 else -1
        groups.setdefault((selected, sign), []).append(point)
    for points in groups.values():
        center = np.mean(points, axis=0)
        ax.annotate(
            f"n={len(points)}", center, xytext=(7, 5), textcoords="offset points",
            fontsize=9,
        )
    lim = max(5.0, float(np.abs(weights).max()) * 1.12)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r"$\lambda_1$",
           ylabel=r"$\lambda_2$", title="Independent training runs")
    ax.set_aspect("equal")

    # Candidate-kernel comparison over coupling strength.
    ax = axes[1, 0]
    coupling_grid = np.linspace(0.15, 1.25, 32)
    for rule, label, color in (
        ("decimate_left", "decimation", "#D1495B"),
        ("majority", "majority rule", "#00798C"),
        ("difference", "difference", "#EDAE49"),
    ):
        values = []
        for current_coupling in coupling_grid:
            current_joint = enumerate_joint(chain_length, float(current_coupling), buffer)
            values.append(deterministic_candidate_mi(current_joint, rule))
        ax.plot(coupling_grid, values, label=label, color=color, lw=2)
    ax.axvline(coupling, color="0.35", ls="--", lw=1)
    ax.text(
        coupling + 0.018,
        deterministic_candidate_mi(joint_ve, "decimate_left"),
        rf"$K={coupling:g}$",
        fontsize=9,
        va="bottom",
    )
    ax.set(xlabel=r"Ising coupling $K$", ylabel=r"$I(H:E)$ [bits]",
           title="Which one-bit kernel retains more long-range information?")
    ax.legend(frameon=False)

    # Learned stochastic truth table after resolving h -> -h and left/right degeneracy.
    ax = axes[1, 1]
    table = aligned_truth_table(representative)
    ax.bar(range(4), list(table.values()), color="#4C956C")
    ax.set_xticks(range(4), table.keys())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$(v_{\rm selected},v_{\rm other})$")
    ax.set_ylabel(r"$P(h=+1\mid V)$")
    ax.set_title("Learned kernel (sign and left/right aligned)")

    fig.suptitle(
        rf"Exact RSMI for a periodic 1D Ising chain: $N={chain_length}$, "
        rf"$K={coupling:g}$, buffer={buffer}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-length", type=int, default=12)
    parser.add_argument("--coupling", type=float, default=0.8)
    parser.add_argument("--buffer", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--joint-source",
        choices=("exact", "empirical"),
        default="exact",
        help="use exact enumeration or a joint distribution estimated only from samples",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="number of configurations used when --joint-source=empirical",
    )
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/rsmi_ising_1d")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joint = (
        enumerate_joint(args.chain_length, args.coupling, args.buffer)
        if args.joint_source == "exact"
        else empirical_joint(
            args.chain_length,
            args.coupling,
            args.buffer,
            args.samples,
            args.sample_seed,
        )
    )
    results = [
        optimize_kernel(joint, seed, args.steps, args.learning_rate)
        for seed in range(args.seeds)
    ]
    representative = max(results, key=lambda item: item.mutual_information_bits)

    candidates = {
        rule: deterministic_candidate_mi(joint, rule)
        for rule in ("decimate_left", "decimate_right", "majority", "difference")
    }
    summary = {
        "setup": {
            "chain_length": args.chain_length,
            "coupling": args.coupling,
            "buffer_per_side": args.buffer,
            "visible_block": [0, 1],
            "hidden_states": 2,
            "environment_sites": args.chain_length - 2 - 2 * args.buffer,
            "objective": "I(H:E) in bits evaluated on the selected joint distribution",
            "joint_source": args.joint_source,
            "configuration_samples": (
                args.samples if args.joint_source == "empirical" else None
            ),
        },
        "candidate_mutual_information_bits": candidates,
        "runs": [asdict(item) for item in results],
        "representative_aligned_truth_table": aligned_truth_table(representative),
        "counts": {
            "selected_v1": sum(item.selected_visible == 1 for item in results),
            "selected_v2": sum(item.selected_visible == 2 for item in results),
        },
    }
    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    figure_path = args.output_dir / "rsmi_ising_1d.png"
    make_figure(
        figure_path,
        joint,
        results,
        representative,
        args.chain_length,
        args.coupling,
        args.buffer,
    )

    print(f"decimation MI: {candidates['decimate_left']:.9f} bits")
    print(f"majority MI:   {candidates['majority']:.9f} bits")
    print(
        "learned MI:    "
        f"{np.mean([r.mutual_information_bits for r in results]):.9f} "
        f"+/- {np.std([r.mutual_information_bits for r in results]):.2e} bits"
    )
    print(f"selected v1/v2: {summary['counts']['selected_v1']}/{summary['counts']['selected_v2']}")
    print(f"representative weights: {representative.weights}")
    print(f"saved {json_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
