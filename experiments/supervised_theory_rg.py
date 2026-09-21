#!/usr/bin/env python3
"""Directly supervise theory_rg to learn exact 1D Ising RG flow."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recurrent_rg.ising import DTYPE, exact_decimation_flow
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def train(model, j_values, epochs=1000, lr=1e-3, batch_size=32):
    optimizer = torch.optim.Adam(model.theory_rg.parameters(), lr=lr)
    j_next = exact_decimation_flow(j_values, levels=2)[:, 1]
    n = len(j_values)
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            j_batch = j_values[idx].unsqueeze(-1)
            j_target = j_next[idx].unsqueeze(-1)

            j_pred = model.theory_rg(j_batch)
            loss = nn.functional.mse_loss(j_pred, j_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            nb += 1

        avg = total_loss / nb
        history.append(avg)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.6e}")

    return history


def evaluate(model, j_test):
    model.eval()
    with torch.no_grad():
        j_in = j_test.unsqueeze(-1)
        j_pred = model.theory_rg(j_in).squeeze(-1)
        j_exact = exact_decimation_flow(j_test, levels=2)[:, 1]
        r_minus_j = j_pred - j_test

        # Find fixed points
        sign_changes = np.where(np.diff(np.sign(r_minus_j.numpy())))[0]
        fps = []
        for idx in sign_changes:
            j0, j1 = j_test[idx].item(), j_test[idx+1].item()
            r0, r1 = r_minus_j[idx].item(), r_minus_j[idx+1].item()
            fps.append(j0 - r0 * (j1 - j0) / (r1 - r0))

        return {
            'j_test': j_test.numpy(),
            'j_pred': j_pred.numpy(),
            'j_exact': j_exact.numpy(),
            'r_minus_j': r_minus_j.numpy(),
            'fixed_points': np.array(fps),
            'mse': nn.functional.mse_loss(j_pred, j_exact).item(),
        }


def plot(results, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    j, pred, exact = results['j_test'], results['j_pred'], results['j_exact']
    rmj, fps = results['r_minus_j'], results['fixed_points']

    ax = axes[0, 0]
    ax.plot(j, pred, 'b-', label='Learned R(J)', lw=2)
    ax.plot(j, exact, 'r--', label='Exact', lw=2)
    ax.plot(j, j, 'k:', alpha=0.5)
    ax.set(xlabel='J', ylabel='R(J)', title='RG Flow')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(j, rmj, 'g-', lw=2)
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    for fp in fps:
        ax.axvline(fp, color='r', ls='--', alpha=0.7, label=f'J*={fp:.3f}')
    ax.set(xlabel='J', ylabel='R(J)-J', title='Fixed Points')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    err = pred - exact
    ax.plot(j, err, 'b-', lw=2)
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.set(xlabel='J', ylabel='Error', title=f'Error (MSE={results["mse"]:.2e})')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    rel_err = np.abs(err) / (np.abs(exact) + 1e-10)
    ax.semilogy(j, rel_err, 'b-', lw=2)
    ax.set(xlabel='J', ylabel='Relative Error', title='Relative Error')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved to {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--j-min', type=float, default=0.01)
    p.add_argument('--j-max', type=float, default=2.0)
    p.add_argument('--n-train', type=int, default=1000)
    p.add_argument('--n-test', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=Path('results/supervised_theory_rg'))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Init model (only use theory_rg)
    model = RecurrentRGNetwork1D(levels=2, theory_dim=1, hidden=args.hidden).to(dtype=DTYPE)
    print(f'theory_rg parameters: {sum(p.numel() for p in model.theory_rg.parameters())}')

    # Train
    j_train = torch.linspace(args.j_min, args.j_max, args.n_train, dtype=DTYPE)
    history = train(model, j_train, args.epochs, args.lr, args.batch_size)

    # Evaluate
    j_test = torch.linspace(args.j_min, args.j_max, args.n_test, dtype=DTYPE)
    results = evaluate(model, j_test)

    print(f'\nTest MSE: {results["mse"]:.6e}')
    print(f'Fixed points: {results["fixed_points"]}')

    # Plot
    plot(results, args.output_dir / 'results.png')

    # Save
    np.savez(args.output_dir / 'results.npz', **results)
    torch.save(model.theory_rg.state_dict(), args.output_dir / 'model.pt')
    print(f'Saved to {args.output_dir}')


if __name__ == '__main__':
    main()
