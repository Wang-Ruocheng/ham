"""Bounded numerical preflight; no physical data, training, or saved model.

The parent admits an idle physical GPU before invoking this script. All RNG
changes are isolated. Native CUDA outputs are recorded as diagnostic evidence,
but only the corrected implementation determines whether preflight passes.
"""
import argparse
import copy
import time
import torch
from recurrent_rg.ensemble_vae import EnsembleVAEQ
from recurrent_rg.probability_normalization import stable_log_softmax, stable_softmax
from scripts.run_dual_ising_current import write_json


def verify(device):
    started = time.monotonic()
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA preflight requires an admitted CUDA device')
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(91625)
    cases = []
    for bins in (16, 256, 625, 626):
        x = torch.randn(16, bins, dtype=torch.double, generator=generator)
        cases.extend([(f'random_{bins}', x), (f'uniform_{bins}', torch.zeros_like(x))])
    base = torch.randn(8, 3, 625, dtype=torch.double, generator=generator)
    cases.extend([('three_dimensional', base), ('noncontiguous', base.transpose(0, 1)),
                  ('large_common_offset', base+1e7), ('peaked', base*1e4)])
    receipts = []
    for name, data in cases:
        cpu = data.detach().clone().requires_grad_()
        upstream = torch.randn(data.shape, dtype=torch.double, generator=generator)
        reference = cpu.log_softmax(-1)  # Independent CPU fused implementation.
        expected_grad, = torch.autograd.grad((reference*upstream).sum(), cpu)
        x = data.to(device).detach().requires_grad_()
        output = stable_log_softmax(x)
        grad, = torch.autograd.grad((output*upstream.to(device)).sum(), x)
        probabilities = stable_softmax(x.detach())
        torch.testing.assert_close(output.cpu(), reference.detach(), atol=2e-10, rtol=2e-12)
        torch.testing.assert_close(grad.cpu(), expected_grad, atol=2e-10, rtol=2e-10)
        error = float((probabilities.sum(-1)-1).abs().max())
        if not torch.isfinite(output).all() or not torch.isfinite(grad).all() or error > 2e-12:
            raise AssertionError('Normalization/finiteness failed: '+name)
        shift_error = float((stable_log_softmax(x.detach()+32)-output.detach()).abs().max())
        if shift_error > 1e-8:
            raise AssertionError('Common-logit-shift invariance failed: '+name)
        receipts.append(dict(case=name, probability_sum_max_error=error,
            log_probability_CPU_max_error=float((output.cpu()-reference.detach()).abs().max()),
            gradient_CPU_max_error=float((grad.cpu()-expected_grad).abs().max()),
            common_shift_max_error=shift_error))

    # Central finite differences exercise the whole 625-bin normalizer with
    # nonconstant upstream derivatives, including both row alignments.
    x = torch.randn(2, 625, generator=generator, dtype=torch.double).to(device).requires_grad_()
    direction = torch.randn(2, 625, generator=generator, dtype=torch.double).to(device)
    direction /= direction.norm()
    target = torch.rand(2, 625, generator=generator, dtype=torch.double).to(device)
    target /= target.sum(-1, keepdim=True)
    loss = -(target*stable_log_softmax(x)).sum()
    gradient, = torch.autograd.grad(loss, x)
    delta = 1e-4
    difference = (-(target*stable_log_softmax(x.detach()+delta*direction)).sum()
                  +(target*stable_log_softmax(x.detach()-delta*direction)).sum())/(2*delta)
    analytic = (gradient*direction).sum()
    torch.testing.assert_close(difference, analytic, atol=1e-9, rtol=1e-6)
    if float(gradient.sum(-1).abs().max()) > 1e-12:
        raise AssertionError('CE gradient violates common-shift invariance')

    # Full weak-KL Q loss: identical weights, inputs and reparameterization
    # noise must produce matching CPU/device gradients, without any update.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        cpu_model = EnsembleVAEQ(states=5)
    hist = torch.rand(16, 625, generator=generator, dtype=torch.double)
    hist /= hist.sum(-1, keepdim=True)
    cpu_model.calibrate(hist); cpu_model.set_objective_step(600)
    model = copy.deepcopy(cpu_model).to(device)
    epsilon = torch.randn(16, 2, generator=generator, dtype=torch.double)
    reference_loss, _ = cpu_model.cross_subset_objective(hist[:8], hist[8:], epsilon=epsilon)
    loss, terms = model.cross_subset_objective(hist[:8].to(device), hist[8:].to(device), epsilon=epsilon.to(device))
    expected = torch.autograd.grad(reference_loss, tuple(cpu_model.parameters()))
    actual = torch.autograd.grad(loss, tuple(model.parameters()))
    torch.testing.assert_close(loss.cpu(), reference_loss, atol=1e-10, rtol=1e-10)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, atol=1e-9, rtol=1e-8)

    # Binary Ising uses 16 oriented motifs and the same weak-KL objective.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        binary_cpu = EnsembleVAEQ(states=2)
    binary_hist = torch.rand(16, 16, generator=generator, dtype=torch.double)
    binary_hist /= binary_hist.sum(-1, keepdim=True)
    binary_cpu.calibrate(binary_hist); binary_cpu.set_objective_step(600)
    binary = copy.deepcopy(binary_cpu).to(device)
    binary_ref, _ = binary_cpu.cross_subset_objective(binary_hist[:8], binary_hist[8:], epsilon=epsilon)
    binary_loss, _ = binary.cross_subset_objective(binary_hist[:8].to(device), binary_hist[8:].to(device), epsilon=epsilon.to(device))
    binary_expected = torch.autograd.grad(binary_ref, tuple(binary_cpu.parameters()))
    binary_actual = torch.autograd.grad(binary_loss, tuple(binary.parameters()))
    torch.testing.assert_close(binary_loss.cpu(), binary_ref, atol=1e-10, rtol=1e-10)
    for a, b in zip(binary_actual, binary_expected):
        torch.testing.assert_close(a.cpu(), b, atol=1e-9, rtol=1e-8)

    # This records the original backend defect without making it a requirement:
    # a fixed PyTorch version is also allowed to pass the preflight.
    native_input = torch.zeros(4, 625, dtype=torch.double, device=device, requires_grad=True)
    native_output = native_input.log_softmax(-1)
    native_gradient, = torch.autograd.grad(-native_output.mean(-1).sum(), native_input)
    receipt = dict(status='passed', device=str(device), torch=torch.__version__,
        torch_git=torch.version.git_version, cases=receipts,
        directional_finite_difference_error=float((difference-analytic).abs()),
        cross_subset_objective_CPU_error=float((loss.cpu()-reference_loss).abs()),
        parameter_gradient_CPU_max_error=max(float((a.cpu()-b).abs().max()) for a, b in zip(actual, expected)),
        binary_objective_CPU_error=float((binary_loss.cpu()-binary_ref).abs()),
        binary_parameter_gradient_CPU_max_error=max(float((a.cpu()-b).abs().max()) for a, b in zip(binary_actual, binary_expected)),
        original_fused_probability_row_sums=native_output.detach().exp().sum(-1).cpu().tolist(),
        original_fused_CE_gradient_row_sums=native_gradient.sum(-1).cpu().tolist(),
        model_updates=0, physical_data_loaded=False, elapsed_seconds=time.monotonic()-started)
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        receipt.update(physical_GPU_uuid=str(properties.uuid), GPU_name=properties.name,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = verify(args.device)
    except Exception as error:
        write_json(args.output, dict(status='failed', error=str(error)))
        raise
    write_json(args.output, result)
    print('VAE normalization and gradient preflight passed', flush=True)


if __name__ == '__main__':
    main()
