"""CPU-only tests for the GPU2-only synthetic execution verifier."""
import pytest
import torch

from scripts import verify_low_ram_cuda as verify


def snapshot(at=0., **changes):
    result = dict(index=2, uuid=verify.EXPECTED_UUID, free_mib=24200,
                  compute_pids=[], observed_monotonic=at)
    return result | changes


@pytest.mark.parametrize('change', [dict(index=0), dict(uuid='other'),
                                  dict(free_mib=12287), dict(compute_pids=[123])])
def test_gate_refuses_unsafe_gpu(change):
    with pytest.raises(RuntimeError):
        verify.validate_snapshot(snapshot(**change))


def test_two_safe_observations_ten_seconds_apart(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    readings = iter([snapshot(), snapshot(10.)])
    sleeps = []
    assert len(verify.acquire_gpu_gate(snapshot=lambda: next(readings), sleep=sleeps.append)) == 2
    assert sleeps == [10.]
    readings = iter([snapshot(), snapshot(9.99)])
    with pytest.raises(RuntimeError, match='10 seconds'):
        verify.acquire_gpu_gate(snapshot=lambda: next(readings), sleep=lambda _: None)


def test_visibility_remapping_is_rejected_before_observations(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    with pytest.raises(RuntimeError, match='Unset'):
        verify.acquire_gpu_gate(snapshot=lambda: pytest.fail('Must not inspect remapped GPU'))


def test_snapshot_filters_compute_pids_by_uuid():
    commands = []
    def command(args, **kwargs):
        commands.append(args)
        return (f'2, {verify.EXPECTED_UUID}, 24200\n' if '-i' in args else
                f'other-uuid, 12\n{verify.EXPECTED_UUID}, 34\n')
    assert verify.gpu_snapshot(command=command)['compute_pids'] == [34]
    assert commands[0][2] == '2'


def test_comparison_checks_structure_nonfinite_and_rng_exactness():
    verify.compare_trees(torch.tensor([1.], dtype=torch.float64),
                         torch.tensor([1.+1e-12], dtype=torch.float64))
    with pytest.raises(AssertionError):
        verify.compare_trees(torch.tensor([1.]), torch.tensor([1.+1e-6]), exact=True)
    with pytest.raises(AssertionError):
        verify.compare_trees(torch.tensor([float('nan')]), torch.tensor([float('nan')]))
    with pytest.raises(AssertionError):
        verify.compare_trees(dict(a=1), dict(b=1))


@pytest.mark.parametrize('dimension', verify.DIMENSIONS)
def test_cpu_host_gather_only_regression_all_four_dimensions(dimension):
    from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
    torch.set_num_threads(1)
    configurations = verify.synthetic_data()
    histogram = binary_lattice_mi_statistics(configurations, buffer=1, block=2)
    histogram.prepare_selection()
    result = verify.compare_dimension(configurations, histogram, dimension, 'cpu', updates=2)
    assert result['passed'] and result['updates'] == 2
    assert not result['shared_execution']['reference']
    for row in result['comparisons']:
        assert row['batch']['maximum_absolute_difference'] == 0.
        assert row['rng']['tensor_count'] == 3
        assert row['gradients']['tensor_count'] > 0


def test_cli_rejects_other_gpu_or_cpu_before_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'acquire_gpu_gate', lambda: pytest.fail('No gate for unsupported device'))
    for device in ('cpu', 'cuda:0', 'cuda:1', 'cuda:3'):
        with pytest.raises(SystemExit):
            verify.main(['--device', device, '--output', str(tmp_path/'receipt.json')])


def test_no_overwrite_before_gpu_gate(tmp_path, monkeypatch):
    path = tmp_path/'receipt.json'
    path.write_text('preserve')
    monkeypatch.setattr(verify, 'acquire_gpu_gate', lambda: pytest.fail('No gate when receipt exists'))
    with pytest.raises(FileExistsError):
        verify.main(['--output', str(path)])
    assert path.read_text() == 'preserve'
