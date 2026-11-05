"""Low-RAM execution preserves ordered batches, RNG, objectives and Adam updates."""
from copy import deepcopy
import gc
import json
import weakref

import pytest
import torch

from experiments import rsmi_separate_qcr as driver
from recurrent_rg.checkpoints import build_model
from recurrent_rg.config import ModelConfig, RunConfig
from recurrent_rg.execution import low_ram_execution_policy, select_training_batch


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('dtype', [torch.int8, torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(5, 11, 8, 1), (5, 11, 8, 8, 1), (5, 11, 4, 4, 4, 1)])
def test_fused_selection_exact_for_repeated_unsorted_negative_indices(dtype, shape):
    configurations = torch.arange(torch.tensor(shape).prod()).reshape(shape).to(dtype)
    ids = torch.tensor([4, 1, 1, -1, 0])
    indices = torch.tensor([8, 2, 2, -1, 0, 7])
    before = torch.get_rng_state().clone()
    reference = select_training_batch(configurations, ids, indices, 'cpu', reference=True)
    actual = select_training_batch(configurations, ids, indices, 'cpu')
    assert torch.equal(actual, reference)
    assert actual.dtype == torch.float64 and actual.stride() == reference.stride()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(actual, configurations[ids][:, indices].double())


def test_fused_selection_handles_noncontiguous_and_empty_axes_without_full_reshape():
    configurations = torch.arange(5*11*4*4).reshape(5, 11, 4, 4, 1).double().transpose(2, 3)
    assert not configurations.is_contiguous()
    for ids, indices in ((torch.tensor([3, 1, 3]), torch.tensor([8, 0, 4])),
                         (torch.empty(0, dtype=torch.long), torch.tensor([2, 0])),
                         (torch.tensor([3, 1]), torch.empty(0, dtype=torch.long))):
        expected = select_training_batch(configurations, ids, indices, 'cpu', reference=True)
        actual = select_training_batch(configurations, ids, indices, 'cpu')
        assert torch.equal(actual, expected) and actual.shape == expected.shape


def test_legacy_mask_and_list_selection_semantics_preserved():
    configurations = torch.arange(3*8*2).reshape(3, 8, 2).double()
    for ids, indices in ((torch.tensor([True, False, True]), torch.tensor([7, 2, 4])),
                         ([2, 0, 2], [6, 2, 2])):
        assert torch.equal(select_training_batch(configurations, ids, indices, 'cpu'),
                           configurations[ids][:, indices].double())


def test_fused_batch_keeps_all_C_draws_and_sampling_rng_identical():
    torch.manual_seed(91)
    model = build_model(ModelConfig(dimension=2, hidden=5, theory_dim=5,
        block_factor=3, kernel_sampling='bernoulli', affine_reconstruction=True), 2)
    generator = torch.Generator().manual_seed(43)
    configurations = (2*torch.randint(2, (5, 8, 9, 9, 1), generator=generator)-1).double()
    ids, indices = torch.tensor([3, 1, 3]), torch.tensor([7, 2, 2, 0])
    before = torch.get_rng_state().clone()
    old = select_training_batch(configurations, ids, indices, 'cpu', reference=True)
    with torch.no_grad():
        old_draws = model.coarse_grain(old)
    after_reference = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    new = select_training_batch(configurations, ids, indices, 'cpu')
    with torch.no_grad():
        new_draws = model.coarse_grain(new)
    assert all(torch.equal(a, b) for a, b in zip(old_draws, new_draws))
    assert torch.equal(after_reference, torch.get_rng_state())


def test_split_MI_placement_order_dtype_and_TEST_cpu_contract():
    calls = []
    class Histogram:
        def prepare_selection(self):
            calls.append(('prepare',))
            return self
        def to(self, device):
            calls.append(('to', str(device)))
            return 'placed'
    assert driver.place_mi_histogram_split(Histogram(), 'train', 'cuda:2', selected=True) == 'placed'
    assert calls == [('prepare',), ('to', 'cuda:2')]
    calls.clear()
    driver.place_mi_histogram_split(Histogram(), 'test', 'cuda:2', selected=True, test_cpu=True)
    assert calls == [('to', 'cpu')]
    calls.clear()
    driver.place_mi_histogram_split(Histogram(), 'train', 'cpu', selected=False)
    assert calls == [('to', 'cpu')]


def test_resume_protocol_rejects_silent_old_or_changed_execution():
    low = low_ram_execution_policy()
    reference = low_ram_execution_policy(reference=True)
    with pytest.raises(ValueError, match='predates'):
        driver.validate_resume_low_ram_execution(dict(metadata={}), low)
    driver.validate_resume_low_ram_execution(dict(metadata={}), reference)
    resume = dict(metadata=dict(execution_optimizations=dict(low_ram_execution=deepcopy(low))))
    driver.validate_resume_low_ram_execution(resume, low)
    with pytest.raises(ValueError, match='preserve'):
        driver.validate_resume_low_ram_execution(resume, reference)
    assert low['source_dtype_unchanged'] and not low['pin_memory'] and not low['asynchronous_prefetch']


def tiny_config(tmp_path, *, updates=2, dimension=5, block=3):
    generator = torch.Generator().manual_seed(43)
    train = (2*torch.randint(2, (3, 8, 36, 36, 1), generator=generator)-1).double()
    test = (2*torch.randint(2, (2, 8, 36, 36, 1), generator=generator)-1).double()
    path = tmp_path/'physical.pt'
    torch.save((train, torch.tensor([.2, .3, .4]), test, torch.tensor([.5, .6])), path)
    config = RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=str(path), physical_model='ising2d', coupling_names=['K'],
                  spin_flip_augment=False),
        model=dict(dimension=2, theory_dim=dimension, hidden=96, block_factor=block,
                   binary_lookup=True, kernel_sampling='bernoulli', affine_reconstruction=True),
        training=dict(max_rg_steps=2, epochs=updates, system_batch=2, sample_batch=4,
                      checkpoint_every=1, log_every=1, gradient_mode='qc_r_separate', threads=1),
        evaluation=dict(max_rg_steps=2, samples=4, systems=2)))
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config.to_dict()))
    return config_path


@pytest.mark.parametrize('dimension', [1, 5, 10, 100])
@pytest.mark.parametrize('block', [2, 3])
def test_CPU_reference_and_low_RAM_full_training_match_every_update_Adam_gradients_and_rng(tmp_path, monkeypatch, dimension, block):
    config = tiny_config(tmp_path, dimension=dimension, block=block)
    traces, panel_traces = {}, {}
    original_gradients = driver.independent_component_gradients
    original_evaluate = driver.evaluate
    active = ['reference']
    def gradients(*args, **kwargs):
        result = original_gradients(*args, **kwargs)
        # q/c/r iterables are materialized lists in the driver, in original order.
        parameters = list(args[3])+list(args[4])+list(args[5])
        traces.setdefault(active[0], []).append([
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters])
        return result
    def evaluate(model, panel, *args, **kwargs):
        panel_traces.setdefault(active[0], []).append(panel.clone())
        return original_evaluate(model, panel, *args, **kwargs)
    monkeypatch.setattr(driver, 'independent_component_gradients', gradients)
    monkeypatch.setattr(driver, 'evaluate', evaluate)
    common = ['--config', str(config), '--skip-final-report']
    old, new = tmp_path/'reference', tmp_path/'low_ram'
    driver.main(common+['--output', str(old), '--reference-execution'])
    active[0] = 'low_ram'
    driver.main(common+['--output', str(new)])
    assert_tree_equal(traces['reference'], traces['low_ram'])
    assert_tree_equal(panel_traces['reference'], panel_traces['low_ram'])
    for step in (1, 2):
        a = torch.load(old/f'training_step_{step}.pt', map_location='cpu', weights_only=True)
        b = torch.load(new/f'training_step_{step}.pt', map_location='cpu', weights_only=True)
        for field in ('model', 'optimizer', 'history', 'torch_rng_state', 'batch_generator_state',
                      'rsmi_generator_state', 'lr_scheduler_state'):
            assert_tree_equal(a[field], b[field])
        assert a['metadata']['data_sha256'] == b['metadata']['data_sha256']
        assert not a['metadata']['labels_used_in_training'] and not b['metadata']['labels_used_in_training']
        assert not a['metadata']['execution_optimizations']['low_ram_execution']['enabled']
        assert b['metadata']['execution_optimizations']['low_ram_execution']['enabled']


def test_low_RAM_driver_releases_each_CPU_MI_object_before_next_build(tmp_path, monkeypatch):
    config = tiny_config(tmp_path, updates=1)
    references = []
    original_statistics = driver.binary_lattice_mi_statistics
    def statistics(*args, **kwargs):
        if references:
            gc.collect()
            assert references[-1]() is None, 'Prior CPU histogram object retained until next split load'
        histogram = original_statistics(*args, **kwargs)
        references.append(weakref.ref(histogram))
        return histogram
    monkeypatch.setattr(driver, 'binary_lattice_mi_statistics', statistics)
    driver.main(['--config', str(config), '--output', str(tmp_path/'low_ram'), '--skip-final-report'])
    assert len(references) == 2 and all(reference() is None for reference in references)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA; CPU tests do not certify CUDA execution')
def test_CUDA_fused_selection_preserves_resident_and_CPU_transfer_paths():
    configurations = torch.arange(3*8*4).reshape(3, 8, 4).double()
    ids, indices = torch.tensor([2, 0, 2]), torch.tensor([7, 2, 7, 1])
    CPU_rng = torch.get_rng_state().clone()
    CUDA_rng = torch.cuda.get_rng_state().clone()
    for source in (configurations, configurations.cuda()):
        a = select_training_batch(source, ids, indices, 'cuda:0', reference=True)
        b = select_training_batch(source, ids, indices, 'cuda:0')
        assert torch.equal(a, b) and a.dtype == b.dtype == torch.float64
    assert torch.equal(CPU_rng, torch.get_rng_state())
    assert torch.equal(CUDA_rng, torch.cuda.get_rng_state())
