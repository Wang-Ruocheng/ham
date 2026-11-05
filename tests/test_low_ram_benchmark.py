"""Synthetic CPU benchmark never reads formal experiment inputs."""
import copy
import json

import pytest

from scripts.benchmark_low_ram_io import (
    MIB, build_parser, compare, main, parse_proc_kib, peak_rss_bytes, workload_shape,
)


def test_ru_maxrss_units_are_explicit_and_platform_correct():
    assert peak_rss_bytes(1234, platform='darwin') == 1234
    assert peak_rss_bytes(1234, platform='linux') == 1234 * 1024
    with pytest.raises(RuntimeError, match='Linux and macOS'):
        peak_rss_bytes(1234, platform='unknown')


def test_linux_rollup_does_not_confuse_pss_and_rss_units():
    values = parse_proc_kib('Rss: 100 kB\nAnonymous: 40 kB\nPss_File: 57 kB\nThreads: 4\n')
    assert values == {'Rss': 100*1024, 'Anonymous': 40*1024, 'Pss_File': 57*1024}


@pytest.mark.parametrize('size', [0, 1, -4, 1025, float('inf'), float('nan')])
def test_size_guard_rejects_large_or_invalid_payload_without_output(tmp_path, size):
    with pytest.raises(ValueError):
        workload_shape(size)
    with pytest.raises(SystemExit):
        main(['--output', str(tmp_path/'new'), '--size-mib', str(size)])
    assert not (tmp_path/'new').exists()


def test_default_is_256_MiB_and_shapes_stay_within_requested_limit():
    assert build_parser().parse_args(['--output', 'new']).size_mib == 256
    for size in (2, 8, 256, 1024):
        shape = workload_shape(size)
        assert shape['tensor_payload_bytes'] <= size*MIB
        assert shape['train_shape'][0] >= 2 and shape['test_shape'][0] >= 2
        assert shape['train_shape'][1:] == shape['test_shape'][1:] == [32, 36, 36, 1]


def test_existing_output_is_never_overwritten(tmp_path):
    sentinel = tmp_path/'sentinel.txt'
    sentinel.write_text('preserve')
    with pytest.raises(SystemExit):
        main(['--output', str(tmp_path), '--size-mib', '2'])
    assert sentinel.read_text() == 'preserve'
    assert not (tmp_path/'synthetic.pt').exists()


def test_tiny_independent_processes_preserve_full_content_order_rng_and_source(tmp_path):
    output = tmp_path/'synthetic_audit'
    assert main(['--output', str(output), '--size-mib', '2']) == 0
    manifest = json.loads((output/'synthetic_manifest.json').read_text())
    reference = json.loads((output/'reference.json').read_text())
    optimized = json.loads((output/'optimized.json').read_text())
    result = json.loads((output/'benchmark.json').read_text())
    assert result['status'] == 'complete' and result['comparison']['passed']
    assert manifest['synthetic'] and manifest['dtype'] == 'torch.float64'
    assert manifest['tensor_payload_bytes'] <= 2*MIB
    assert result['synthetic'] and not result['model_training'] and not result['CUDA_used']
    assert not result['MI_cache_loaded'] and not result['formal_experiment_RAM_requirement_certified']
    assert reference['process_id'] != optimized['process_id']
    assert reference['content_and_selection_sha256'] == optimized['content_and_selection_sha256']
    assert reference['source_sha256_before'] == optimized['source_sha256_after'] == manifest['cache_sha256']
    for item in (reference, optimized):
        assert item['source_unchanged'] and item['device'] == 'cpu' and item['threads'] == 1
        assert item['global_rng_before'] == item['global_rng_after']
        assert item['peak_rss_bytes'] > 0
        assert [stage['name'] for stage in item['stages']] == [
            'load_and_complete_validation', 'evaluation_panel_setup',
            'same_order_training_batch_gathers',
            'same_order_one_split_at_a_time_evaluation_panels', 'all_source_values_digest']
        assert all(stage['seconds'] >= 0 for stage in item['stages'])
    changed = copy.deepcopy(optimized)
    changed['content_and_selection_sha256'] = 'changed'
    assert not compare(reference, changed)['passed']
    changed = copy.deepcopy(optimized)
    changed['source_unchanged'] = False
    assert not compare(reference, changed)['passed']
