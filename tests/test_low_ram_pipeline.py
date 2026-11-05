"""Low-RAM kwargs do not bypass the high-dimensional cache-only guard."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('cache_hit', [True, False])
@pytest.mark.parametrize('benchmark_warmup', [None, 0])
def test_cache_only_forwards_mmap_but_never_allows_rebuilding(tmp_path, monkeypatch, cache_hit, benchmark_warmup):
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import mi_cache
    from scripts import run_ising2d_dimg_ablation as pipeline
    from scripts import report_ising_zh

    calls = []
    def existing(directory, key_data, build, *, mmap=False):
        calls.append((directory, key_data, mmap))
        if not cache_hit:
            build()
        return 'unchanged statistics', dict(hit=cache_hit)
    def training(argv):
        assert '--skip-final-report' in argv
        if benchmark_warmup is None:
            assert '--benchmark-warmup' not in argv
        else:
            assert argv[argv.index('--benchmark-warmup')+1] == str(benchmark_warmup)
        statistics, receipt = mi_cache.load_or_build_mi_statistics('existing', {'split': 'train'},
            lambda: pytest.fail('Caller builder must not run'), mmap=True)
        assert statistics == 'unchanged statistics' and receipt['hit']
    monkeypatch.setattr(mi_cache, 'load_or_build_mi_statistics', existing)
    monkeypatch.setattr(trainer, 'main', training)
    monkeypatch.setattr(report_ising_zh, 'chinese_figures', nullcontext)
    original_model = trainer.build_model
    args = SimpleNamespace(config=tmp_path/'c.json', output=tmp_path/'run', device='cuda:2',
        mi_cache_dir=tmp_path/'mi', initialization_receipt=tmp_path/'initial.json', benchmark_warmup=benchmark_warmup)
    if cache_hit:
        pipeline.cached_training(args)
    else:
        with pytest.raises(RuntimeError, match='禁止重算'):
            pipeline.cached_training(args)
    assert calls == [('existing', {'split': 'train'}, True)]
    assert mi_cache.load_or_build_mi_statistics is existing
    assert trainer.build_model is original_model
