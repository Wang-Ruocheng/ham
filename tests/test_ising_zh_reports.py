"""Chinese presentation must not change the scientific reporting pipeline."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pytest
import torch

from scripts import report_ising_zh as zh


@pytest.fixture
def chinese_font():
    try:
        return zh.select_chinese_font()
    except RuntimeError as error:
        pytest.skip(str(error))


def visible_text(figure):
    return [artist for artist in figure.findobj(match=Text) if artist.get_visible()]


@pytest.mark.parametrize('source', list(zh._TEXT))
def test_all_static_display_strings_are_translated(source):
    result = zh.translate_text(source)
    assert zh._CHINESE.search(result)
    assert zh._check_prose(result) == result


@pytest.mark.parametrize(('source', 'expected'), [
    ('ensemble 117, g2: Q', '系综 117，g2：Q'),
    ('F2: unstable; extrapolation', 'F2：不稳定；外推'),
    ('F1: stable', 'F1：稳定'),
    ('F13: nonhyperbolic', 'F13：非双曲'),
    ('train: color NLL', '训练集：颜色保持 NLL'),
    ('test: color NLL', '测试集：颜色保持 NLL'),
    ('separation (raw)', 'Q 表示目标（未加权）'),
    ('coarse_consistency (raw)', '粗粒化子集一致性（未加权）'),
    ('balanced_reconstruction (raw)', '平衡重构（未加权）'),
    ('coupling_supervision (raw)', '微观 K 标签监督（未加权）'),
    ('weighted_coupling_supervision (raw)', '加权微观 K 标签监督（未加权）'),
    ('train: coupling_supervision', '训练集：微观 K 标签监督'),
    ('test: weighted_coupling_supervision', '测试集：加权微观 K 标签监督'),
    ('Random C + independent gradients: test', '随机 C 与独立梯度：测试集'),
    ('Frozen learned C + independent QR: train', '冻结已学习的 C 与独立 Q、R 梯度：训练集'),
    ('Scale 0 -> 1', '尺度 0 → 1'),
    ('Theory thermal lambda=b=2', '理论热乘子 λ=b=2'),
    ('Theory thermal lambda=b=3', '理论热乘子 λ=b=3'),
    ('Input Kc=0.4406867935', '输入 Kc=0.4406867935'),
    ('nu absent for 7 reported roots', '7 个已报告不动点的 ν 缺失'),
    ('outside TRAIN; negative lambda; unstable', '训练支持之外；负 λ；不稳定'),
    ('unknown support; nonhyperbolic', '支持未知；非双曲'),
    ('inside TRAIN; stable', '训练支持之内；稳定'),
    ('2 completed previews; 1 skipped. Hollow markers: outside/unknown TRAIN support.',
     '2 个预览已完成；跳过 1 个。空心标记：位于训练支持之外或支持未知。'),
])
def test_dynamic_labels_preserve_values(source, expected):
    assert zh.translate_text(source) == expected


@pytest.mark.parametrize('formula', [
    '', 'g', 'g1', 'k=2', 'K', 'R(g)', 'R(g)-g', 'Q(C^2D)',
    '−0.5', '1e−3', r'$\mathdefault{10^{−3}}$',
])
def test_identifiers_formulae_and_numeric_ticks_are_unchanged(formula):
    assert zh.translate_text(formula) == formula


@pytest.mark.parametrize('prose', ['New English title', '中文仍有 forgotten prose', 'loss', 'input g',
                                  '$Unknown English prose$', r'$\mathrm{unmapped}$'])
def test_unmapped_prose_is_rejected(prose):
    with pytest.raises(ValueError, match='Untranslated figure prose'):
        zh.translate_text(prose)


def test_missing_explicit_font_fails_without_fallback(tmp_path):
    with pytest.raises(RuntimeError, match='No existing font'):
        zh.select_chinese_font(tmp_path / 'missing.ttf')


def test_save_translates_all_text_and_restores_hook(tmp_path, chinese_font):
    original = Figure.savefig
    fig, ax = plt.subplots(layout='constrained')
    line, = ax.plot([0., 1.], [0., 2.], label='Random C + independent gradients: test')
    collection = ax.scatter([.5], [1.], c=[.25])
    fig.colorbar(collection, ax=ax, label='|R(g)-g| (arrows normalized)')
    ax.set(title='Discrete scalar RG map', xlabel='updates', ylabel='weighted loss')
    ax.annotate('F1: stable', (.5, 1.))
    ax.legend(title='test')
    fig.suptitle('Full discrete map')
    fig.text(.1, .01, 'Saved R roots')
    before = (line.get_xdata().copy(), line.get_ydata().copy())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with zh.chinese_figures(chinese_font):
                fig.savefig(tmp_path / 'chinese.png', dpi=70)
        assert (tmp_path / 'chinese.png').stat().st_size > 0
        assert not any('Glyph' in str(w.message) and 'missing' in str(w.message) for w in caught)
        assert all(Path(artist.get_fontproperties().get_file()) == chinese_font
                   for artist in visible_text(fig))
        assert ax.get_title() == '离散标量 RG 映射'
        assert ax.get_legend().get_title().get_text() == '测试集'
        assert all(zh.translate_text(artist.get_text()) == artist.get_text() for artist in visible_text(fig))
        np.testing.assert_array_equal(line.get_xdata(), before[0])
        np.testing.assert_array_equal(line.get_ydata(), before[1])
    finally:
        plt.close(fig)
    assert Figure.savefig is original


def test_unknown_figure_text_fails_before_saving_or_partial_translation(tmp_path, chinese_font):
    fig, ax = plt.subplots()
    ax.set(title='Identity', xlabel='Unmapped words')
    original = Figure.savefig
    try:
        with pytest.raises(ValueError, match='Untranslated figure prose'):
            with zh.chinese_figures(chinese_font):
                fig.savefig(tmp_path / 'must_not_exist.png')
        assert not (tmp_path / 'must_not_exist.png').exists()
        assert ax.get_title() == 'Identity'
        assert Figure.savefig is original
    finally:
        plt.close(fig)


class ToyModel:
    def advance_theory(self, x):
        return .5 * x


def test_generic_inline_reports_keep_flow_arrays_and_json(tmp_path, chinese_font, monkeypatch):
    from recurrent_rg.plotting.reports import plot_diagnostics
    direct = np.array([[[-.4], [-.2], [-.1]], [[.4], [.2], [.1]]])
    trajectories = {'direct': direct, 'rollout': direct.copy()}
    before = direct.copy()
    plain, chinese = tmp_path / 'plain', tmp_path / 'chinese'
    plot_diagnostics(ToyModel(), trajectories, np.array([[.4], [.5]]), ['K'], plain,
                     training_inputs=direct[:, :-1])
    captured = []
    original = Figure.savefig

    def inspect(figure, *args, **kwargs):
        captured.append([artist.get_text() for artist in visible_text(figure)])
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', inspect)
    with zh.chinese_figures(chinese_font):
        plot_diagnostics(ToyModel(), trajectories, np.array([[.4], [.5]]), ['K'], chinese,
                         training_inputs=direct[:, :-1])
    assert (plain / 'fixed_points.json').read_bytes() == (chinese / 'fixed_points.json').read_bytes()
    assert len(captured) == 3
    assert any('恒等映射' in texts for texts in captured)
    assert all(zh.translate_text(text) == text for texts in captured for text in texts)
    np.testing.assert_array_equal(direct, before)
    assert {path.name for path in plain.iterdir()} == {path.name for path in chinese.iterdir()}


def test_actual_mi_loss_report_preserves_json_and_all_history_values(tmp_path, chinese_font, monkeypatch):
    from scripts import report_mi_training as reporter
    run = tmp_path / 'run'
    run.mkdir()
    weights = dict(anchor=20., coarse_consistency=5., kernel_noncollapse=2.,
                   separation=10., commutator=5.)
    loss = dict(weights=weights, rollout_weight=1., kernel_sign_weight=1.,
                balanced_reconstruction_weight=10.)
    config = dict(loss=loss, model=dict(categorical_states=0))
    manifest = dict(config=config, independent_c=True)
    history = [dict(epoch=epoch, **{
        f'{split}_{key}': (10. if key in ('total', 'qcr_total') else .1) + epoch / 1000
        for split in ('train', 'test')
        for key in ('anchor', 'coarse_consistency', 'kernel_noncollapse', 'separation',
                    'commutator', 'rollout', 'kernel_sign', 'balanced_reconstruction',
                    'total', 'qcr_total', 'rsmi_bits')}) for epoch in (1, 2)]
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'history_progress.json').write_text(json.dumps(history))
    loaded = SimpleNamespace(config=config, payload={}, model=SimpleNamespace(
        rg_kernel=SimpleNamespace(geometry=SimpleNamespace(block_volume=4))))
    monkeypatch.setattr(reporter, 'load_run', lambda *_: loaded)
    monkeypatch.setattr(reporter, 'kernel_probabilities',
                        lambda _, patterns: torch.full((1, patterns.shape[1], 2), .5))
    plain, chinese = tmp_path / 'plain_loss', tmp_path / 'chinese_loss'
    source_before = (run / 'history_progress.json').read_bytes()
    for output in (plain, chinese):
        monkeypatch.setattr(sys, 'argv', ['report', '--run', str(run), '--output', str(output)])
        if output == chinese:
            with zh.chinese_figures(chinese_font):
                reporter.main()
        else:
            reporter.main()
    assert (plain / 'training_comparison.json').read_bytes() == (chinese / 'training_comparison.json').read_bytes()
    assert (chinese / 'loss_comparison.png').stat().st_size > 0
    assert (chinese / 'balanced_Q_loss.png').stat().st_size > 0
    assert (run / 'history_progress.json').read_bytes() == source_before
    plt.close('all')


def test_exponent_and_trend_figures_keep_block_two_theory_and_all_roots(tmp_path, chinese_font):
    from scripts import report_ising_scalar_exponents as exponent
    from scripts import report_scalar_exponent_trend as trend
    assert exponent.exponent_from_multiplier(2., 2.)['nu'] == pytest.approx(1.)
    roots = [dict(g=[0.], stability='stable', outside_training_support=False,
                  microscopic_inverse_coupling=[.4, .5])]
    knots = np.array([.4, .5])
    means = np.array([[-.2, -.1], [.2, .1]])
    reference = dict(Kc=exponent.KC, g_by_scale=[0., 0.], R_at_g_by_scale=[0., 0.])
    report = dict(snapshots=[dict(checkpoint_epoch=200, roots=[dict(
        category='outside TRAIN; negative lambda; unstable', sign_reversing=True,
        stability='unstable', outside_training_support=True, multiplier=-2.,
        conditional_nu=None, microscopic_inverse_coupling=[.4, .5],
        inverse_plateau_intervals_from_saved_encodings=[])])], skipped=[],
        theory_references=dict(thermal_lambda=[dict(block_factor=2, multiplier=2)]))
    before = json.dumps(report, sort_keys=True)
    with zh.chinese_figures(chinese_font):
        exponent.draw_summary(ToyModel(), roots, knots, means, reference, [-.2, .2], [-.4, .4], tmp_path)
        trend.plot_trend(report, tmp_path)
        trend.plot_trend(dict(snapshots=[], skipped=[], theory_references=dict(thermal_lambda=[])), tmp_path)
    assert (tmp_path / 'scalar_critical_summary.png').stat().st_size > 0
    assert (tmp_path / 'trend.png').stat().st_size > 0
    assert json.dumps(report, sort_keys=True) == before
    assert report['theory_references']['thermal_lambda'][0]['multiplier'] == 2
    assert len(report['snapshots'][0]['roots'][0]['microscopic_inverse_coupling']) == 2


def test_wrapper_forwards_arguments_without_changing_them(chinese_font, monkeypatch):
    original_argv = sys.argv
    original_savefig = Figure.savefig
    seen = []

    def module(name, run_name):
        seen.append((name, run_name, list(sys.argv), Figure.savefig is not original_savefig))

    monkeypatch.setattr(zh.runpy, 'run_module', module)
    assert zh.main(['--module', 'experiments.rsmi_separate_qcr', '--font', str(chinese_font),
                    '--config', 'new.json', '--output', 'new_run']) == 0
    assert seen == [('experiments.rsmi_separate_qcr', '__main__',
                     ['experiments.rsmi_separate_qcr', '--config', 'new.json', '--output', 'new_run'], True)]
    assert sys.argv is original_argv
    assert Figure.savefig is original_savefig


def test_wrapper_rejects_modules_outside_audited_scope():
    with pytest.raises(SystemExit):
        zh.main(['--module', 'unrelated.module'])


def test_wrapper_allows_the_supervised_report_only(chinese_font, monkeypatch):
    seen = []
    monkeypatch.setattr(zh.runpy, 'run_module',
                        lambda name, run_name: seen.append((name, run_name, list(sys.argv))))
    assert zh.main(['--module', 'scripts.report_k_supervised_pilot', '--font', str(chinese_font),
                    '--run', 'new_run']) == 0
    assert seen == [('scripts.report_k_supervised_pilot', '__main__',
                     ['scripts.report_k_supervised_pilot', '--run', 'new_run'])]
    assert 'scripts.run_ising2d_k_supervised_pilot' not in zh.SUPPORTED_MODULES
