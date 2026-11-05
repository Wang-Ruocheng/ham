"""Run an existing Ising report with Chinese text on newly saved figures.

Examples::

    python -m scripts.report_ising_zh --module scripts.report_scalar_joint \
        --run NEW_RUN --output NEW_RUN/full_report
    python -m scripts.report_ising_zh --module experiments.rsmi_separate_qcr \
        --config NEW_CONFIG --output NEW_RUN

Only Matplotlib text and its font are changed. The selected module still owns
all computations, output paths, numerical reports, and training arguments.
No saved historical figure is opened or rewritten. Each subprocess producing
figures must use this wrapper separately; a patch does not cross subprocesses.
Unknown English prose fails before a figure is saved, so a new report label
cannot silently escape the Chinese-figure requirement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
import re
import runpy
import sys


SUPPORTED_MODULES = (
    'experiments.rsmi_separate_qcr',
    'scripts.report_scalar_joint',
    'scripts.report_ising_scalar_exponents',
    'scripts.report_mi_training',
    'scripts.report_ising_local_response',
    'scripts.report_scalar_exponent_trend',
    'scripts.report_k_supervised_pilot',
    'scripts.report_k_supervised_continuation',
)


# These are display strings only: JSON keys and scientific calculations never
# pass through the translator. In particular no block factor is substituted.
_TEXT = {
    'Identity': '恒等映射',
    'Training input range': '训练输入范围',
    'Discrete scalar RG map': '离散标量 RG 映射',
    '|R(g)-g| (arrows normalized)': '|R(g)-g|（箭头已归一化）',
    'Evaluated encodings': '评估编码',
    'Training R inputs': 'R 的训练输入',
    'Discrete displacement R(g)-g; off-data arrows are extrapolation':
        '离散位移 R(g)-g；数据范围外的箭头表示外推',
    'Microscopic Q(D)': '微观编码 Q(D)',
    'Direct Q: solid; autonomous R rollout: dashed': '直接 Q 编码：实线；R 自主迭代：虚线',
    'Q(C^k D): mean over ensembles': 'Q(C^k D)：系综平均',
    'updates': '更新次数',
    'Training update': '训练更新次数',
    'weighted loss': '加权损失',
    'train': '训练集',
    'test': '测试集',
    'anchor': '微观子集一致性',
    'coarse_consistency': '粗粒化子集一致性',
    'kernel_noncollapse': 'C 方差下限',
    'separation': 'Q 表示目标',
    'commutator': '单步闭合',
    'rollout': '多步闭合',
    'isotropy': '各向同性',
    'tangent': '切向约束',
    'kernel_sign': '均匀块颜色保持',
    'kernel_neutral': '中性块正则项',
    'balanced_reconstruction': '平衡重构',
    'coupling_supervision': '微观 K 标签监督',
    'weighted_coupling_supervision': '加权微观 K 标签监督',
    'q_coupling_weight': 'K 标签监督权重',
    'QCR total (MI excluded if present)': 'QCR 总损失（不含 MI）',
    'Common QCR loss (C and MI excluded)': '共同 QCR 损失（不含 C 与 MI）',
    'Weighted one-step + rollout': '加权单步与多步闭合',
    'Weighted C regularizers': '加权 C 正则项',
    'New run: exact empirical MI, bits': '新实验：精确经验 MI（比特）',
    'Frozen learned C + independent QR': '冻结已学习的 C 与独立 Q、R 梯度',
    'Independent C + MI init': '独立 C 梯度与 MI 初始化',
    'Random C + independent gradients': '随机 C 与独立梯度',
    'Random C + QC/R gradients': '随机 C 与 Q、C/R 梯度',
    'Baseline': '基线',
    'Full R(g)': '完整 R(g)',
    'TRAIN R-input range': 'R 的训练输入范围',
    'Outside TRAIN range': '训练范围之外',
    'Known input Kc (reference)': '已知输入 Kc（参照）',
    'Encoded Kc input (not an R root)': 'Kc 输入编码（非 R 不动点）',
    'Saved R roots': '已保存的 R 不动点',
    'Full discrete map': '完整离散映射',
    'Discrete RG displacement': '离散 RG 位移',
    'Saved coupling-to-g maps': '已保存的耦合到 g 映射',
    'Initial K=J/T, B=0': '初始 K=J/T，B=0',
    'All reported roots; no root or checkpoint selection. Scalar B=0 R does not determine y_h or beta.':
        '保留全部已报告不动点；未筛选不动点或检查点。B=0 的标量 R 无法确定 y_h 或 β。',
    'Saved mean trajectory': '已保存的平均轨迹',
    'g input': '输入 g',
    'g output': '输出 g',
    'Data secant response': '数据割线响应',
    'R paired response': 'R 配对响应',
    'Original K': '初始 K',
    'Response ratio': '响应比',
    'Adjacent K pairs; input SNR >= 3': '相邻 K 对；输入信噪比 ≥ 3',
    '12000 endpoint: local closure response (not thermal eigenvalue estimates)':
        '12000 次更新终点：局部闭合响应（非热本征值估计）',
    'Theory nu=1': '理论参照 ν=1',
    'Full-R lambda': '完整 R 的 λ',
    'Conditional thermal nu': '有条件的热指数 ν',
    'Interpolated microscopic K': '插值得到的微观 K',
    'All saved root multipliers': '全部已保存不动点的乘子',
    'Eligible values only; null stays absent': '仅显示满足条件的值；空值保持缺失',
    'All inverse-coupling branches': '全部逆耦合分支',
    'Actual checkpoint update': '检查点实际更新次数',
    'No completed usable exponent previews': '暂无已完成且可用的指数预览',
    'Independent per-checkpoint scatter points; no cross-epoch root matching':
        '各检查点散点相互独立；未跨更新次数匹配不动点',
}
_STABILITY = {
    'stable': '稳定', 'unstable': '不稳定', 'nonhyperbolic': '非双曲',
    'saddle': '鞍点', 'unknown': '未知',
}
_RUN_LABELS = (
    'Frozen learned C + independent QR', 'Independent C + MI init',
    'Random C + independent gradients', 'Random C + QC/R gradients', 'Baseline',
)
_IDENTIFIER = re.compile(r'(?:R|Q|C|D|RG|QCR|MI|NLL|Kc|K|B|J|T|g\d*|F\d*|k|b|y_h|e|inf|nan)')
_WORDS = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')
_CHINESE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
_MATH_COMMAND = re.compile(
    r'\\(?:mathdefault|mathregular|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|text|'
    r'frac|dfrac|tfrac|sqrt|left|right|langle|rangle|cdot|times|pm|mp|infty|'
    r'lambda|nu|beta|alpha|theta|delta|Delta|sigma|pi|epsilon|partial|sum|prod|'
    r'ln|log|exp|abs|vert|geq|leq|neq|approx|equiv|quad|qquad)(?![A-Za-z])')


def _check_prose(text):
    # Remove known TeX commands, not their contents: wrapping English prose in
    # mathtext must not bypass the guard. Numeric ticks/formulae stay unchanged.
    prose = _MATH_COMMAND.sub('', text)
    unknown = [word for word in _WORDS.findall(prose) if not _IDENTIFIER.fullmatch(word)]
    if unknown:
        raise ValueError(f'Untranslated figure prose {unknown!r} in {text!r}; add an explicit Chinese translation')
    return text


def translate_text(text):
    """Translate a complete display string, retaining all numeric values."""
    if '\n' in text:
        return '\n'.join(translate_text(line) for line in text.split('\n'))
    if text in _TEXT:
        return _check_prose(_TEXT[text])
    if text in _STABILITY:
        return _STABILITY[text]
    match = re.fullmatch(r'ensemble (\d+), g(\d+): Q', text)
    if match:
        return f'系综 {match[1]}，g{match[2]}：Q'
    match = re.fullmatch(r'F(\d+): (stable|unstable|nonhyperbolic|saddle|unknown)(; extrapolation)?', text)
    if match:
        suffix = '；外推' if match[3] else ''
        return f'F{match[1]}：{_STABILITY[match[2]]}{suffix}'
    match = re.fullmatch(r'(train|test): color NLL', text)
    if match:
        return f'{_TEXT[match[1]]}：颜色保持 NLL'
    match = re.fullmatch(r'(train|test): (coupling_supervision|weighted_coupling_supervision)', text)
    if match:
        return f'{_TEXT[match[1]]}：{_TEXT[match[2]]}'
    match = re.fullmatch(r'(.+) \(raw\)', text)
    if match and match[1] in _TEXT:
        return _check_prose(_TEXT[match[1]] + '（未加权）')
    for label in _RUN_LABELS:
        match = re.fullmatch(re.escape(label) + r': (train|test)', text)
        if match:
            return _TEXT[label] + '：' + _TEXT[match[1]]
    match = re.fullmatch(r'Scale (\d+) -> (\d+)', text)
    if match:
        return f'尺度 {match[1]} → {match[2]}'
    match = re.fullmatch(r'Theory thermal lambda=b=(.+)', text)
    if match:
        return _check_prose(f'理论热乘子 λ=b={match[1]}')
    match = re.fullmatch(r'Input Kc=(.+)', text)
    if match:
        return _check_prose(f'输入 Kc={match[1]}')
    match = re.fullmatch(r'nu absent for (\d+) reported roots', text)
    if match:
        return f'{match[1]} 个已报告不动点的 ν 缺失'
    match = re.fullmatch(r'(\d+) completed previews; (\d+) skipped\. Hollow markers: outside/unknown TRAIN support\.', text)
    if match:
        return f'{match[1]} 个预览已完成；跳过 {match[2]} 个。空心标记：位于训练支持之外或支持未知。'
    match = re.fullmatch(r'(outside TRAIN|inside TRAIN|unknown support); (negative lambda; )?(stable|unstable|nonhyperbolic|saddle|unknown)', text)
    if match:
        support = {'outside TRAIN': '训练支持之外', 'inside TRAIN': '训练支持之内',
                   'unknown support': '支持未知'}[match[1]]
        negative = '；负 λ' if match[2] else ''
        return f'{support}{negative}；{_STABILITY[match[3]]}'
    return _check_prose(text)


@lru_cache(maxsize=8)
def _font_codepoints(font_path):
    from matplotlib.ft2font import FT2Font
    return frozenset(FT2Font(str(font_path)).get_charmap())


def select_chinese_font(explicit=None):
    """Use an existing CJK font; never install or download system fonts."""
    from matplotlib import font_manager
    explicit = explicit or os.environ.get('QCR_ZH_FONT')
    candidates = [Path(explicit).expanduser()] if explicit else [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'),
        Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),
        Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
        Path('/System/Library/Fonts/PingFang.ttc'),
    ]
    if not explicit:
        candidates += [Path(item.fname) for item in font_manager.fontManager.ttflist
                       if any(name in item.name for name in
                              ('Noto Sans CJK', 'Source Han Sans', 'WenQuanYi',
                               'Droid Sans Fallback', 'Arial Unicode', 'PingFang'))]
    needed = {ord(c) for c in ''.join(_TEXT.values()) + ''.join(_STABILITY.values())
              if _CHINESE.fullmatch(c)}
    for path in dict.fromkeys(candidates):
        if path.is_file() and needed <= _font_codepoints(str(path.resolve())):
            return path.resolve()
    raise RuntimeError('No existing font covers the Chinese figure text; pass --font /path/to/CJK-font or set QCR_ZH_FONT')


def translate_figure(figure, font_path):
    """Translate all Text artists, including legends, colorbars and annotations."""
    from matplotlib.text import Text
    font_path = str(Path(font_path).resolve())
    available = _font_codepoints(font_path)
    artists = [artist for artist in figure.findobj(match=Text) if artist.get_visible()]
    prepared = []
    for artist in artists:
        translated = translate_text(artist.get_text())
        missing = sorted({c for c in translated if _CHINESE.fullmatch(c) and ord(c) not in available})
        if missing:
            raise RuntimeError(f'Chinese font {font_path} lacks figure glyphs: {"".join(missing)}')
        properties = artist.get_fontproperties().copy()
        properties.set_file(font_path)
        prepared.append((artist, translated, properties))
    # Validate the entire figure before mutating any artist.
    for artist, translated, properties in prepared:
        artist.set_text(translated)
        artist.set_fontproperties(properties)
    return [translated for _, translated, _ in prepared]


@contextmanager
def chinese_figures(font=None):
    """Scope the save hook to this execution, restoring it even on failure."""
    from matplotlib.figure import Figure
    font_path = select_chinese_font(font)
    original = Figure.savefig

    def savefig(figure, *args, **kwargs):
        translate_figure(figure, font_path)
        figure.canvas.draw()  # Materialize numeric ticks before the final check.
        translate_figure(figure, font_path)
        return original(figure, *args, **kwargs)

    Figure.savefig = savefig
    try:
        yield font_path
    finally:
        Figure.savefig = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True, choices=SUPPORTED_MODULES)
    parser.add_argument('--font', type=Path, help='Existing Chinese font file; no installation')
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    old_argv = sys.argv
    try:
        sys.argv = [args.module, *forwarded]
        with chinese_figures(args.font):
            runpy.run_module(args.module, run_name='__main__')
    finally:
        sys.argv = old_argv
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
