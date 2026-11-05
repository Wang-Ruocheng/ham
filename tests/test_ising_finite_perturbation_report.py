from pathlib import Path
import numpy as np
import pytest
from scripts import report_ising_finite_perturbation as report


def test_standard_response_zero_variance_not_infinite():
    a=np.ones((2,1,2,3,6));std=np.ones((2,1,2,6));std[...,0]=0
    x,y=report.standard_response(a,a,std)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all(x[...,0]==0)


def test_complete_jacobian_response_not_displacement_only():
    rng=np.random.default_rng(1);a=rng.normal(size=(4,2,2,3,6))
    std=np.ones((4,2,2,6));dr=np.broadcast_to(2*np.eye(3),(4,2,3,3,3)).copy()
    d=report.relative_response(dr,a,2*a,std)
    np.testing.assert_allclose(d['R_relative_per_replica'],0,atol=1e-15)
    np.testing.assert_allclose(d['best_J_relative_per_replica'],0,atol=1e-12)


def test_chain_bootstrap_shape():
    d=report.interval(np.arange(12*3).reshape(12,3),repeats=100)
    assert d.shape==(3,3)


def test_no_model_or_checkpoint_loading():
    code=Path(report.__file__).read_text()
    assert 'torch.load' not in code and 'load_run(' not in code
    assert 'thermal_mode_certified=False' in code
    assert "thermal_nu=None" in code


def test_missing_saved_array_never_triggers_inference():
    with pytest.raises(ValueError,match='不能临时模型推理'):
        report.find_array({},'initial','L108','direct')


def test_alias_names_are_explicit():
    value=np.ones(1)
    assert report.find_array({'initial_L108_direct':value},'initial','L108','direct') is value


def test_pointwise_R_plot_keeps_nonlinear_draw_points_not_C_means():
    direct=np.array([[[[-1.,2.,3.]],[[1.,4.,5.]]]])
    advanced=direct**2
    g,rg=report.pointwise_R_plot_arrays(dict(direct=direct,r_of_g=advanced))
    np.testing.assert_array_equal(g,direct.reshape(-1,3))
    np.testing.assert_array_equal(rg,advanced.reshape(-1,3))
    np.testing.assert_array_equal(g[:,0],[-1.,1.])
    # mean R(g)=1 but R(mean g)=0 in the first coordinate: neither mean is
    # allowed to masquerade as a real point of the nonlinear mapping.
    assert advanced.mean(1)[0,0,0]==1. and direct.mean(1)[0,0,0]**2==0.


def test_pointwise_R_plot_subsampling_is_fixed_paired_and_bounded():
    direct=np.arange(401*3,dtype=float).reshape(1,401,1,3)
    advanced=direct**2+3
    p=dict(direct=direct,r_of_g=advanced)
    g,rg=report.pointwise_R_plot_arrays(p,max_points=180)
    np.testing.assert_array_equal(g,direct.reshape(-1,3)[::3])
    np.testing.assert_array_equal(rg,advanced.reshape(-1,3)[::3])
    assert len(g)<=180
    repeated=report.pointwise_R_plot_arrays(p,max_points=180)
    np.testing.assert_array_equal(g,repeated[0]);np.testing.assert_array_equal(rg,repeated[1])
    with pytest.raises(ValueError,match='正整数'):
        report.pointwise_R_plot_arrays(p,max_points=0)


def test_report_scatter_and_arrows_use_the_same_real_draws_and_honest_aggregate_label(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    monkeypatch.setattr(report,'chinese_render_guard',lambda:nullcontext())
    monkeypatch.setattr(Figure,'savefig',lambda *a,**k:None)
    monkeypatch.setattr(Figure,'tight_layout',lambda *a,**k:None)
    scatter,arrows,titles=[],[],[]
    original_scatter,original_quiver,original_title=Axes.scatter,Axes.quiver,Axes.set_title
    def capture_scatter(self,x,y,*a,**k):
        if np.asarray(x).dtype.kind!='U':
            scatter.append((np.asarray(x).copy(),np.asarray(y).copy()))
        return original_scatter(self,x,y,*a,**k)
    def capture_quiver(self,*args,**kwargs):
        arrows.append(tuple(np.asarray(value).copy() for value in args[:4]))
        return original_quiver(self,*args,**kwargs)
    def capture_title(self,label,*a,**k):
        titles.append(label)
        return original_title(self,label,*a,**k)
    monkeypatch.setattr(Axes,'scatter',capture_scatter)
    monkeypatch.setattr(Axes,'quiver',capture_quiver)
    monkeypatch.setattr(Axes,'set_title',capture_title)
    direct=(np.arange(18,dtype=float)-9).reshape(1,2,3,3)
    advanced=direct**2+1
    panel=dict(direct=direct,r_of_g=advanced,condition=np.array([0]))
    result=dict(smoke=True,scientific_stop=False,arms={'A':[dict(split='test',nrmse=1.)]},identifiability={})
    report.plot_report(tmp_path,result,{},dict(initial=dict(L108=panel)),{},
        [dict(index=0,family='zero',K0=.44)])
    g,rg=report.pointwise_R_plot_arrays(panel)
    assert len(scatter)==3 and len(arrows)==1
    for component,(x,y) in enumerate(scatter):
        np.testing.assert_array_equal(x,g[:,component]);np.testing.assert_array_equal(y,rg[:,component])
    x,y,u,v=arrows[0]
    np.testing.assert_array_equal(x,g[:,0]);np.testing.assert_array_equal(y,g[:,1])
    np.testing.assert_array_equal(u,(rg-g)[:,0]);np.testing.assert_array_equal(v,(rg-g)[:,1])
    assert any('条件×路径等权描述统计' in title for title in titles)


def test_chinese_guard_selects_existing_font_before_layout_and_restores_settings(monkeypatch):
    from contextlib import contextmanager
    from matplotlib import font_manager
    font=Path(font_manager.findfont('DejaVu Sans'))
    before=list(report.plt.rcParams['font.family'])
    observed=[]
    monkeypatch.setattr(report,'select_chinese_font',lambda explicit=None:font)
    @contextmanager
    def existing_hook(explicit=None):
        observed.append(explicit)
        yield explicit
    monkeypatch.setattr(report,'chinese_figures',existing_hook)
    with report.chinese_render_guard() as selected:
        assert selected==font and observed==[font]
        assert report.plt.rcParams['font.family']==[font_manager.FontProperties(fname=str(font)).get_name()]
    assert list(report.plt.rcParams['font.family'])==before


def test_real_save_missing_CJK_glyph_fails_instead_of_silent_font_fallback(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from matplotlib import font_manager
    font=Path(font_manager.findfont('DejaVu Sans'))
    # Simulate an erroneous font selector/hook: the actual renderer warning at
    # savefig must still stop publication, not quietly produce tofu boxes.
    monkeypatch.setattr(report,'select_chinese_font',lambda explicit=None:font)
    monkeypatch.setattr(report,'chinese_figures',lambda *a,**k:nullcontext())
    fig=None
    try:
        with report.chinese_render_guard():
            fig,ax=report.plt.subplots()
            ax.set_title('中文缺字验证')
            with pytest.raises(UserWarning,match='Glyph'):
                fig.savefig(tmp_path/'must_not_silently_render.png')
    finally:
        if fig is not None:report.plt.close(fig)


def test_real_Chinese_font_layout_and_save_have_complete_CJK_coverage(tmp_path):
    from matplotlib.text import Text
    try:
        font=report.select_chinese_font()
    except RuntimeError:
        pytest.skip('Host has no existing CJK font; do not install one for a unit test')
    before=list(report.plt.rcParams['font.family'])
    fig=None
    path=tmp_path/'中文真实绘图检查.png'
    try:
        with report.chinese_render_guard(font):
            fig,ax=report.plt.subplots()
            ax.plot([-1,0,1],[1,0,1],label='有限扰动验证')
            ax.set_title('独立物理链与条件×路径等权描述统计')
            ax.set_xlabel('实际编码 g1');ax.set_ylabel('完整 R(g) 分量')
            ax.legend()
            fig.tight_layout()
            fig.savefig(path)
            visible=[artist for artist in fig.findobj(match=Text) if artist.get_visible() and artist.get_text()]
            assert visible and all(Path(artist.get_fontproperties().get_file()).resolve()==font.resolve()
                                   for artist in visible)
        assert path.is_file() and path.stat().st_size>1000
    finally:
        if fig is not None:report.plt.close(fig)
    assert list(report.plt.rcParams['font.family'])==before
