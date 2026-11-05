"""Leading-order dilute-vortex BKT phase portrait, not fitted QCR output."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from scripts.report_phi4_hopping_qcr import _setup_plot

KC = 2 / np.pi
SOURCE = 'https://www.nobelprize.org/uploads/2018/06/kosterlitz-lecture.pdf'


def beta(k, y):
    return -4*np.pi**3*k*k*y*y, (2-np.pi*k)*y


def invariant_primitive(k):
    # Integral of d(y^2)/dK, with F(Kc)=0. Valid for these truncated ODEs.
    u = (np.asarray(k)-KC)/KC
    return (np.log1p(u)-u/(1+u))/(2*np.pi**2)


def render(output, metadata=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    plt = _setup_plot(None)
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    kmin, kmax, ymax = .40, 1.05, .06
    kk, yy = np.meshgrid(np.linspace(kmin+.012, kmax-.012, 29),
                         np.linspace(.0018, ymax-.0018, 22))
    bk, by = beta(kk, yy)
    magnitude = np.hypot(bk, by)
    sep_k = np.linspace(KC, kmax, 1200)
    sep_y = np.sqrt(np.maximum(invariant_primitive(sep_k), 0))
    in_frame = sep_y <= ymax
    sep_k, sep_y = sep_k[in_frame], sep_y[in_frame]

    # Verify signs, fixed line, invariant conservation and the marginal endpoint.
    assert np.all(bk < 0)
    assert np.all(by[kk<KC] > 0) and np.all(by[kk>KC] < 0)
    assert all(np.all(v == 0) for v in beta(kk, np.zeros_like(kk)))
    bp, bq = beta(sep_k, sep_y)
    derivative = (np.pi*sep_k-2)/(2*np.pi**3*sep_k**2)
    tangency_error = float(np.max(abs(2*sep_y*bq-derivative*bp)))
    assert tangency_error < 1e-14
    traces = []; errors = []
    for k0, y0 in [(.85,.02),(.68,.025)]:
        def stop(_, z): return ymax-z[1]
        stop.terminal = True; stop.direction = -1
        sol = solve_ivp(lambda _,z: beta(*z), (0,40), (k0,y0),
                        rtol=1e-10,atol=1e-12,max_step=.1,events=stop)
        assert sol.success and np.all(np.diff(sol.y[0]) <= 1e-12)
        invariant = sol.y[1]**2-invariant_primitive(sol.y[0])
        errors.append(float(np.max(abs(invariant-invariant[0]))))
        traces.append(dict(K0=k0,y0=y0,final_K=float(sol.y[0,-1]),
                           final_y=float(sol.y[1,-1]),left_display=bool(len(sol.t_events[0]))))
    assert max(errors) < 1e-9
    assert traces[0]['final_K'] > KC and traces[0]['final_y'] < 1e-8
    assert traces[1]['left_display']

    fig = plt.figure(figsize=(11,8.3))
    ax = fig.add_axes([.095,.23,.85,.55])
    ax.set(xlim=(kmin,kmax),ylim=(-.0025,ymax),
           xlabel=r'当前尺度的无量纲有效刚度 $K_R(\ell)$',ylabel=r'涡旋逸度 $y(\ell)$')
    ax.set_xticks([.4,.5,KC,.8,.9,1.0],['0.4','0.5',r'$2/\pi$','0.8','0.9','1.0'])
    ax.set_yticks([0,.01,.02,.03,.04,.05,.06])
    ax.grid(alpha=.12)
    # Equal displayed lengths while retaining the beta-vector direction in data coordinates.
    fig.canvas.draw()
    width,height=ax.get_window_extent().width,ax.get_window_extent().height
    screen_norm=np.hypot(bk*width/(kmax-kmin),by*height/(ymax+.0025))
    length=min(width/29,height/22)*.62
    dk,dy=bk/screen_norm*length,by/screen_norm*length
    cmap=LinearSegmentedColormap.from_list('flow_gray',['#eeeeee','#333333'])
    q=ax.quiver(kk,yy,dk,dy,magnitude,cmap=cmap,norm=Normalize(0,float(magnitude.max())),
                angles='xy',scale_units='xy',scale=1,width=.0022,zorder=1)
    ax.plot(sep_k,sep_y,c='#bc3f4a',lw=2.4,zorder=3,label='临界分界轨道')
    ax.plot([KC,kmax],[0,0],c='#16816b',lw=4,zorder=4,label='固定点线：对涡旋扰动稳定段')
    ax.plot([kmin,KC],[0,0],c='#8d795a',lw=2.2,ls='--',zorder=3,
            label='固定点线：对涡旋扰动不稳定段')
    ax.scatter([KC],[0],c='#bc3f4a',s=68,zorder=5)
    ax.annotate('BKT 边际端点\n'+r'$(2/\pi,\,0)$',xy=(KC,0),xytext=(.49,.010),
                color='#7a2630',fontsize=11,arrowprops=dict(arrowstyle='->',color='#7a2630'),
                bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=3))
    ax.text(.435,.045,'无序相一侧\n涡旋增多，最终离开低逸度区',color='#93424a',fontsize=12,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=4))
    ax.text(.86,.0105,'低温临界相一侧\n流向 '+r'$y=0,\ K_R>2/\pi$',color='#116e5b',fontsize=12,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=4))
    # Emphasize forward RG orientation on the critical trajectory itself.
    j=np.searchsorted(sep_k,.84)
    ax.annotate('',xy=(sep_k[j-22],sep_y[j-22]),xytext=(sep_k[j+22],sep_y[j+22]),
                arrowprops=dict(arrowstyle='-|>',color='#bc3f4a',lw=2.4),zorder=5)
    fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.52,.815),
               ncol=3,fontsize=10,frameon=False)
    fig.suptitle('低逸度近似：二维 XY 的 BKT 理论 RG 相图',fontsize=20,y=.965)
    fig.text(.52,.905,r'$\frac{dK_R^{-1}}{d\ell}=4\pi^3y^2$'
             r'      $\frac{dy}{d\ell}=(2-\pi K_R)y$',ha='center',fontsize=18)
    fig.text(.52,.856,'箭头指向粗粒化尺度增大的方向；等长箭头越浅，局部流动越弱',ha='center',fontsize=11)
    cb=fig.colorbar(q,cax=fig.add_axes([.24,.112,.57,.025]),orientation='horizontal')
    cb.set_label(r'流动幅度 $\|\beta(K_R,y)\|_2$（上述坐标中的线性灰度）',fontsize=11)
    fig.text(.52,.047,'理论示意：省略高阶逸度项；未将原始温度或 QCR 潜变量标定到此平面。',
             ha='center',fontsize=10,color='#555555')
    coverage = None
    if metadata is not None:
        # This is a side note, deliberately NOT a rectangle in (K_R,y):
        # finite-L helicity and total vortex density do not calibrate y(ell).
        data = json.loads(Path(metadata).read_text())
        temperatures = sorted({row['temperature'] for row in data['chains']})
        bare = sorted({row['K'] for row in data['chains']})
        coverage = dict(metadata=str(Path(metadata).resolve()),
            metadata_sha256=hashlib.sha256(Path(metadata).read_bytes()).hexdigest(),
            temperatures=temperatures,bare_K_range=[bare[0],bare[-1]],
            lattice_size=data['lattice_size'],physical_chains=data['independent_physical_chains'],
            samples_per_chain=data['samples_per_chain'],
            fugacity_calibrated=False,theory_plane_coverage_known=False,
            rectangle_meaning='Dashed annotation outside phase plane; known sampling parameters only')
        ratio=11/15.4
        fig.set_size_inches(15.4,8.3)
        for axis in fig.axes:
            pos=axis.get_position();axis.set_position([pos.x0*ratio,pos.y0,pos.width*ratio,pos.height])
        for text in fig.texts:
            x,y=text.get_position();text.set_position((x*ratio,y))
        for legend in fig.legends:
            legend.set_bbox_to_anchor((.52*ratio,.815),transform=fig.transFigure)
        note=fig.add_axes([.738,.235,.248,.55]);note.set_axis_off()
        from matplotlib.patches import Rectangle
        note.add_patch(Rectangle((.01,.02),.98,.96,fill=False,ls='--',lw=1.8,
                                ec='#3c5e7a',transform=note.transAxes))
        note.text(.08,.90,'当前数据：已知采样范围',fontsize=14,color='#294b67',weight='bold')
        note.text(.08,.73,rf'$T/J\in[{temperatures[0]:g},\,{temperatures[-1]:g}]$'+'\n'+
                  rf'$K_{{\mathrm{{bare}}}}=J/T\in[{bare[0]:g},\,{bare[-1]:g}]$',fontsize=14,linespacing=1.8)
        note.text(.08,.57,f"{len(temperatures)} 个温度；{data['independent_physical_chains']} 条物理链\n"+
                  f"格点 L={data['lattice_size']}；每链 {data['samples_per_chain']} 个构型",fontsize=11,linespacing=1.6)
        note.text(.08,.37,'已保存刚度和涡旋密度\n尚未标定涡旋逸度 '+r'$y(T,\ell)$',fontsize=12,linespacing=1.6)
        note.text(.08,.16,'虚线框是采样范围说明\n不是 '+r'$(K_R,y)$'+' 平面的覆盖区域',
                  fontsize=12,color='#a33842',linespacing=1.6)
    for suffix in ('png','pdf'):fig.savefig(output/f'BKT_KR_y_phase_portrait.{suffix}',dpi=190)
    plt.close(fig)
    np.savez_compressed(output/'theory_arrays.npz',K_R=kk,y=yy,beta_K_R=bk,beta_y=by,
                        magnitude=magnitude,separatrix_K_R=sep_k,separatrix_y=sep_y)
    receipt=dict(kind='theory_illustration_not_experiment',source=SOURCE,
        equations=['d(1/K_R)/dell=4*pi^3*y^2','dy/dell=(2-pi*K_R)*y'],
        approximation='leading dilute-vortex equations; omit O(y^4) and O(y^3), respectively',
        K_critical_endpoint=KC,critical_endpoint_linear_eigenvalues=[0.,0.],
        fixed_line='y=0; transverse eigenvalue 2-pi*K_R, tangent eigenvalue zero',
        separatrix='y^2=log(K_R/Kc)/(2*pi^2)+(1/K_R-1/Kc)/pi^3, K_R>=Kc; exact only within the truncated ODEs',
        displayed_domain=dict(K_R=[kmin,kmax],y=[0,ymax]),
        no_bare_temperature_or_QCR_coordinate_matching=True,data_coverage=coverage,
        validation=dict(separatrix_tangency_error=tangency_error,invariant_errors=errors,example_integrations=traces),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(output.resolve()),validation=receipt['validation']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path)
    args=parser.parse_args();render(args.output,args.metadata)
