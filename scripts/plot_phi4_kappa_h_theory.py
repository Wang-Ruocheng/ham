"""Theoretical phase topology and chosen grid, not learned or measured data."""
from pathlib import Path
import argparse
import numpy as np

def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from scripts.report_ising_zh import select_chinese_font
    from experiments.phi4_kappa_h_cnn import KAPPA_VALUES,H_VALUES
    font=FontProperties(fname=str(select_chinese_font()))
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
    critical=.680601
    fig,ax=plt.subplots(figsize=(10,6))
    fig.subplots_adjust(bottom=.22,top=.91,left=.11,right=.98)
    ax.axhspan(0,.012,color='#f8d9d5',alpha=.65)
    ax.axhspan(-.012,0,color='#dce8f8',alpha=.75)
    ax.plot([critical,.81],[0,0],color='#8d243b',lw=4,label='一级共存线（热力学极限）',zorder=3)
    x,y=np.meshgrid(KAPPA_VALUES,H_VALUES)
    ax.scatter(x,y,facecolors='none',edgecolors='#65717b',s=30,lw=.8,label='计划采样参数点',zorder=4)
    ax.scatter([critical],[0],s=95,c='black',zorder=5,label='连续临界端点')
    ax.text(.60,.006,'正磁化偏向：$h>0$',ha='center',fontsize=13)
    ax.text(.60,-.006,'负磁化偏向：$h<0$',ha='center',fontsize=13)
    ax.text(.60,.0006,'零场无序区',ha='center',fontsize=11)
    ax.annotate(r'$\kappa_c\simeq0.680601,\ h=0$',xy=(critical,0),xytext=(.697,-.004),
        arrowprops={'arrowstyle':'->','color':'black'},fontsize=11)
    ax.text(.755,.0012,'零场有序相共存',ha='center',fontsize=11,color='#8d243b')
    ax.set(xlim=(.55,.81),ylim=(-.012,.012),xlabel='近邻耦合 $\kappa$',ylabel='外场 $h$',
        title='二维标量场：固定 $\lambda=1$ 的理论相图与本轮采样范围')
    ax.legend(loc='upper left',fontsize=9)
    ax.grid(alpha=.15)
    fig.text(.5,.025,'上、下色块仅表示磁化偏向；左侧可连续绕行。有限尺寸会抹圆奇异性。\n'
        '临界值为文献无限体积数值参考；此图不是有限尺寸测量，也不是裸耦合 RG 流场。',
        ha='center',fontsize=10)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('No historical figure overwrite')
    fig.savefig(output,dpi=170,bbox_inches='tight');plt.close(fig)
    return str(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(render(p.parse_args().output))
