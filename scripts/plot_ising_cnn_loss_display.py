"""JSON-only display correction; retain the original SHA-verified remote report.

The shared phi4 plot helper used two continuous-field legend labels. This
additional Ising figure labels the same saved values as histogram MI/color NLL.
No checkpoint, dataset, model, optimizer, or statistical resampling is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.report_phi4_hopping_qcr import _setup_plot


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve()
    status=json.loads((root/'status.json').read_text())
    assert status['status']=='complete' and status['workflow_complete'] and status['returncode']==0
    for item in status['public_files']:
        path=root/item['path']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    rows=json.loads((root/'history.json').read_text())
    assert rows[-1]['step']==status['actual_steps']==850
    output=root/'display'
    output.mkdir(exist_ok=False)
    plt=_setup_plot(None)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    groups=[
        [('q_total','Q 总目标'),('q_reconstruction','局部 motif 概率重构'),
         ('q_subset_micro','微观子集一致性'),('q_subset_coarse','粗尺度子集一致性')],
        [('c_total','C 总目标'),('c_information','经验直方图互信息 MI'),
         ('c_variance','系综方差下限'),('c_uniform','均匀块保色 NLL')],
        [('r_total','R 全起点锚定目标'),('closure_one_step','一步闭合诊断'),
         ('closure_rollout','微观起点多步闭合诊断')]]
    for ax,group,title in zip(axes,groups,['表示学习 Q','二值粗粒化 C','闭环映射 R']):
        for key,label in group:
            ax.plot([row['step'] for row in rows],[row[key] for row in rows],label=label)
        ax.set(title=title,xlabel='更新次数',ylabel='损失或诊断量')
        ax.set_yscale('symlog',linthresh=1e-5)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[1].set_yticks([-1.,0.,.001,.1,1.],['−1','0',r'$10^{-3}$',r'$10^{-1}$','1'])
    axes[1].minorticks_off()
    fig.suptitle('二维 Ising，B=0，逐构型 CNN；固定 850 步\n总目标含权重，分项显示未加权值；C = −2 MI + 2 方差下限 + 保色 NLL',fontsize=11)
    figure=output/'loss_curves.png'
    fig.savefig(figure,dpi=160)
    plt.close(fig)
    receipt=dict(official_public_files_verified=len(status['public_files']),
        official_files_unmodified=True,only_saved_JSON_read=True,
        change='Correct continuous-field legend labels to binary histogram MI and uniform-block color NLL; unclutter C-axis ticks.',
        history_sha256=sha(root/'history.json'),figure_sha256=sha(figure),
        script_sha256=sha(Path(__file__)),thermal_nu=None)
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__': main()
