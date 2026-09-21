"""Universal linearized Ising RG, NOT an exact global nearest-neighbor map."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path(__file__).resolve().parents[1]/'results/ising2d_theoretical_local_flow'
    out.mkdir(parents=True, exist_ok=True)
    kc = np.log1p(np.sqrt(2))/2
    b = 2.
    lt, lh = b, b**(15/8)
    # Leading local identification of thermal/magnetic scaling coordinates
    # with K-Kc and B. This is not a global two-coupling decimation rule.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    ax = axes[0]
    ax.axvspan(-.2, 0, color='#387bb8', alpha=.07)
    ax.axvspan(0, .2, color='#c94c4c', alpha=.07)
    ax.plot([0, 0], [.2, kc], '--', color='.6', lw=1.4)
    ax.plot([0, 0], [kc, .68], color='#9d2798', lw=3, label='Coexistence line')
    ax.scatter(0, kc, marker='*', s=160, c='black', zorder=5, label='Critical endpoint')
    ax.text(-.15, .58, '$m<0$', fontsize=15, color='#28699e')
    ax.text(.085, .58, '$m>0$', fontsize=15, color='#a43b3b')
    ax.annotate('No phase boundary here', (0, .3), (-.185, .245), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='.4'))
    ax.set(xlim=(-.2, .2), ylim=(.2, .68), title='Thermodynamic phase diagram\n(not a global RG vector field)')
    ax.legend(loc='upper left', fontsize=8)

    bb, kk = np.meshgrid(np.linspace(-.008, .008, 17), kc+np.linspace(-.016, .016, 17))
    db, dk = (lh-1)*bb, (lt-1)*(kk-kc)
    # Common normalization in the actual plotted B,K coordinate units.
    norm = np.hypot(db, dk)
    valid = norm > 1e-12
    axes[1].quiver(bb[valid], kk[valid], .0012*db[valid]/norm[valid],
                   .0012*dk[valid]/norm[valid], angles='xy', scale_units='xy',
                   scale=1, width=.004, color='#29768b')
    axes[1].set_title('Local discrete RG: $R(K,B)-(K,B)$\nArrows normalized; coarse-graining factor $b=2$')
    trajectories = []
    for dt, field, color in [(-.003, -.0004, 'C0'), (-.003, .0004, 'C1'),
                              (.003, -.0004, 'C2'), (.003, .0004, 'C3'),
                              (-.003, 0, 'C4'), (.003, 0, 'C5')]:
        k = kc + dt*lt**np.arange(3)
        h = field*lh**np.arange(3)
        axes[2].plot(h, k, 'o-', c=color, ms=5)
        for i in range(2):
            axes[2].annotate('', (h[i+1], k[i+1]), (h[i], k[i]),
                             arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        trajectories.append({'K': k.tolist(), 'B': h.tolist()})
    axes[2].set_title('Two successive RG steps\nDots: k=0,1,2 from inside to outside')
    for ax in axes[1:]:
        ax.plot([0, 0], [kc, kc+.018], color='#9d2798', lw=2, alpha=.75)
        ax.plot([0, 0], [kc-.018, kc], '--', color='.55', lw=1)
        ax.axhline(kc, color='.7', ls=':', lw=1)
        ax.scatter(0, kc, c='black', marker='*', s=130, zorder=10)
        ax.set(xlim=(-.009, .009), ylim=(kc-.018, kc+.018))
        ax.ticklabel_format(axis='y', useOffset=False)
    for ax in axes:
        ax.set(xlabel='$B=\\beta h$', ylabel='$K=\\beta J$')
        ax.grid(alpha=.12)
    fig.suptitle('2D Ising: two relevant directions at the critical fixed point', fontsize=15)
    fig.text(.5, .015,
             r'Local scaling chart: $K\!\prime-K_c\simeq 2(K-K_c)$,  $B\!\prime\simeq 2^{15/8}B$. '
             'Not an exact globally closed map of microscopic K,B.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .92))
    fig.savefig(out/'theoretical_KB_flow.png', dpi=190)
    plt.close(fig)
    (out/'theory.json').write_text(json.dumps(dict(Kc=kc, block_factor=b,
        thermal_eigenvalue=lt, magnetic_eigenvalue=lh, trajectories=trajectories,
        assumption='Leading-order scaling-coordinate identification with K-Kc,B; not global microscopic RG.',
        source='https://academic.oup.com/book/8876/chapter/155113909'), indent=2)+'\n')


if __name__ == '__main__':
    main()
