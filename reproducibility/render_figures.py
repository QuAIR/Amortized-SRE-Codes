"""Plot Rz curves and three-site Heisenberg witnesses from saved arrays."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import ROOT
OUT = ROOT / 'reproducibility' / 'results'
FIGS = ROOT / 'Figs'


def render():
    from paths import prepare_output
    prepare_output()
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    metadata = {'CreationDate': None, 'ModDate': None}
    rows = np.loadtxt(OUT / 'rz_curves.csv', delimiter=',', skiprows=1)
    fig, ax = plt.subplots(figsize=(3.5, 2.65), layout='constrained')
    ax.plot(rows[:, 1], rows[:, 2], color='#0072B2', label=r'$M_2^{\mathcal{A}}$ lower bound')
    ax.plot(rows[:, 1], rows[:, 3], '--', color='#0072B2', label=r'Stabilizer-input $M_2$')
    ax.plot(rows[:, 1], rows[:, 4], color='#D55E00', label='Amortized log robustness')
    ax.plot(rows[:, 1], rows[:, 5], '--', color='#D55E00', label='Amortized log extent')
    ax.set(xlabel=r'$\theta/\pi$', ylabel='Magic increase (bits)', xlim=(0, .5), ylim=(0, .55))
    ax.legend(fontsize=7); ax.grid(alpha=.18)
    fig.savefig(FIGS / 'rz_magic.pdf', metadata=metadata)
    fig.savefig(FIGS / 'rz_magic.png', dpi=200)
    plt.close(fig)

    stored = np.load(OUT / 'heisenberg_integer_bounds.npz')
    times = stored['times']
    fig, ax = plt.subplots(figsize=(3.5, 2.65), layout='constrained')
    for w, color in [(1., '#0072B2'), (3., '#D55E00')]:
        for e, style in [(0, '--'), (1e-6, '-'), (1e-4, ':')]:
            samples = stored[f'W{w:g}_eps{e:g}']
            mean = samples.mean(axis=0)
            error = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
            eps_label = {0: '0', 1e-6: r'10^{-6}', 1e-4: r'10^{-4}'}[e]
            label = fr'$W={w:g},\ \epsilon={eps_label}$'
            ax.plot(times, mean, style, color=color, label=label, lw=1.4)
            if e == 1e-6:
                ax.fill_between(times, mean-error, mean+error, color=color, alpha=.18, linewidth=0)
    ax.set(xlabel='Evolution time', ylabel='Mean integer lower bound', ylim=(0, 9))
    ax.legend(fontsize=6.5, ncol=2, loc='lower right'); ax.grid(alpha=.18)
    fig.savefig(FIGS / 'heisenberg3.pdf', metadata=metadata)
    fig.savefig(FIGS / 'heisenberg3.png', dpi=200)
    plt.close(fig)
    print('Rendered Rz and three-site Heisenberg figures.')


if __name__ == '__main__':
    from paths import prepare_output
    prepare_output()
    render()
    if (OUT/'heisenberg4_integer_bounds.npz').exists():
        from render_heisenberg4 import render as render_four_site
        render_four_site()
