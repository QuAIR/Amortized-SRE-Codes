"""Render the two manuscript figures from their fully specified source arrays.

No smoothing, fitting, data selection, or new random sampling is performed.
Figure 1 source rows are analytic formula evaluations. Figure 2 first takes
integer bounds for each realization, then means and standard errors.
"""
from pathlib import Path
import hashlib
import json
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
    rows = np.loadtxt(OUT / 'Rz_figure_source.csv', delimiter=',', skiprows=1)
    fig, ax = plt.subplots(figsize=(3.5, 2.65), layout='constrained')
    ax.plot(rows[:, 1], rows[:, 2], color='#0072B2', label=r'$M_2^{\mathcal{A}}$ lower bound')
    ax.plot(rows[:, 1], rows[:, 3], '--', color='#0072B2', label=r'Stabilizer-input $M_2$')
    ax.plot(rows[:, 1], rows[:, 4], color='#D55E00', label='Amortized log robustness')
    ax.plot(rows[:, 1], rows[:, 5], '--', color='#D55E00', label='Amortized log extent')
    ax.set(xlabel=r'$\theta/\pi$', ylabel='Magic increase (bits)', xlim=(0, .5), ylim=(0, .55))
    ax.legend(fontsize=7); ax.grid(alpha=.18)
    fig.savefig(FIGS / 'Rz_revised.pdf', metadata=metadata)
    fig.savefig(FIGS / 'Rz_revised.png', dpi=200)
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
    fig.savefig(FIGS / 'heisenberg_revised.pdf', metadata=metadata)
    fig.savefig(FIGS / 'heisenberg_revised.png', dpi=200)
    plt.close(fig)
    paths = [OUT/'Rz_figure_source.csv', OUT/'heisenberg.csv', OUT/'heisenberg_disorder.json',
             OUT/'heisenberg_P2_W1.npy', OUT/'heisenberg_P2_W3.npy', OUT/'heisenberg_integer_bounds.npz']
    paths += [FIGS / name for name in ('Rz_revised.pdf', 'Rz_revised.png', 'heisenberg_revised.pdf', 'heisenberg_revised.png')]
    manifest = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'figure_source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Rendered both figures from recorded arrays; source and output hashes saved.')


if __name__ == '__main__':
    from paths import prepare_output
    prepare_output()
    render()
    if (OUT/'heisenberg4_integer_bounds.npz').exists():
        from render_heisenberg4 import render as render_four_site
        render_four_site()
