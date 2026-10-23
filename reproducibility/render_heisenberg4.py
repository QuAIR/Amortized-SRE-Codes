"""Render the restored single-panel N=4 zero-error/nullity comparison."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import ROOT
OUT=ROOT/'reproducibility/results'


def render():
    from paths import prepare_output
    prepare_output()
    stored=np.load(OUT/'heisenberg4_integer_bounds.npz');times=stored['times']
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.5,2.65),layout='constrained')
    for W,color,marker in ((.5,'#0072B2','o'),(1.,'#D55E00','s'),(2.,'#E69F00','^'),(3.,'#7B3294','d')):
        values=stored[f'W{W:g}_eps0'].astype(float)
        mean=values.mean(axis=0);se=values.std(axis=0,ddof=1)/np.sqrt(len(values))
        ax.plot(times,mean,color=color,label=fr'$W={W:g}$',lw=1.25,marker=marker,markersize=2.4,markevery=2)
        ax.fill_between(times,mean-se,mean+se,color=color,alpha=.12,linewidth=0)
    ax.set(xscale='log',xlim=(.1,500),xlabel=r'Evolution time $t$',ylim=(0,13),title=r'$N=4$, periodic boundary')
    ax.set_yticks((0,2,4,6,7,8,10,12))
    ax.grid(alpha=.18,which='major')
    ax.legend(fontsize=7,ncol=2,loc='lower right')
    ax.axhline(7,color='#444444',ls='--',lw=1.25)
    ax.text(.14,7.25,r'Nullity ceiling: $\nu(U)\leq7$',fontsize=7.3,color='#333333')
    ax.set_ylabel('Mean integer SRE lower bound')
    fig.savefig(ROOT/'Figs/heisenberg_N4_periodic.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(ROOT/'Figs/heisenberg_N4_periodic.png',dpi=250)
    plt.close(fig)
    names=['reproducibility/results/'+p for p in ('heisenberg4_fields.npz','heisenberg4_moments.npz','heisenberg4_integer_bounds.npz','heisenberg4.csv','heisenberg4_model.json','heisenberg4_bound_reduction.csv')]
    names+=['Figs/heisenberg_N4_periodic.pdf','Figs/heisenberg_N4_periodic.png','Figs/t-count_lower_bounds_0919.png']
    manifest={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    (OUT/'heisenberg4_figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Rendered the single-panel N=4 comparison from the saved integer arrays.')


if __name__=='__main__':
    from paths import prepare_output
    prepare_output()
    render()
