"""Check prescribed half-diamond tolerances applied to ideal-target moments."""
import json
import math
import numpy as np
from paths import ROOT, prepare_output

OUT = ROOT/'reproducibility/results'


def verify():
    prepare_output()
    counts = {}
    checked = total = changed = 0
    with np.load(OUT/'heisenberg4_moments.npz') as raw, np.load(OUT/'heisenberg4_integer_bounds.npz') as saved:
        assert np.array_equal(raw['times'],saved['times'])
        for w in (.5,1.,2.,3.):
            p = raw[f'W{w:g}']
            assert p.shape == (200,50) and np.all(np.isfinite(p)) and np.all((p>0)&(p<=1))
            for epsilon in (0.,1e-6,1e-4):
                expected = np.maximum(0,np.ceil(-np.log2(np.minimum(1.,p+16*epsilon))/math.log2(4/3)-1e-11)).astype(int)
                assert np.array_equal(expected,saved[f'W{w:g}_eps{epsilon:g}'])
                checked += p.size
            difference = saved[f'W{w:g}_eps0'].astype(int)-saved[f'W{w:g}_eps0.0001'].astype(int)
            assert set(np.unique(difference)) <= {0,1}
            counts[f'{w:g}'] = int(np.count_nonzero(difference))
            total += difference.size
            changed += counts[f'{w:g}']
    assert checked == 120000 and total == 40000 and changed == 3574
    record = {'epsilon':1e-4,'moment_correction':.0016,
              'integer_entries_checked':checked,'paired_entries':total,
              'decreased_by_one':changed,'unchanged':total-changed,
              'decreased_by_one_by_W':counts}
    (OUT/'tolerance_checks.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'Tolerance checks passed: {checked} integer witnesses; {changed} paired entries decrease by one.')
    return record


if __name__ == '__main__': verify()
