"""Independent, complete checks of the saved quantitative figure data.

No fields are resampled and no saved moments are overwritten. The Heisenberg
Hamiltonian is built from spin-basis exchange rules, evolved by scipy.expm,
and checked with direct Pauli-transfer traces rather than the original
eigendecomposition plus Walsh implementation. These are floating-point
cross-checks, not interval certificates of every time-series value.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import os

# Small matrix batches are faster and more reproducible with one BLAS thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy
from scipy.linalg import expm

from paths import ROOT
OUT = ROOT / 'reproducibility' / 'results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paulis_three():
    one = [np.eye(2), np.array([[0, 1], [1, 0]]),
           np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    return np.array([np.kron(np.kron(a, b), c)
                     for a, b, c in itertools.product(one, repeat=3)])


def spin_hamiltonian(fields):
    h = np.zeros((8, 8), complex)
    for j in range(8):
        bits = np.array([(j >> k) & 1 for k in (2, 1, 0)])
        z = 1 - 2 * bits
        h[j, j] = np.dot(fields, z) + .2 * (z[0] * z[1] + z[1] * z[2])
        for site in (0, 1):
            if bits[site] != bits[site + 1]:
                mask = (1 << (2 - site)) | (1 << (1 - site))
                h[j ^ mask, j] += 2  # XX+YY exchanges opposite spins.
    return h


def check_dynamics():
    paths = [OUT / 'heisenberg_disorder.json', OUT / 'heisenberg_integer_bounds.npz',
             OUT / 'heisenberg_P2_W1.npy', OUT / 'heisenberg_P2_W3.npy',
             OUT / 'heisenberg.csv']
    before = {p.name: digest(p) for p in paths}
    fields = json.loads(paths[0].read_text(encoding='utf-8'))
    saved = np.load(paths[1])
    times = saved['times']
    assert np.array_equal(times, np.linspace(0, 3, 61))
    stats = list(csv.DictReader((OUT / 'heisenberg.csv').open(encoding='utf-8')))
    rng = np.random.default_rng(20261018)
    basis = paulis_three()
    flat_basis = basis.reshape(64, 64)
    max_moment_error = 0.
    max_unitarity_error = 0.
    checked = 0
    for W in (1., 3.):
        hs = np.array(fields[str(W)])
        assert np.array_equal(hs, rng.uniform(-W, W, (200, 3)))
        old = np.load(OUT / f'heisenberg_P2_W{W:g}.npy')
        recomputed = []
        for row, field in enumerate(hs):
            h = spin_hamiltonian(field)
            u = expm(-1j * times[:, None, None] * h)
            adjoint = u.conj().swapaxes(-1, -2)
            max_unitarity_error = max(max_unitarity_error,
                float(np.max(np.abs(u @ adjoint - np.eye(8)))))
            rotated = (u[:, None] @ basis[None]) @ adjoint[:, None]
            transfer = rotated.reshape(61, 64, 64) @ flat_basis.conj().T / 8
            assert np.max(np.abs(transfer.imag)) < 2e-13
            p2 = np.sum(transfer.real ** 4, axis=(1, 2)) / 64
            max_moment_error = max(max_moment_error,
                                   float(np.max(np.abs(p2 - old[row]))))
            recomputed.append(p2)
            checked += 61
            if (row + 1) % 50 == 0:
                print(f'Independent evolution: W={W:g}, {row+1}/200 field vectors', flush=True)
        recomputed = np.array(recomputed)
        assert max_moment_error < 2e-11
        for epsilon in (0., 1e-6, 1e-4):
            moment = np.minimum(1., recomputed + 16 * epsilon)
            integers = np.maximum(0, np.ceil(-np.log2(moment) / math.log2(4/3) - 1e-11)).astype(int)
            original_integers = saved[f'W{W:g}_eps{epsilon:g}']
            assert np.array_equal(integers, original_integers)
            mean = original_integers.mean(axis=0)
            sem = original_integers.std(axis=0, ddof=1) / np.sqrt(200)
            rows = [r for r in stats if float(r['W']) == W and float(r['epsilon']) == epsilon]
            assert len(rows) == 61
            assert np.allclose([float(r['mean_bound']) for r in rows], mean, rtol=0, atol=1e-14)
            assert np.allclose([float(r['standard_error']) for r in rows], sem, rtol=0, atol=1e-14)
    assert before == {p.name: digest(p) for p in paths}
    return {'evolutions_checked': checked, 'field_vectors': 400,
            'max_absolute_moment_difference': max_moment_error,
            'max_unitarity_residual': max_unitarity_error,
            'all_73200_integer_entries_reproduced': True,
            'all_366_mean_and_standard_error_rows_match': True,
            'raw_data_sha256': before,
            'qualification': 'Numerical simulation of the stated three-site model; not experimental measurements or a general scaling result. Cross-checked in binary64, not interval-certified pointwise.'}


def rotation_data():
    from paths import prepare_output
    prepare_output()
    theta = np.linspace(0, np.pi / 2, 401)
    a = 49 - np.cos(4 * theta)
    amortized = np.log2((a + np.sqrt(np.maximum(0, a*a - 48**2))) / 48)
    strict = -np.log2((7 + np.cos(4 * theta)) / 8)
    robustness = np.log2(np.cos(theta) + np.sin(theta))
    extent = np.log2(1 + (np.sqrt(2) - 1) * (np.cos(theta) + np.sin(theta) - 1))
    # Independent stabilizer-decomposition and dual certificates for extent.
    coefficient_sum = np.sqrt(2) * (np.sin(np.pi/4 - theta/2) + np.sin(theta/2))
    dual_value = np.cos(theta/2 - np.pi/8)**2 / np.cos(np.pi/8)**2
    assert np.max(np.abs(2**extent - coefficient_sum**2)) < 2e-15
    assert np.max(np.abs(2**extent - dual_value)) < 2e-15
    # A dense phase check must sit below the analytic equatorial maximum.
    phases = np.linspace(0, 2*np.pi, 16384, endpoint=False)
    input_moments = (7 + np.cos(4 * phases)) / 8
    largest_gap = 0.
    for t, exact in zip(theta, amortized):
        ratio = input_moments / ((7 + np.cos(4 * (phases + t))) / 8)
        value = math.log2(float(ratio.max()))
        assert value <= exact + 2e-14
        largest_gap = max(largest_gap, exact - value)
    assert largest_gap < 2e-7
    data = np.column_stack([theta, theta/np.pi, amortized, strict, robustness, extent])
    path = OUT / 'Rz_figure_source.csv'
    np.savetxt(path, data, delimiter=',',
               header='theta,theta_over_pi,amortized_2SRE_lower_bound,strict_2SRE,amortized_log_robustness,amortized_log_extent', comments='')
    return {'points': 401, 'source_csv': path.name,
            'maximum_dense_phase_search_gap': largest_gap,
            'extent_primal_dual_agree': True,
            'qualification': 'Analytic curves; the solid blue curve is an achievable equatorial-input lower bound, not a globally optimized amortized capacity.'}


if __name__ == '__main__':
    from paths import prepare_output
    prepare_output()
    report = {'date': '2026-10-19', 'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
              'figure_1': rotation_data(), 'figure_2': check_dynamics(),
              'no_resampling_or_raw_data_replacement': True}
    (OUT / 'figure_data_integrity_20261019.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
