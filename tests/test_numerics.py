"""Independent values, adversarial inputs, rounding boundaries and data identity."""
import json
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'reproducibility'))
from verify_revision import moments, choi, qft, lower, ccrz, cswap, paulis
from certify_qft import upper_qft, certified_lower, positive_sum_up
from integrity import verify_manifest
import paths


class MomentTests(unittest.TestCase):
    def test_stabilizers_and_known_magic(self):
        bell = np.array([1, 0, 0, 1])/np.sqrt(2)
        np.testing.assert_allclose(moments(bell, (1, 1.5, 2, 3), batch=3), 1, atol=2e-14)
        t = np.diag([1, np.exp(1j*np.pi/4)])
        ccz = np.diag([1]*7+[-1])
        self.assertAlmostEqual(moments(t @ np.ones(2)/np.sqrt(2))[0], 3/4)
        self.assertAlmostEqual(moments(ccz @ np.ones(8)/np.sqrt(8))[0], 11/32)
        self.assertAlmostEqual(moments(choi(t))[0], 3/4)
        self.assertAlmostEqual(moments(choi(ccz))[0], 11/32)

    def test_walsh_against_direct_noninteger_orders(self):
        rng = np.random.default_rng(7231)
        orders = (1, 1.5, 2, 3)
        for n in (1, 2, 3, 4):
            psi = rng.normal(size=2**n)+1j*rng.normal(size=2**n)
            psi /= np.linalg.norm(psi)
            expected = [sum(abs(np.vdot(psi, p @ psi))**(2*a) for p in paulis(n))/2**n for a in orders]
            np.testing.assert_allclose(moments(psi, orders, batch=3), expected, rtol=2e-13, atol=1e-14)

    def test_clifford_invariance_and_tensor_product(self):
        psi = np.array([1, 2j, -3, .7+.2j]); psi /= np.linalg.norm(psi)
        # A CNOT permutes the last two amplitudes in the |10>, |11> block.
        np.testing.assert_allclose(moments(psi), moments(psi[[0,1,3,2]]), atol=1e-14)
        phi = np.array([1, np.exp(.37j)])/np.sqrt(2)
        np.testing.assert_allclose(moments(np.kron(psi,phi)), moments(psi)*moments(phi), atol=1e-14)

    def test_controlled_gate_analytic_polynomials(self):
        for theta in (.173, .9, np.pi/2, np.pi):
            c = math.cos(theta)
            self.assertAlmostEqual(moments(choi(ccrz(theta)))[0], (151+84*c+20*c*c+c**4)/256)
        for s in (.17, .5, .75, 1):
            c = math.cos(np.pi*s)
            self.assertAlmostEqual(moments(choi(cswap(s)))[0], (109+84*c+56*c*c+7*c**4)/256)

    def test_invalid_state_and_orders(self):
        for psi in ([], [1,0,0], [1,1], [np.nan,0], np.eye(2)/2):
            with self.subTest(psi=psi), self.assertRaises(ValueError): moments(psi)
        for orders in ([], [0], [-1], [np.nan], [[2]]):
            with self.subTest(orders=orders), self.assertRaises(ValueError): moments([1,0], orders)
        for batch in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): moments([1,0], batch=batch)

    def test_invalid_unitaries(self):
        for u in ([], np.ones((2,3)), np.eye(3), np.eye(2)*2, np.full((2,2),np.nan)):
            with self.subTest(u=u), self.assertRaises(ValueError): choi(u)
        for n in (-1, 1.5, True):
            with self.assertRaises(ValueError): qft(n)


class BoundTests(unittest.TestCase):
    def test_integer_boundaries(self):
        self.assertEqual(certified_lower(1), 0)
        for k in range(1, 30):
            threshold = .75**k  # exactly representable for these k
            self.assertEqual(certified_lower(threshold), k)
            self.assertEqual(certified_lower(np.nextafter(threshold, 0)), k+1)
            self.assertEqual(certified_lower(np.nextafter(threshold, 1)), k)
        self.assertEqual(certified_lower(.75, .02), 0)

    def test_invalid_bound_inputs(self):
        for p in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError): lower(p)
            with self.assertRaises(ValueError): certified_lower(p)
        for e in (-1e-5, 1.1, np.nan):
            with self.assertRaises(ValueError): lower(.5,e)
            with self.assertRaises(ValueError): certified_lower(.5,e)
        with self.assertRaises(ValueError): lower(2)
        with self.assertRaises(ValueError): lower(.5,adaptive='no')

    def test_prescribed_tolerance_monotonicity(self):
        self.assertEqual(lower(3/4), 1)
        self.assertEqual(lower(11/32), 4)
        for p in np.linspace(.001,1,101):
            bounds = [lower(p,e) for e in (0,1e-8,1e-6,1e-4,.01,1)]
            self.assertEqual(bounds, sorted(bounds,reverse=True))
            self.assertGreaterEqual(lower(p,1e-4,False),lower(p,1e-4,True))

    def test_qft_interval_and_known_exact_value(self):
        self.assertAlmostEqual(moments(choi(qft(2)))[0], 7/16)
        for n, bound in ((2,3),(3,6),(4,8)):
            upper = upper_qft(n, batch=3)
            numeric = moments(choi(qft(n)),(2,))[0]
            self.assertGreaterEqual(upper, numeric-2e-15)
            self.assertLess(upper-numeric, 1e-10)
            self.assertEqual(certified_lower(upper,1e-6),bound)
        with self.assertRaises(ValueError): positive_sum_up([-1])
        with self.assertRaises(ValueError): upper_qft(2,batch=0)


class ReleasedDataTests(unittest.TestCase):
    def test_output_directory_guards(self):
        for root in (REPO, REPO.parent, REPO/'reproducibility/results/nested', REPO/'Figs'):
            with self.subTest(root=root), patch.object(paths,'ROOT',root), self.assertRaises(SystemExit):
                paths.prepare_output()

    def test_redirected_output_subdirectory(self):
        root = REPO/'.tmp/safety-test'
        original_resolve = Path.resolve
        for relative in ('Figs','reproducibility/results'):
            def redirected(path, *args, **kwargs):
                return REPO/relative if path == root/relative else original_resolve(path,*args,**kwargs)
            with self.subTest(relative=relative), patch.object(paths,'ROOT',root), patch.object(Path,'resolve',redirected), self.assertRaises(SystemExit):
                paths.prepare_output()

    def test_individual_output_file_link(self):
        scratch = REPO/'.tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            root = Path(name).resolve()
            self.assertTrue(root.is_relative_to(scratch.resolve()))
            (root/'Figs').mkdir()
            target = root/'Figs/probe.png'
            target.write_bytes(b'probe')
            original = Path.is_symlink
            def linked(path):
                return path == target or original(path)
            with patch.object(paths,'ROOT',root), patch.object(Path,'is_symlink',linked), self.assertRaises(SystemExit):
                paths.prepare_output()

    def test_release_hashes(self):
        self.assertGreater(verify_manifest(), 25)

    def test_embedded_provenance_hashes(self):
        out = REPO/'reproducibility/results'
        for name in ('figure_source_manifest.json','heisenberg4_figure_manifest.json'):
            record = json.loads((out/name).read_text(encoding='utf-8'))
            for path, expected in record.items():
                self.assertEqual(hashlib.sha256((REPO/path).read_bytes()).hexdigest(),expected,path)
        records = [json.loads((out/name).read_text(encoding='utf-8'))['sha256'] for name in (
            'heisenberg4_independent_audit.json','heisenberg4_finite_error_audit.json')]
        records.append(json.loads((out/'figure_data_integrity_20261019.json').read_text(encoding='utf-8'))['figure_2']['raw_data_sha256'])
        for record in records:
            for path,expected in record.items():
                self.assertEqual(hashlib.sha256((out/path).read_bytes()).hexdigest(),expected,path)

    def test_full_prescribed_tolerance_arrays(self):
        out = REPO / 'reproducibility/results'
        with np.load(out/'heisenberg4_moments.npz') as raw, np.load(out/'heisenberg4_integer_bounds.npz') as saved:
            count = 0
            for w in (.5,1,2,3):
                p = raw[f'W{w:g}']
                for eps in (0,1e-6,1e-4):
                    expected = np.array([[lower(v,eps) for v in row] for row in p])
                    np.testing.assert_array_equal(expected, saved[f'W{w:g}_eps{eps:g}'])
                count += np.count_nonzero(saved[f'W{w:g}_eps0']-saved[f'W{w:g}_eps0.0001'])
            self.assertEqual(count,3574)


if __name__ == '__main__': unittest.main()
