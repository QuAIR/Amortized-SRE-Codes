# Reproducing the revised results

Use Python 3.12 and the exact versions in `requirements.txt`. Run all commands
from the repository root. MATLAB is independent of the Python environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r reproducibility/requirements.txt
.\.venv\Scripts\python.exe reproducibility/run_checks.py
.\.venv\Scripts\python.exe reproducibility/run_checks.py --full
```

On Linux/macOS, use `python3.12 -m venv .venv` and `.venv/bin/python` instead.
The full audit takes several minutes. It reconstructs every saved evolution
independently and writes its reports and redrawn figures to `.tmp/audit/`.
Released data and figures are checked against `data_manifest.json` before and
after the audit. This checksum manifest establishes file identity; mathematical
and numerical checks are separate.

## Files and evidence

| Files / script | Role |
| --- | --- |
| `results/Rz_figure_source.csv`, `Figs/Rz_revised.*` | Figure 1: 401 analytic curve evaluations. The blue amortized curve is an attainable equatorial-input lower bound. |
| `results/heisenberg4_*`, `Figs/heisenberg_N4_periodic.*` | Figure 2: periodic four-site model, per-realization target moments and integer witnesses. |
| `results/heisenberg_P2_W*.npy`, `heisenberg_disorder.json`, `heisenberg_integer_bounds.npz`, `heisenberg.csv`, `Figs/heisenberg_revised.*` | Supplementary Figure 3: open three-site model. |
| `verify_revision.py` | Walsh/direct-Pauli checks, exact CCZ blocks and four-copy PSD checks, random reference diagnostics; generator for QFT, controlled gates and the three-site example. |
| `symbolic_gates.py`, `results/exact_gate_polynomials.json` | Exact Laurent polynomials for CCRz and controlled fractional SWAP. |
| `certify_qft.py`, `results/qft_certified.json` | Outward-rounded interval upper bounds and rational-threshold certificates for QFT Choi witnesses. |
| `audit_proofs.py` | Independent exact CCZ identities and PSD calculations, plus continuity/ensemble/reference diagnostics. |
| `restore_heisenberg4.py` | Four-site generator: Kronecker Hamiltonian, eigendecomposition, Walsh moments. |
| `verify_heisenberg4.py` | Independent spin-basis Hamiltonian, `scipy.expm`, direct PTM traces for all 40,000 four-site points. |
| `verify_figure_data.py` | Independent reconstruction of all 24,400 three-site points and the analytic Rz curves. |
| `verify_finite_error_panel.py` | Historical filename: checks prescribed-tolerance sensitivity arrays; the current Figure 2 is a single zero-error panel. |
| `render_figures.py`, `render_heisenberg4.py` | Render saved arrays without sampling, smoothing or fitting. |

Paths in the first column are relative to `reproducibility/`, except `Figs/`,
which is at the repository root. The released JSON reports record the original
2026-10-18 to 2026-10-22 calculations; new audit reports are written separately.
The historical `figure_2` field in the dated three-site audit refers to what is
now supplementary Figure 3. `Figs/t-count_lower_bounds_0919.png` is the unchanged
author-supplied historical image, retained for provenance rather than as a
source of numerical values.

## Model and error conventions

All logarithms are base two. For a pure N-qubit state,

`P_alpha = 2^(-N) sum_P |<psi|P|psi>|^(2 alpha)` and
`M_alpha = log2(P_alpha)/(1-alpha)` for `alpha != 1`.

Choi vectors are normalized vectorizations `vec(U)/sqrt(d)`. The standard QFT
uses matrix indices `0,...,d-1`. Rz(theta) is `exp(-i theta Z/2)`. The first
qubit controls fractional SWAP; its antisymmetric eigenvalue is `exp(i*pi*s)`.

The synthesis error epsilon is half the diamond norm. The adaptive witness is

`ceil(-log2(min(1, P2 + 16*epsilon))/log2(4/3))`.

It concerns deterministic finite stabilizer protocols with stabilizer work
qubits and measurements, counting the maximum T-state consumption over
branches. The unitary-implementation correction is `4*epsilon`. These formulas
use ideal-target moments and a prescribed tolerance; they do not simulate an
approximate circuit or implementation noise. A mixed-state Pauli polynomial
differs from the mixed-state roof used in the theorem.

`lower()` uses a `1e-11` downward guard before the integer ceiling. Its outputs
are binary64 numerical witnesses, not certified enclosures. QFT certification
instead encloses roots of unity at 60 decimal digits, propagates outward
binary64 intervals with `nextafter`, bounds positive summations, and compares
to exact rational `(3/4)^k` thresholds without logarithms. The certificate
assumes standard IEEE-754 arithmetic and mpmath's interval transcendental
enclosures. QFT n=3,4,5,6 gives 6,8,9,11 at epsilon=1e-6, and 6,7,9,11 at
epsilon=1e-4.

The Heisenberg Hamiltonian is

`H = sum_<j,k> (X_j X_k + Y_j Y_k + 0.2 Z_j Z_k) + sum_j h_j Z_j`.

For Figure 2, use periodic N=4, W=0.5,1,2,3, 200 samples per W,
`geomspace(0.1,500,50)`, and PCG64 seed 20261021 with successive W draws.
Each field is independently uniform in [-W,W]. These are fresh realizations
of the author-confirmed original model; the historical raw samples/grid were
unavailable. For supplementary Figure 3, use open N=3, W=1,3, 200 samples per W,
`linspace(0,3,61)` and successive PCG64 draws with seed 20261018.

Each integer witness is computed before averaging; shading is one standard
error over disorder. For N=4, parity `Z tensor 4` and identity give at least
two exact Pauli mappings, hence unitary nullity `nu(U) <= 7`. The dashed line
is a ceiling on that zero-error witness. All four sample means exceed seven
at the sampled times t>=10; individual realizations need not. At epsilon=1e-4,
3,574 of 40,000 integer entries decrease by one; the remaining entries agree
with the zero-error ones. `heisenberg4_bound_reduction.csv` stores optional
paired-difference statistics and is not plotted in the current manuscript.

The complete independent binary64 checks originally found maximum moment
differences of 2.35e-13 (N=4) and 5.11e-15 (N=3). Their scope is numerical
cross-validation, not pointwise interval certification or asymptotic scaling.
Random ancillary-state tests are finite diagnostics; the dimension-uniform
arguments are analytic proofs in the paper.

## Regeneration

Generation requires an explicitly separate output directory. First copy the
released inputs there, then regenerate and check them:

```powershell
$env:SRE_OUTPUT_ROOT = Join-Path (Get-Location) '.tmp/reproduced'
.\.venv\Scripts\python.exe reproducibility/paths.py
.\.venv\Scripts\python.exe reproducibility/verify_revision.py
.\.venv\Scripts\python.exe reproducibility/restore_heisenberg4.py
.\.venv\Scripts\python.exe reproducibility/verify_figure_data.py
.\.venv\Scripts\python.exe reproducibility/verify_heisenberg4.py
.\.venv\Scripts\python.exe reproducibility/verify_finite_error_panel.py
.\.venv\Scripts\python.exe reproducibility/render_figures.py
Remove-Item Env:SRE_OUTPUT_ROOT
```

For POSIX shells, use `export SRE_OUTPUT_ROOT="$PWD/.tmp/reproduced"` and
`unset SRE_OUTPUT_ROOT`. Floating moments can differ slightly across BLAS
implementations; compare moments numerically and integer arrays exactly.
Figures can differ in font rendering across platforms. The saved arrays and
formulas, rather than pixel identity across platforms, specify the results.
