# Reproducibility

Use Python 3.12 and the pinned packages in `requirements.txt`. Run commands
from the repository root.

```sh
python -m pip install -r reproducibility/requirements.txt
python reproducibility/run_checks.py
python reproducibility/run_checks.py --full
```

The full check writes calculation summaries and figures to `.tmp/checks/`.
Released data remain in `reproducibility/results/`; their hashes are checked
against `data_manifest.json` before and after the calculation.

## Calculations

| Script | Purpose |
| --- | --- |
| `compute_examples.py` | Pauli moments, T/CCZ checks, QFT and controlled-gate witnesses, and three-site dynamics. |
| `generate_heisenberg4.py` | Periodic four-site dynamics and saved disorder inputs. |
| `symbolic_gates.py` | Exact CCRz and controlled fractional-SWAP polynomials. |
| `certify_qft.py` | QFT moment enclosures and integer certificates. |
| `check_identities.py` | Exact CCZ identities and finite-dimensional continuity diagnostics. |
| `check_figure_data.py` | Analytic Rz curves and independent three-site evolution checks. |
| `check_heisenberg4.py` | Independent four-site evolution and integer-witness checks. |
| `check_tolerance.py` | Witness sensitivity to prescribed tolerances. |
| `render_figures.py` | Rz and both Heisenberg figures from saved data. |

Rz curve samples are in `results/rz_curves.csv`. Heisenberg input fields,
target moments and per-realization integer witnesses are stored separately.
CSV statistics give means and standard errors after integer rounding. The
JSON files ending in `_checks.json` contain numerical verification summaries.

## Conventions and models

All logarithms are base two. For a pure N-qubit state,
`P_alpha = 2^(-N) sum_P |<psi|P|psi>|^(2 alpha)` and
`M_alpha = log2(P_alpha)/(1-alpha)` for alpha different from one.
Choi vectors use `vec(U)/sqrt(d)`; QFT indices run from 0 to d-1.
Rz(theta) is `exp(-i theta Z/2)`. Controlled fractional SWAP has antisymmetric
phase `exp(i*pi*s)` on its two target qubits.

The Heisenberg Hamiltonian is
`H = sum_<j,k> (X_j X_k + Y_j Y_k + 0.2 Z_j Z_k) + sum_j h_j Z_j`,
with independent fields uniform in [-W,W].

| Model | W | Realizations per W | Time grid | PCG64 seed |
| --- | --- | --- | --- | --- |
| Periodic N=4 | 0.5, 1, 2, 3 | 200 | `geomspace(0.1,500,50)` | 20261021 |
| Open N=3 | 1, 3 | 200 | `linspace(0,3,61)` | 20261018 |

Each model initializes its generator once and draws successive W batches.
The same realization is reused across times and tolerances. Figure shading
is one standard error over disorder. In the periodic model,
`[H,Z tensor 4]=0` gives the exact zero-error nullity ceiling `nu(U)<=7`.

The adaptive witness at half-diamond tolerance epsilon is
`ceil(-log2(min(1,P2+16*epsilon))/log2(4/3))`.
It counts the maximum T-state consumption over branches of a finite deterministic
stabilizer protocol with stabilizer work qubits and measurements. The
unitary-implementation correction is `4*epsilon`. Epsilon is a prescribed
synthesis tolerance applied to the ideal target. At epsilon=1e-4, 3,574 of
40,000 four-site integer entries decrease by one.

Numerical witnesses use a `1e-11` downward rounding guard. QFT certification
instead propagates outward intervals for the roots of unity and compares
moment upper bounds with exact rational `(3/4)^k` thresholds. Its assumptions
are IEEE-754 binary64 arithmetic and mpmath interval transcendental enclosures.
Heisenberg checks use independent spin-basis Hamiltonians, matrix exponentials
and direct Pauli-transfer traces. They provide binary64 cross-validation.
Finite ancillary-state tests are diagnostics; the paper gives the general
analytic arguments. The Rz amortized curve is an attainable equatorial-input
lower bound.

## Regeneration

Choose a separate output directory. For PowerShell:

```powershell
$env:SRE_OUTPUT_ROOT = Join-Path (Get-Location) '.tmp/reproduced'
python reproducibility/paths.py
python reproducibility/compute_examples.py
python reproducibility/generate_heisenberg4.py
python reproducibility/check_figure_data.py
python reproducibility/check_heisenberg4.py
python reproducibility/check_tolerance.py
python reproducibility/render_figures.py
Remove-Item Env:SRE_OUTPUT_ROOT
```

For POSIX shells, use `export SRE_OUTPUT_ROOT="$PWD/.tmp/reproduced"` and
`unset SRE_OUTPUT_ROOT`. Floating moments and font rendering may vary slightly
across platforms; compare moments numerically and integer arrays exactly.
