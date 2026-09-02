# Circuits_training

PennyLane implementation of training different parameterized-circuit 
architectures on artificial Fourier-series datasets,
and estimating each architecture's frame potential `F^(t)` (expressibility,
compared to the Haar reference `F/F_Haar`).

## Contents

- `circuits.py`: 19 architectures from a paper (***mettre la ref***)(`circuit_set(num)`, `num` 1-19)
  plus a few extras (30-36, see below), `weight_tensor_shape` (parameter
  tensor shape per architecture), and `n_trainable` (how many of those
  parameters a circuit actually reads — several architectures allocate more
  than they use).
- `two_designs/`: ensembles from the "Building 2-Designs" design notes that
  aren't `circuit_set` architectures, plus `haar_reparam.py` — see below.
- `functions.py`: the target Fourier functions to fit, the training loop
  (`train`), and the QNode builder (`build_model`).
- `experiment_tracker.py`: runs `train` and appends one row per run to
  `results/experiments.csv` (metadata), saving each cost curve to its own
  `results/costs/{experiment_id}.npy`.
- `frame_potential.py`: batched, GPU-compatible frame-potential estimation
  computed directly from `circuit_set` (see below).
- `run.py`: CLI for both training and frame-potential estimation (`train`
  and `frame-potential` subcommands). Retires the old `test.py` (hardcoded
  constants, no CLI, no seeding) — `run.py train` reproduces the same sweep
  with proper flags and best-effort seeding.
- `check.py` / `checks/`: CLI for one-off validation and benchmark scripts,
  as opposed to run.py's repeatable experiments — see "Checks" below.
- `notebooks/`: `building_circuits.ipynb` (circuit sanity checks),
  `Fourier.ipynb`/`training_and_saving.ipynb` (training), `post_processing.ipynb`
  (plots from `results/experiments.csv`).

## Install

```
make install
```

## Training

```python
from experiment_tracker import train_and_record
from functions import function_to_learn
import torch

target = function_to_learn(degree=2)
x = torch.linspace(-torch.pi, torch.pi, 800)
y = target(x)

train_and_record(x, y, circuit_num=7, n_qubits=6, layers=3, anzats_reps=1,
                  max_steps=600, path="results/")
```

Every run appends a row to `results/experiments.csv` (`n_params` is the
number of parameters the circuit actually reads, not the raw tensor size —
see `n_trainable` in `circuits.py`) and saves its cost curve to
`results/costs/{experiment_id}.npy` — one small binary file per run, so a
run's `max_steps` never has to match any other run's, and nothing repeats
the experiment id or step index the way a CSV would need to. Load one curve
with `load_cost_curve(experiment_id, path=...)`, or every curve at once
(as a list of `{experiment_id, n_steps, costs}` dicts, ready for
`pd.DataFrame(...)`) with `load_costs(path=...)`.

`layers` is the number of variational blocks; data (`x`) is only
re-encoded *between* blocks, so `layers=1` means the model never reads `x`
at all (a degenerate, constant-output model) — use `layers >= 2` for the
model to actually depend on its input. `run.py train`'s default is 3,
matching `test.py`'s original sweep.

### CLI

```bash
# a few architectures on one target function, quick check
python run.py train --circuits 1 7 11 --n-qubits 6 --layers 3 --reps 1 --max-steps 600

# sweep matching test.py's original defaults
python run.py train --circuits 1-19 30 31 32 --n-qubits 6 --layers 3 --reps 1 2 3 \
    --degrees 10 --n-functions 5 --max-steps 600
```

For each `--degrees`/`--n-functions` draw, a fresh random target function is
generated and every `(--n-qubits, --layers, --reps, --circuits)` combination
is trained on that *same* function/dataset — matching the paper's
architecture-comparison methodology. `--out` (default `results/`) is the
directory `experiments.csv` and `costs/` are written to. `--seed` gives
best-effort reproducibility (seeds target-function generation and PyTorch;
not a bit-for-bit guarantee). Run `python run.py train --help` for the
full flag list.

## Frame potential

`frame_potential.py` estimates `F^(t)` for any `circuit_set` architecture by
sampling many random-parameter unitaries and averaging `|Tr(Ui†Uj)|^(2t)`
over pairs. It's built from four pieces:

- `sample_unitaries(num, n_qubits, reps, batch_size, device=..., dtype=...)`
  — the batched unitary construction. Traces `circuit_set(num)`'s gate
  sequence once (via PennyLane's own queuing/tape mechanism, so it's always
  in sync with the circuit actually used for training) and applies each
  gate to the whole batch via a reshape + axis-contraction, not a full
  kron(I, gate, I) + matmul — `O(batch * d^2)` per gate instead of
  `O(batch * d^3)`, plus fusing consecutive same-wire single-qubit gates
  into one matrix first. Pure torch ops throughout, so `device=torch.device("cuda")`
  runs it on GPU with no other changes. Verified exact against `qml.matrix()`
  for circuits 1-19, 31-32, and 33-36 (circuit 30, a PennyLane built-in template, is
  not supported — see its docstring).
- `Estimate` — a frozen dataclass holding one Monte Carlo batch's raw sums;
  `frame_potential`, `delta`, `ratio`, `fidelity_error` are derived
  properties, and `estimate_a + estimate_b` pools two independent batches
  (with a statistically correct pooled variance, not just summed samples).
- `estimate_once(...)` / `estimate_until_converged(...)` — one fixed-size
  batch, or a loop that keeps pooling batches until the 95% CI is tight
  relative to `|delta|` (`rel_tol`, default 0.4).
- `save_estimate(...)` / `load_frame_potential(...)` — append one row to
  `results/frame_potential.csv` (git-commit-stamped) / load it as a pandas
  DataFrame.

### CLI

```bash
# quick single-circuit check
python run.py frame-potential --circuits 7 --n-qubits 4 --reps 1 --t 2 --device cpu

# real sweep over the 19 paper architectures, converged
python run.py frame-potential --circuits 1-19 --n-qubits 6 --reps 1 2 3 --t 2 --converge --seed 0
```

`--circuits` accepts individual numbers and ranges (`1-19`), mixable and
space-separated. `--converge` uses `estimate_until_converged` instead of a
single batch — that's what you want for real numbers; without it you get
one batch of `--n-samples` (default `2**n_qubits * t`). `--device` defaults
to CUDA if available, else CPU; `--out` defaults to `results/frame_potential.csv`.
Run `python run.py frame-potential --help` for the full flag list.

(`train` and `frame-potential` are subcommands of the same `run.py` entry
point, each with their own flags — `python run.py --help` lists both.)

### Circuits 33/34: exact-Haar KAK1 ansatz

Circuit 33 is Tucci's KAK1 decomposition (`quant-ph/0507171`) as a literal
2-qubit circuit: `U = (A1 ⊗ A0) exp(i(k1 XX + k2 YY + k3 ZZ)) (B1 ⊗ B0)`,
i.e. 4 local `SU(2)` blocks (3 Euler-angle rotations each) around a 3-CNOT
canonical core — 15 single-qubit rotations + 3 CNOTs total, matching the
paper's construction exactly.

Unlike every other circuit here, sampling its 15 raw parameters uniformly
does **not** give a Haar-random unitary — `two_designs/haar_reparam.py`
reshapes those raw `Uniform(0, 2*pi)` draws (closed-form for the 12 local angles, an
empirically-built Rosenblatt/quantile transform for the 3 non-local ones —
see that module's docstring for the derivation and why it's built
empirically rather than from a hand-derived closed form) so that the
resulting unitary is exactly Haar-distributed on `SU(4)`. Verified via
`frame_potential`: `F^(t)` matches the exact Haar value `t!` for `t=1,2,3`
to within Monte Carlo error.

Circuit 34 applies the same 15-parameter Haar-exact block brickwise across
`n_qubits > 2` (same alternating-offset brick pattern as circuit 32), so
each local 2-qubit interaction is individually exactly Haar-random; it
reduces to circuit 33 when `n_qubits == 2`. This does *not* make the whole
`n`-qubit unitary Haar-random on its own (that needs enough `reps` for the
brickwork to mix, same as any local random circuit) — `F^(t)` for circuit
34 decreases towards the Haar value as `reps` grows, as expected.

```bash
python run.py frame-potential --circuits 33 --n-qubits 2 --reps 1 --t 1 2 3 --device cpu
python run.py frame-potential --circuits 34 --n-qubits 6 --reps 1 2 4 --t 2 --device cpu
```

`two_designs/haar_reparam.py`'s tables (`two_designs/kak1_rosenblatt_tables.npz`,
checked into the repo) were built from 50,000 Haar-random `SU(4)` samples;
regenerate with `python two_designs/haar_reparam.py --build [--n-samples N]`
(needs `scipy`, offline only — not a runtime dependency). Calibration checks
for circuits 33/34 — including an explicit check for the convergence-loop
pitfall described under "Checks" below — live in `checks/validate_local_random.py`,
run via `python check.py validate --only local-random` (or the short alias
`--only b`).

**Circuits 35/36** are the ablation: the exact same gate structure as 33/34
(same `kak1_core`, same 4 local `SU(2)` blocks, same parameter count), but
the raw `Uniform(0, 2*pi)` parameters are used directly as gate angles
instead of being pushed through `haar_reparam` — no Bloch-sphere correction
on the local angles, no Rosenblatt transform on the 3 non-local ones. They
exist purely so `frame_potential` can quantify what the reparametrization
buys you, by comparing 33 against 35 (and 34 against 36) directly:

```bash
python run.py frame-potential --circuits 33 35 --n-qubits 2 --reps 1 --t 1 2 3 --device cpu
```

`F^(1)` is a weak invariant and matches Haar for both (local Haar averaging
alone already gives a 1-design) — the reparametrization's effect only shows
up from `F^(2)` on: at `t=3`, circuit 33's `F/F_Haar` ratio is ~1.002 versus
circuit 35's ~1.05, a ~20x larger deviation from Haar, growing with `t`.

### `two_designs/`: ensembles from the "Building 2-Designs" notes

Some ensembles worth benchmarking against aren't `circuit_set` architectures
at all — no continuous parameters, no PennyLane gates to trace — so they
don't fit `circuit_set(num)`'s numbering. `two_designs/` is where the
reusable code for those lives, one file per ensemble from the design notes
(named for what they are, not for the design notes' own "Family A/B/C..."
labels — those are cross-referenced in each module's docstring for anyone
going back to the source, but aren't used as identifiers here); the
runnable script that exercises each one lives in `checks/` instead (see
"Checks" below).

To support them without duplicating the accumulation/pooling/confidence-
interval logic, `frame_potential.py`'s estimators are split into a
circuit_set-specific layer and a generic one underneath:

- `estimate_once_from_sampler(sampler, d, t, n_samples, ...)` /
  `estimate_until_converged_from_sampler(sampler, d, t, ...)` — take any
  `sampler(batch_size, *, device, dtype, generator) -> Tensor[batch_size, d, d]`
  callable and `d` (the sampler can't be introspected for it). This is where
  the actual math lives.
- `estimate_once(num, n_qubits, reps, t, ...)` /
  `estimate_until_converged(num, n_qubits, reps, t, ...)` — unchanged
  signatures, now thin wrappers that build a `circuit_set`-backed sampler
  and delegate. Every existing call site (`run.py`, this README,
  `checks/benchmark.py`) keeps working exactly as before.

**Random Clifford circuits** (`two_designs/clifford_group.py`) — not an
ansatz, a calibration check. The Clifford group is an exact 3-design, so
`F^(t)` must come out to exactly `t!` for `t = 1, 2, 3` at every `n` (`t! `
is only the exact Haar value for `t <= d = 2**n_qubits` — Schur–Weyl needs
that many independent permutation operators, so e.g. `n_qubits=1, t=3` isn't
a valid check and isn't one). Any value that isn't `t!` (for `t <= d`)
within its confidence interval is a bug in the estimator, not a discovery.

Two ways to check it, both in `checks/validate_clifford.py` (see "Checks"
below for why the runnable script lives in `checks/` while the reusable
sampler/exact-group functions it calls stay in `two_designs/`):

- **Exact.** For `n_qubits` small enough to enumerate the *entire* Clifford
  group (`n=1`: 24 elements, `n=2`: 11,520 — via `stim.Tableau.iter_all`),
  `exact_estimate_from_group` sums `|Tr(Ui^dagger Uj)|^(2t)` over literally
  every pair, `i` and `j` both ranging over the whole group. Zero Monte Carlo
  variance — residual ~1e-7 disagreement with `t!` is `stim`'s native
  `complex64` output, not noise.
- **Sampled.** `sample_clifford_unitaries` (uniformly random Cliffords via
  `stim.Tableau.random`, the same Bravyi–Maslov canonical form `qiskit`'s
  `random_clifford` uses) pushed through the generic
  `estimate_once_from_sampler` — the same code path any future ensemble here
  will use.

```bash
python check.py validate --only clifford   # or: --only a
```

Note: use `estimate_once_from_sampler` (fixed sample count), not
`estimate_until_converged_from_sampler`, for a near-exact design like this
one. Convergence there is judged by `rel_tol * |delta|`, and `delta` is
supposed to be ~0 here — so the target the loop chases shrinks along with
the thing it's measuring, and it'll burn every `max_batches` doubling the
sample size chasing noise. That's a real property of the relative-tolerance
stopping rule, worth knowing about for any near-exact-design ensemble, not
specific to Cliffords.

## Checks

`check.py` is a second, separate CLI from `run.py` — deliberately: `run.py`
runs the repeatable experiments (training sweeps, frame-potential sweeps)
that get logged to a CSV, while `check.py` runs one-off validation and
performance scripts that just print a report. Mirrors `run.py`'s own
subcommand style:

```bash
python check.py benchmark                     # timing benchmarks (everything)
python check.py benchmark --only fp           # timing benchmarks, frame-potential only
python check.py validate                      # every two_designs calibration check
python check.py validate --only clifford      # just one, by explicit name...
python check.py validate --only a             # ...or by short alias, either works
python check.py validate --only local-random  # the other one (alias: b)
```

`validate`'s checks are registered in `checks/validate.py`'s `CHECKS` list,
each with an explicit `name` (used everywhere in output and docs) and a
tuple of `aliases` it can also be called by — short letters like `a`/`b` are
there purely for fast typing, never used as the ensemble's actual identity
in code, file names, or documentation (the design notes this project builds
on label these "Family A", "Family B", etc.; that labeling is cross-referenced
once in each module's docstring for anyone going back to the source, but
isn't used as an identifier anywhere in this repo — `--only a` is offered as
a convenience alias precisely so the terse form stays available without
making it the primary name). Add a new one by writing
`checks/validate_something.py` with a `main()`, then adding one entry to
`CHECKS`.

**On `estimate_until_converged`'s relative-tolerance stopping rule:** the
Clifford-group and local-random checks hit the same failure mode
independently — worth calling out here since it's a property of the
stopping rule itself, not of either ensemble. It stops when
`fidelity_error <= rel_tol * |delta|` (or an absolute floor). For any
ensemble that's *supposed* to be at or near the Haar value — an exact
design like the Clifford group, or an intentionally-Haar-exact block like
circuit 33 — `delta` is small by construction, so the relative target
shrinks about as fast as sampling can shrink `fidelity_error`, and the loop
burns every `max_batches` without ever satisfying its own criterion
(confirmed for both: ~13s / ~190M pairs on circuit 33 at n_qubits=2 alone).
Both checks use `estimate_once`/`estimate_once_from_sampler` with a fixed
sample count instead — the right tool whenever you already expect
`delta ≈ 0`, since there's no target to converge *towards*, just a spread
to report.

Both `check.py benchmark` and `check.py validate` forward their trailing
arguments straight to `checks/benchmark.py`'s / `checks/validate.py`'s own
`argparse` parsers — add an entirely new top-level command (as opposed to a
new ensemble under `validate`) by writing `checks/your_command.py` with a
`main()` (optionally taking `argv=None` if it wants its own flags), then
adding one `elif` in `check.py`.

`check.py` lives at the repo root for the same reason `run.py` does:
running `python check.py ...` puts the repo root on `sys.path`
automatically (Python does this for whatever file you invoke directly), so
`checks/*.py` can `import frame_potential` and `from two_designs... import
...` without needing `python -m` or manual `sys.path` edits — which is also
why none of these are runnable as `python -m checks.validate_clifford`
anymore; go through `check.py` instead.

Every run appends to the CSV rather than overwriting it, so a sweep can be
resumed or extended across sessions by pointing `--out` at the same file.
Load it for post-processing with `frame_potential.load_frame_potential()`.

Qubit counts above ~8 get noticeably slower on CPU — the unitary
construction is memory-bandwidth-bound, which is exactly where a GPU (much
higher bandwidth) pays off; correctness doesn't depend on which device you use.


### Generative IA Usage

Generative AI (Claude code and Chatgpt) were used to generate, explaine and advise me on this project.
