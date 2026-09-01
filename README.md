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
- `haar_reparam.py`: the exact-Haar reparametrization used by circuits 33/34
  (see below) — converts raw `Uniform(0, 2*pi)` circuit parameters into
  Haar-distributed SU(2) Euler angles and Weyl-chamber canonical angles.
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
does **not** give a Haar-random unitary — `haar_reparam.py` reshapes those
raw `Uniform(0, 2*pi)` draws (closed-form for the 12 local angles, an
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

`haar_reparam.py`'s tables (`kak1_rosenblatt_tables.npz`, checked into the
repo) were built from 50,000 Haar-random `SU(4)` samples; regenerate with
`python haar_reparam.py --build [--n-samples N]` (needs `scipy`, offline
only — not a runtime dependency).

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

Every run appends to the CSV rather than overwriting it, so a sweep can be
resumed or extended across sessions by pointing `--out` at the same file.
Load it for post-processing with `frame_potential.load_frame_potential()`.

Qubit counts above ~8 get noticeably slower on CPU — the unitary
construction is memory-bandwidth-bound, which is exactly where a GPU (much
higher bandwidth) pays off; correctness doesn't depend on which device you use.


### Generative IA Usage

Generative AI (Claude code and Chatgpt) were used to generate, explaine and advise me on this project.
