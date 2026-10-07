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
  (plots from `results/experiments.csv`), `frame_potential_post_processing.ipynb`
  (plots from `results/frame_potential.csv` -- frame potential vs. number of
  parameters, the same comparison as the paper's Fig. 2a).

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
see `n_trainable` in `circuits.py`; `n_expvals` is the number of times the
circuit's expectation value was actually estimated over the whole run --
see `build_model`'s `counted_circuit` in `functions.py` -- a more honest cost
metric than `max_steps`, since unlike a step count it scales with
`batch_size` and `n_train_samples` too, the way the real quantum-resource
cost does; with the defaults above it's dominated by the two full-dataset
`cost(...)` calls per step used for logging, not the batched gradient step
itself) and saves its cost curve to
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

# compare architectures at a matched *parameter* budget instead of matched reps
python run.py frame-potential --circuits 1 18 34 --n-qubits 6 --max-params 100 --t 2
```

`--circuits` accepts individual numbers and ranges (`1-19`), mixable and
space-separated. `--converge` uses `estimate_until_converged` instead of a
single batch — that's what you want for real numbers; without it you get
one batch of `--n-samples` (default `2**n_qubits * t`). `--device` defaults
to CUDA if available, else CPU; `--out` defaults to `results/frame_potential.csv`.

`--max-params N` replaces `--reps` (mutually exclusive with it): different
architectures spend wildly different numbers of trainable parameters per
rep, so a fixed `--reps` list isn't a fair comparison across them. With
`--max-params`, each circuit instead sweeps `reps=1,2,3,...` on its own,
checking `circuits.n_trainable` (the actual used count, not the raw
allocated weight-tensor size — see that function's docstring) at every
step, and stops once the next rep would exceed the budget — so every
circuit's last reported reps is its own best match to the same budget,
and the rows along the way show how `F^(t)` evolves with depth up to it.
A circuit whose `reps=1` already exceeds the budget is skipped with a
printed warning rather than silently producing nothing.

With `--converge`, a reps sweep (from either `--reps` or `--max-params`)
also stops early per `(circuit, t)` once `estimate_until_converged`
exhausts `--max-batches` without satisfying its own stopping rule — the
known relative-tolerance pathology from "Checks" below, where `delta` is
already small enough that the `rel_tol * delta` target shrinks about as
fast as sampling can shrink `fidelity_error`. Since more reps only pushes
`F^(t)` closer to Haar (shrinking `delta` further), every larger reps for
that same `(circuit, t)` would hit exactly the same wall, so they're
skipped with a printed note instead of each burning another full
`--max-batches` for no new information; raise `--max-batches` if you
actually need a tighter bound there.

Run `python run.py frame-potential --help` for the full flag list.

(`train` and `frame-potential` are subcommands of the same `run.py` entry
point, each with their own flags — `python run.py --help` lists both.)

### Circuits 33/34: exact-Haar KAK1 ansatz

Both circuits implement Tucci's KAK1 decomposition (`quant-ph/0507171`),
`U = (A1 ⊗ A0) exp(i(k1 XX + k2 YY + k3 ZZ)) (B1 ⊗ B0)`, brickwork-tiled
across any `n_qubits` (same alternating-offset brick pattern as circuit
32) — two local `SU(2)` dressings around a 3-CNOT canonical core, sharing
exactly two gate-level primitives in `circuits.py`: `core_kak(tz, ty1, ty2,
wires)` (the canonical core) and `local_su2(params, wires=None)` (`qp.Rot`
per wire; `params` shaped `(num_wires, 3)`, applied to `wires[i]` for each
row `i`). Neither primitive decides Haar vs. naive itself — that choice is
made by whichever `circuit_set` branch calls them.

**Circuit 34** is the exact-Haar version: every local dressing's raw
`Uniform(0, 2*pi)` parameters go through `haar_reparam.euler_angles`
(closed-form) before `local_su2`, and the canonical core's raw parameters
go through `haar_reparam.sample_canonical` (an empirically-built
Rosenblatt/quantile transform — see that module's docstring for the
derivation and why it's built empirically rather than from a hand-derived
closed form) before `core_kak`. Each dressed 2-qubit gate is exactly
Haar-distributed on `SU(4)`; verified via `frame_potential`: `F^(t)`
matches the exact Haar value `t!` for `t=1,2,3` to within Monte Carlo
error. This does *not* make the whole `n`-qubit unitary Haar-random on its
own (that needs enough `reps` for the brickwork to mix, same as any local
random circuit) — `F^(t)` decreases towards the Haar value as `reps`
grows, as expected.

**Circuit 33** is the ablation: the identical gate structure, but raw
`Uniform(0, 2*pi)` parameters go straight into `local_su2`/`core_kak` with
no `haar_reparam` call at all — no Bloch-sphere correction on the local
angles, no Rosenblatt transform on the canonical ones. It exists purely so
`frame_potential` can quantify what the reparametrization buys you, by
comparing the two directly:

```bash
python run.py frame-potential --circuits 33 34 --n-qubits 2 --reps 3 --t 1 2 3 --device cpu
python run.py frame-potential --circuits 34 --n-qubits 6 --reps 1 2 4 --t 2 --device cpu
```

`F^(1)` is a weak invariant and matches Haar for both (local Haar averaging
alone already gives a 1-design) — the reparametrization's effect only
shows up from `F^(2)` on: at `t=3, n_qubits=2, reps=3`, circuit 34's
`F/F_Haar` ratio is ~1.002 versus circuit 33's ~1.05, a ~20x larger
deviation from Haar, growing with `t`.

Neither circuit spends the paper's literal `15` parameters per gate.
Every layer applies ONE upfront (Haar-random for 34, raw for 33)
`local_su2` dressing per wire before any 2-qubit gate at all, and every
gate from then on — every layer, every pair — reads only a truncated
9-parameter block (`core_kak` + both *trailing* `local_su2` calls, no
leading dressing). A gate's leading dressing would just be a second,
independent local rotation stacked on whatever's already on that wire —
for circuit 34 specifically, already Haar-random by the upfront layer or
an earlier gate's trailing dressing, and composing with anything
independent leaves it Haar-random, so it's redundant, not merely
approximable away. `weight_tensor_shape` for both is `(reps, num_wires//2,
3, 3)`: `params[layer, pair, :2]` is both wires' local dressing (2 rows of
3), read only at layer 0; `params[layer, pair, 2]` is the canonical core's
3 parameters, read at every layer ≥ 1 alongside the trailing dressing.

One quirk worth knowing about specifically at `n_qubits=2`: the
alternating-offset brick pattern's per-layer gate count is `[1, 0, 1, 0,
...]` there (see `circuits._brickwork_layer_pairs`) — layer 0 is always
dressing-only by construction, and layer 1 happens to be an *odd* layer
that fires zero gates at this qubit count, so `reps=1` or `2` only ever
apply local dressings, no canonical core at all. `reps=3` is the smallest
value that actually applies one real 2-qubit gate — that's why the
examples above use it.

`two_designs/haar_reparam.py`'s tables (`two_designs/kak1_rosenblatt_tables.npz`,
checked into the repo) were built from 50,000 Haar-random `SU(4)` samples;
regenerate with `python two_designs/haar_reparam.py --build [--n-samples N]`
(needs `scipy`, offline only — not a runtime dependency). Calibration
checks for circuit 34 — including an explicit check for the
convergence-loop pitfall described under "Checks" below — live in
`checks/validate_local_random.py`, run via `python check.py validate
--only local-random` (or the short alias `--only b`).

`two_designs/range_connectivity.py`'s `draw_display_circuit` uses the same
two primitives directly, for a different reason than either circuit above:
every gate there is independent, with no earlier gate's dressing to
inherit, so it always builds a full, standalone (both-sides-dressed)
block rather than relying on an upfront layer.

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

**Range-limited connectivity** (`two_designs/range_connectivity.py`) —
same exact-Haar KAK1 block as circuit 34, same gate count, but each
layer wires up a *fresh random* pairing of qubits instead of circuit 34's
fixed nearest-neighbour brickwork, restricted to pairs at most `max_range`
apart on a line of qubits (`|i - j| <= max_range`). `max_range = 1` is
local in the same style as circuit 34 (independently randomized per layer
rather than circuit 34's fixed alternating pattern — see
`random_matching`'s docstring); `max_range >= n_qubits - 1` removes the
restriction entirely, so any pair can be wired together ("permuted
brickwork" in the design notes) — connectivity as an axis independent of
gate count, which is the whole point.

The wiring has to vary *per sample*, which a `circuit_set` architecture has
no room for (one gate sequence traced once, then batch-applied with random
angles — see `frame_potential.sample_unitaries`). Retracing circuits.py's
PennyLane queue per sample turned out to dominate runtime completely once
measured, so `sample_range_connected_unitaries` instead calls
`haar_reparam.kak1_block_matrix` — a direct, PennyLane-free reimplementation
of circuit 34's exact math (verified against `qml.matrix()` to machine
precision) — and embeds it at each drawn wire pair via
`frame_potential.apply_embedded_gate` (promoted from a `circuit_set`-internal
helper to a shared utility for exactly this reason).

```bash
python check.py validate --only connectivity   # or: --only c
```

Unlike the other two checks, this one isn't a pass/fail calibration — it's
a comparison tool, since the entire point is *how* the approach to Haar
depends on connectivity range, not whether it eventually gets there. Its
`main` prints a sweep table: `F^(t)/Haar` ratio for every
`(max_range, reps)` combination at matched gate count, e.g. at `n_qubits=6`:

```
reps        range<=1      range<=3      range<=5
2             2.6081        1.9871        1.1129
4             1.7103        1.0394        1.0047
8             1.1946        0.9971        0.9978
```

Reading a column down shows the gain from more depth at fixed range;
reading a row across shows the gain from more reach at fixed depth — here,
`range<=1` at `reps=8` (ratio 1.19) is still further from Haar than
`range<=3` reaches at half the depth, `reps=4` (ratio 1.04), reproducing
the design notes' claim that letting gates reach further matters as much
as adding more of them at fixed gate count. Customize the sweep — it
forwards extra arguments straight to `checks/validate_connectivity.py`'s
own `argparse` parser:

```bash
python check.py validate --only c --n-qubits 8 --reps 2 4 8 16 --ranges 1 2 4 7 --n-samples 3000
```

## Checks

`check.py` is a second, separate CLI from `run.py` — deliberately: `run.py`
runs the repeatable experiments (training sweeps, frame-potential sweeps)
that get logged to a CSV, while `check.py` runs one-off validation and
performance scripts that just print a report. Mirrors `run.py`'s own
subcommand style:

```bash
python check.py benchmark                       # timing benchmarks (everything)
python check.py benchmark --only fp             # timing benchmarks, frame-potential only
python check.py benchmark --only stress         # frame potential at higher n_qubits + convergence time
python check.py validate                        # every two_designs calibration check
python check.py validate --only clifford        # just one, by explicit name...
python check.py validate --only a               # ...or by short alias, either works
python check.py validate --only local-random    # another one (alias: b)
python check.py validate --only connectivity    # another one (alias: c) -- extra args
python check.py validate --only c --n-qubits 8  # forward straight to its own CLI
```

**`benchmark --only stress`** is separate from `--only fp` on purpose: `fp`
is a quick, fixed-size regression check run on every available device as
part of the default everything-sweep; `stress` is for finding out how far
*this* machine actually reaches — higher `n_qubits`, and full
`estimate_once`/`estimate_until_converged` timing (not just the pairwise-trace
microbenchmark `fp` covers), on a single device (CUDA if available, not
cpu+cuda both — cpu at n_qubits >= 12 would just time out for no
information gained). Batch sizes are capped by
`frame_potential.recommended_batch_size` the same way production code is; a
size that heuristic predicts should fit but still OOMs is reported, not
treated as an error — that's exactly the information this is for.

```bash
python check.py benchmark --only stress --device cuda
python check.py benchmark --only stress --n-qubits 8 10 12 14
python check.py benchmark --only stress --circuits 1 18 34 --reps 2
```

`validate`'s checks are registered in `checks/validate.py`'s `CHECKS` list,
each with an explicit `name` (used everywhere in output and docs) and a
tuple of `aliases` it can also be called by — short letters like `a`/`b`/`c`
are there purely for fast typing, never used as the ensemble's actual
identity in code, file names, or documentation (the design notes this
project builds on label these "Family A", "Family B", "Family C"; that
labeling is cross-referenced once in each module's docstring for anyone
going back to the source, but isn't used as an identifier anywhere in this
repo — `--only a` is offered as a convenience alias precisely so the terse
form stays available without making it the primary name). Add a new one by
writing `checks/validate_something.py` with a `main(argv=None)`, then
adding one entry to `CHECKS`.

**On `estimate_until_converged`'s relative-tolerance stopping rule:** the
Clifford-group and local-random checks hit the same failure mode
independently — worth calling out here since it's a property of the
stopping rule itself, not of either ensemble. It stops when
`fidelity_error <= rel_tol * |delta|` (or an absolute floor). For any
ensemble that's *supposed* to be at or near the Haar value — an exact
design like the Clifford group, or an intentionally-Haar-exact block like
circuit 34 — `delta` is small by construction, so the relative target
shrinks about as fast as sampling can shrink `fidelity_error`, and the loop
burns every `max_batches` without ever satisfying its own criterion
(confirmed for both: ~13s / ~190M pairs on circuit 34 at n_qubits=2 alone).
Both checks use `estimate_once`/`estimate_once_from_sampler` with a fixed
sample count instead — the right tool whenever you already expect
`delta ≈ 0`, since there's no target to converge *towards*, just a spread
to report.

`check.py benchmark`/`check.py validate` forward their trailing arguments
straight to `checks/benchmark.py`'s / `checks/validate.py`'s own `argparse`
parsers, and `validate` forwards a second time when `--only` picks a
specific check with flags of its own (`checks/validate_connectivity.py`'s
`--n-qubits`/`--reps`/`--ranges`/etc., via `parse_known_args` — anything
`--only` itself doesn't recognize is passed straight through). Add an
entirely new top-level command (as opposed to a new ensemble under
`validate`) by writing `checks/your_command.py` with a `main(argv=None)`,
then adding one `elif` in `check.py`.

`check.py` lives at the repo root for the same reason `run.py` does:
running `python check.py ...` puts the repo root on `sys.path`
automatically (Python does this for whatever file you invoke directly), so
`checks/*.py` can `import frame_potential` and `from two_designs... import
...` without needing `python -m` or manual `sys.path` edits — which is also
why none of these are runnable as `python -m checks.validate_clifford`
anymore; go through `check.py` instead.

**`check.py show`** (`checks/show.py`) answers "what circuit am I actually
testing" directly, rather than requiring a read through `circuits.py` or
`two_designs/range_connectivity.py`'s source. Two targets:

```bash
python check.py show 34 --n-qubits 6 --reps 2                        # any circuit_set number
python check.py show connectivity --n-qubits 5 --reps 3 --max-range 1 --seed 42
```

A `circuit_set` number (any of 1–36, retroactively — they're already built
from real PennyLane operations, so `qml.draw` already knows how to render
them) prints the weight-tensor shape plus the full ASCII gate diagram.
`connectivity` is the one case that needs its own path:
`two_designs/range_connectivity.py`'s fast sampler deliberately never
builds a PennyLane circuit at all (see "Range-limited connectivity" above
— that retracing was the exact performance bug fixed last session), so
there's no existing object to draw. `draw_display_circuit` is a second,
display-only builder — never called from the sampler — that draws one
concrete wiring plus one concrete set of gate angles (both reproducible via
`--seed`) and queues real `core_kak`/`local_su2` calls for it. Its output prints
the wiring in plain text first (which pairs connect in which layer, and
which qubits sit idle that layer), then the matching ASCII diagram, so the
connectivity pattern doesn't have to be reverse-engineered from the
diagram's wire lines. Not covered: the Clifford-group ensemble — it's a
`stim` stabilizer tableau, not a gate sequence; `stim.Tableau.to_circuit()`
is the right tool if you want to look inside one instead.

Every run appends to the CSV rather than overwriting it, so a sweep can be
resumed or extended across sessions by pointing `--out` at the same file.
Load it for post-processing with `frame_potential.load_frame_potential()`.

Qubit counts above ~8 get noticeably slower on CPU — the unitary
construction is memory-bandwidth-bound, which is exactly where a GPU (much
higher bandwidth) pays off; correctness doesn't depend on which device you use.


### Generative IA Usage

Generative AI (Claude code and Chatgpt) were used to generate, explaine and advise me on this project.
