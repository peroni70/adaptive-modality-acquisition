# Adaptive Modality Acquisition

Companion code for *[paper title]*.

Information is rarely free. A clinician orders one test before another, a
sensor rig spends power per reading, an annotation pipeline pays per label.
This package learns **when to stop paying**: a value model scores a proposed
subset of modalities against what has already been observed, and an
acquisition policy uses it to buy information in stages, stopping once nothing
left is worth its cost.

```
observe context -> score candidate subsets -> acquire the best -> repeat -> predict
                          ^                                          |
                          +------------------------------------------+
                                     until nothing pays for itself
```

## Install

```bash
pip install -e .
```

That is all you need to run everything below on a laptop. A GPU is used
automatically when one is visible, and is not required.

For a pinned environment, or on a cluster, see [Environments](#environments).

## Quickstart

```bash
ama run configs/toy_sensors.yaml
```

A few minutes on a CPU, no downloads. It trains a classifier, ranks the
modalities, trains a value model, and evaluates the policy against fixed
baselines at three cost levels. The next section walks through what it did.

For a real dataset, `configs/patch_mnist.yaml` cuts MNIST digits into a 4x4
grid of patches and treats each patch as separately acquirable.

## A worked example

`configs/toy_sensors.yaml` describes eight sensors on a machine. Sensor 0 is
free - the ambient reading - and the other seven cost money.

The machines come in **two types**, and the free reading says which. On a type
A machine the fault shows up in sensors 2 and 5; on a type B machine it shows
up in sensors 3 and 4. Sensors 1, 6 and 7 are noise for everyone.

That structure is the point. No fixed set of sensors is efficient here: a
population-level ranking has to buy all four informative sensors to cover both
types, paying twice over on every machine. A policy that reads the free sensor
first knows which type it is looking at, and buys only the two that matter.
**No ranking of sensors, however good, can do this**, because the right answer
depends on the case in front of you.

Setting this up took three things.

### 1. A loader

Return your splits as anything indexable that yields `(x, y)`. `TensorSource`
wraps a pair of tensors; a `torch.utils.data.Dataset` works directly.

```python
from ama.data.sources import TensorSource
from ama.registry import register_app

@register_app("toy_sensors")
def load_splits(cfg):
    generator = torch.Generator().manual_seed(cfg.seed)
    return {
        split: TensorSource(*make_split(int(sizes[split]), generator))
        for split in ("train", "val", "test")
    }
```

The whole app is [`src/ama/apps/toy_sensors.py`](src/ama/apps/toy_sensors.py),
about fifty lines including the data generator.

### 2. A masker

The masker is the one piece that depends on how your data is laid out. Each
sensor here owns four consecutive columns of the feature vector, which is
exactly what `SliceMasker` describes, so there is no code to write - just
config:

```yaml
masker:
  type: slice
  spans: [[0, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]]
```

Three built-in layouts cover most data:

| Masker | Layout | Config |
|---|---|---|
| `SliceMasker` | contiguous spans of a flat vector | `{type: slice, spans: [[0,4], ...]}` |
| `PatchMasker` | rectangles of a `(C, H, W)` image | `{type: patch, grid: {image_size: 28, patch_size: 7}}` |
| `ChannelMasker` | entries along the modality axis, e.g. ECG leads | `{type: channel, n_modes: 12}` |

**If your layout differs**, subclass `Masker` and implement one method. Say
the readings were interleaved rather than blocked - sensor `i` owning columns
`i`, `i + 8`, `i + 16`, ... - which no built-in covers:

```python
from ama.masking import Masker

class InterleavedMasker(Masker):
    def region(self, mode):
        # Index expression selecting this modality's slice of one example.
        return (..., slice(mode, None, self.n_modes))
```

`region` returns an index expression; the base class handles the rest, batching
included.

One thing to know: masking **zeroes what each modality claims** and leaves
everything else alone. Any part of the input that no modality's `region`
covers is therefore always visible, which is convenient for shared context
that comes for free - but it does mean your regions should account for every
part of the input you intend to be able to hide.

### 3. A config

Point at the app, name the masker, and describe the sweep. See
[`configs/toy_sensors.yaml`](configs/toy_sensors.yaml) - it is 40 lines, and
most of it is model sizes.

### What comes out

The free sensor alone gets 50% - it says which type of machine you have, never
whether it is faulty. The static ranking is forced to interleave both types'
sensors, and needs all four before it reaches full accuracy:

```
greedy order : [3, 2, 4, 5, 7, 6, 1]
```

At a cost level where buying everything is a bad deal:

| method | accuracy | cost | reward |
|---|---|---|---|
| context only | 0.501 | 0.000 | 0.000 |
| best fixed prefix (4 sensors) | 0.960 | 0.217 | 0.242 |
| fixed order, adaptive stopping | 0.934 | 0.189 | 0.244 |
| **adaptive policy** | **0.967** | **0.107** | **0.359** |

Higher accuracy at **half the cost**, for a 48% better reward than the best
fixed prefix. The third row is the control worth dwelling on: taking the same
population-level ranking and merely choosing *when to stop* per machine buys
almost nothing (0.244 vs 0.242). The gain comes from conditioning *which*
sensors to buy on the case at hand.

And it does exactly that - buying each type's informative sensors on ~99% of
machines, and the other type's on ~15%:

```
 sensor    type A    type B
      2     99.1%     12.7%   <- informative for A
      5     97.6%     21.1%   <- informative for A
      3     17.1%     99.5%   <- informative for B
      4     22.8%     98.8%   <- informative for B
      6     21.6%     19.4%      noise
```

It buys 2.85 sensors per machine on average, against the four a fixed prefix
needs.

## Costs

Costs enter the objective the policy maximizes at each stage:

```
max_Q  v(Q | P)  -  cost(Q)
```

`v` is what the value model predicts. For `acc_change` that is a **change in
expected accuracy**, so a cost in dollars cannot be subtracted from it as-is.
The two must be put in the same units first.

### Real costs

Supply the costs you actually face, plus the exchange rate:

```bash
ama eval-policy configs/toy_sensors.yaml --costs costs.npy --lambda 40
```

`--costs` takes a `.pt`, `.npy` or `.csv` holding either

- a `(n_modes,)` vector - the same price list for every case, or
- an `(n_examples, n_modes)` matrix - **per-case costs**, for when the price
  depends on the subject: a test already on file is free, a scan at one site
  costs more than at another.

`--lambda` is the exchange rate, and reads as **what you are willing to spend
per unit of expected accuracy gained**, in the same currency as the costs:

```
effective_cost = cost / lambda
```

With costs in dollars and `acc_change` predicting a change in expected
accuracy, `--lambda 40` says *one additional point of expected accuracy is
worth $40 to me* (0.01 accuracy for $0.40). Raise it and the policy buys more
freely; lower it and it acquires only when the expected gain is large. Setting
it is a policy decision about what accuracy is worth, and it is the knob that
turns this from a benchmark into a deployable rule.

### Simulated costs

With no `--costs`, the pipeline sweeps `policy.alphas`, drawing costs per
example from a multivariate normal clipped at zero and scaled by `alpha`. This
traces out how each method responds as acquisition gets more expensive, which
is what the paper's figures show. `alpha` is the reciprocal of `lambda`.

The context modality is never billed. Reward is reported as

```
reward = accuracy - baseline_accuracy - cost
```

against a baseline that observes context alone. Per-example rows record the
same quantities, so two methods can be compared on identical examples under
identical costs.

## The four stages

Each stage reads the previous one's artifacts, so a sweep over value functions
shares one classifier and a long evaluation can be re-run without retraining.

| Stage | What it does |
|---|---|
| `train-classifier` | Trains one classifier on randomly masked examples, so it can predict from any subset of modalities. Frozen thereafter. |
| `greedy-order` | Ranks modalities by greedy validation gain. The static baseline, and the backbone of the `adaptive_greedy` policy. |
| `train-value` | Trains the value model on (state, proposal) pairs labelled by what the frozen classifier actually does. Optionally calibrates it. |
| `eval-policy` | Applies costs, runs each policy, and scores it against the static prefixes. |

```bash
ama train-classifier configs/toy_sensors.yaml
ama train-value      configs/toy_sensors.yaml --value-fn acc_change
ama eval-policy      configs/toy_sensors.yaml --costs costs.npy --lambda 40
```

`ama run` does all four. Any config field can be overridden with `-o key=value`,
and `ama show <config> -o ...` prints the resolved config so you can confirm an
override landed before starting a long run.

Results are written to `<run_dir>/<value_fn>/`: one summary row per method and
one row per test example, for each cost setting.

## Value functions

What the value model is trained to predict about a proposed acquisition.
Selected per run; each defines its own target, output width, loss and metric.

| Name | Target | Reads as |
|---|---|---|
| `acc_change` | `{0, 1, 2}` | correctness lost / unchanged / gained. Its score, `P(gained) − P(lost)`, is an expected accuracy change |
| `bit_flip` | `{0, 1}` | acquiring turns a wrong prediction right |
| `info_gain` | real | reduction in cross-entropy loss |

`acc_change` is the default, and the one whose units make `lambda`
interpretable. Adding another means subclassing `ValueFunction` and
registering it.

## Subset optimizers

Each stage solves `max_Q v(Q | P) − cost(Q)` over unacquired modalities. The
value model is not guaranteed submodular and the objective is not monotone, so
these are heuristics, differing in how many value-model evaluations they spend.

| Name | Cost | Notes |
|---|---|---|
| `single_item_greedy` | one pass | proposes at most one modality per stage |
| `greedy_usm` | cheap | adds the best remaining modality until it stops helping |
| `rand_usm` | moderate | randomized double greedy for unconstrained maximization |
| `hybrid_usm` | both | runs the two above, keeps the better subset |
| `enum` | `2^M` | exact; only practical for few modalities |

## Policies

* **`eama`** - the adaptive multi-stage policy driven by the value model.
* **`adaptive_greedy`** - fixed greedy order, but the stopping point is chosen
  per example from that example's realized costs. A cost-aware static baseline.
* **`preset`** - a fixed subset for every example, used for the static
  greedy-prefix baselines.

## Environments

`pip install -e .` is enough to run locally. Device selection is automatic;
force it with `--device cpu` or `--device cuda`.

For a pinned environment, or on a cluster where the interpreter must be
unambiguous:

```bash
bash scripts/setup_env.sh     # creates a conda env named 'ama'
conda activate ama
```

That pins the tested combination - torch 2.10.0 / torchvision 0.25.0 from the
CUDA 12.8 index - and verifies that every dependency resolves from inside the
environment. It handles two hazards common on shared machines: a user
site-packages directory shadowing the environment, and a home-backed `TMPDIR`
too small for torch's wheels. Override with `AMA_TORCH_INDEX`, `AMA_SKIP_TORCH`,
`AMA_TMPDIR`, or `AMA_CONDA_ENV`.

### Clusters

`scripts/slurm/` holds SLURM job scripts. They request the least site-specific
resources that work (`--gres=gpu:1`, 8 cores, default partition); each header
lists the usual adjustments. Point them at an environment with
`AMA_CONDA_ENV=<name>`, and name any required modules with
`AMA_MODULES="..."`. Every job reports its interpreter, torch version, CUDA
build and GPU visibility before doing any work.

Trailing arguments pass through to every stage:

```bash
sbatch scripts/slurm/patch_mnist.sh -o value_model.epochs=40
```

## Layout

```
src/ama/
  masking.py      Masker + the three built-in layouts
  value_fns.py    the value functions
  costs.py        simulated and user-supplied acquisition costs
  optimizers.py   the five subset optimizers
  metrics.py      AUC / accuracy conventions per task
  data/           datasets for each stage; policy.py holds the acquisition loop
  modeling/       encoders, set embeddings, classifier and value model wrappers
  training/       training loops, submodular hinge loss, calibration
  evaluation/     greedy ordering, policy scoring
  pipeline.py     the four stages
  apps/           per-dataset loaders
```

## Tests

```bash
pytest
```

## Note on preparation

The research this accompanies - the method, the experimental design, and the
results reported in the paper - is the authors' own work.

Claude (Anthropic) was used to prepare this repository for release: it
consolidated four separate per-application implementations into the shared
library here, and wrote the tests, documentation, packaging and the synthetic
example used in the Quickstart. It was not used to generate the ideas, design
the experiments, or produce the results in the paper.

## Citation

```bibtex
@article{TODO,
  title  = {TODO},
  author = {TODO},
  year   = {TODO}
}
```
