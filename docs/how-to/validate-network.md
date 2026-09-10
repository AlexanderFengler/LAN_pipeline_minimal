# Validate and inspect a candidate network

Validation asks a narrow operational question: will HSSM load this ONNX
likelihood and obtain numerically and scientifically plausible values? Run it
on artifacts you trust before any publication attempt.

## Install the validation profile

```bash
uv sync --locked --group validate
```

This adds HSSM and ONNX Runtime to the normal pipeline environment. The docs
environment is intentionally separate and cannot run these gates.

## Run all four gates

```bash
uv run python validation/validate_network.py \
  --onnx-path /path/to/run_uuid_lan_ddm__network.onnx \
  --model-name ddm \
  --network-type lan
```

The command writes `validation_report.json` next to the ONNX unless
`--report-path` is given. It prints one compact JSON line and exits non-zero
when a gate fails.

| Gate | Evidence |
| --- | --- |
| G1 `structure` | ONNX loads; all input dimensions are concrete; there is one input, one scalar output, and the input width matches the model |
| G2 `parity` | The exported ONNX matches the JAX trainer state when the required siblings exist |
| G3 `hssm_load` | HSSM accepts the likelihood and obtains a finite initial log probability |
| G4 `density` | Integrated mass is near one and Hellinger error is acceptable relative to a measured simulator sampling floor |

G2 legitimately skips for Torch artifacts and for a bare ONNX without the JAX
state/config pair. G1, G3, and G4 are required for publication. A skipped gate
may carry `passed: true` in the detailed report to mean it did not itself fail;
inspect the compact `gates` states rather than treating that as evidence the
check ran.

!!! danger "Only validate trusted artifact folders"

    The parity gate unpickles the sibling `*_network_config.pickle`. Unpickling
    can execute code. Validate folders you produced or fetched from a repository
    you control; do not point the command at arbitrary downloads.

## Use skips only for diagnosis

`--skip-hssm` and `--skip-density` shorten a diagnostic run, but do not create
publishable evidence. In particular, the publisher refuses any report where a
required gate is skipped or missing.

If the density comparison needs a reviewed tolerance change, pass
`--hellinger-ratio-max` and preserve the resulting threshold in the report.
Do not loosen it merely to turn one candidate green.

## Validate an auxiliary network (cpn, opn, gonogo)

Auxiliary networks are not densities, so the density gate does not apply to
them. They share G1 and G2 with a LAN and then run their own pair:

```bash
uv run python validation/validate_network.py \
  --onnx-path /path/to/run_uuid_opn_ddm__network.onnx \
  --model-name ddm \
  --network-type opn
```

`--model-name` is always the base model. Pass `ddm`, never `ddm_deadline`: the
deadline variant is derived internally wherever a simulation needs it, and the
command rejects a `_deadline` name outright.

| Gate | Evidence |
| --- | --- |
| G1 `structure` | As for a LAN, with the width contract `n_params + 1`: `[θ…, choice]` for a cpn, `[θ…, deadline]` for an opn or gonogo |
| G2 `parity` | As for a LAN |
| A3 `hssm_missing_load` | HSSM accepts the network as `loglik_missing_data` next to the base LAN and obtains a finite initial log probability on simulated data with missing rows, at `p_outlier = 0` and at HSSM's default lapse |
| A4 `accuracy` | The network's probability tracks a Monte-Carlo truth at 20 in-bounds parameter draws: `P(choice | θ)` for a cpn, `P(rt > deadline | θ)` for an opn |

The network's output must already be a log-probability (log-sigmoid baked into
the graph, every value ≤ 0). A raw-logit export fails A4 before any truth is
simulated.

A cpn must be told what it predicts with `--aux-category choice`; the flag is
the only record of the category the network was derived for, so it has no
default. An opn or gonogo defaults to `deadline`. The report carries the
resolved value as top-level `aux_category`.

A3 resolves the base LAN by model name through HSSM, which downloads it for
registry models. Pass `--lan-onnx` to use a local LAN instead; models outside
HSSM's registry require it. Either way the assembly is LAN + auxiliary net
(`loglik_kind="approx_differentiable"`), never HSSM's analytical likelihood.

!!! warning "Auxiliary reports do not promote yet"

    The publisher still requires the LAN gate set and does not pass
    `--aux-category`, so it refuses every auxiliary report, and
    `lan-publish --network-type cpn` cannot validate a cpn at all. Use this
    command to validate an auxiliary artifact; publishing it is a follow-up.

Two skips are built in and reported rather than silent:

- A gonogo network has no HSSM consumer, so A3 and A4 are `skipped` with the
  reason `HSSM has no gonogo consumer`. Only G1 and G2 judge it.
- The cpn A3 gate needs an HSSM that passes `response` to the network on
  missing rows (HSSM ≥ 0.6.0). Under an older HSSM it is `skipped` with the
  reason `HSSM < 0.6.0 ignores response on missing rows`, and activates by
  itself once the locked HSSM moves.

`--skip-accuracy` shortens a diagnostic run the way `--skip-density` does for a
LAN. The A4 thresholds (`--accuracy-mean-abs-max`, default `0.01`, and
`--accuracy-max-abs-max`, default `0.03`) are provisional: they are to be
calibrated once on the first `ddm_sdv` gate run and then frozen. Record a
changed threshold in the report; do not loosen it to pass one candidate.

!!! note "The cpn truth counts every trial"

    ssm-simulators does not censor a base model at `max_t`: a trial that has
    not terminated by then comes back at `rt ≈ max_t + t` with its
    sign-implied choice, while a derived cpn integrates only up to 20 s. A4
    judges the all-trials proportion and records the windowed one as
    `truth_rt_lt_max_t` on every draw, so the two can be compared where a
    visible share of trials outlasts the window.

## Inspect the result visually

The marimo inspector compares the network's implied likelihood with simulator
KDEs and shows a parameter manifold. It explains a gate result; it does not
replace one.

```bash
export INSPECT_ONNX="/path/to/run_uuid_lan_ddm__network.onnx"
export INSPECT_MODEL="ddm"
uv run --group inspect marimo edit validation/inspect_network.py
```

To produce a static local report without opening an editor:

```bash
uv run --group inspect marimo export html validation/inspect_network.py \
  -o inspection.html
```

If `validation_report.json` is present, the inspector displays it only when its
recorded ONNX filename matches the artifact being viewed.
