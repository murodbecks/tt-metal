# Chronos-2 Forecast

[Chronos](https://github.com/amazon-science/chronos-forecasting) pretrained time-series models. This tree vendors **Chronos-2** inference sources as a golden PyTorch reference for a later TTNN port.

## Setup

All commands are run from the tt-metal repo root, with the tt-metal build and
Python environment active (`ttnn` comes from that build, not PyPI).

**Install the requirements before running anything, or every test fails at
import.** `requirements.txt` installs the upstream `chronos-forecasting==2.3.2`
package, which the reference code imports and the golden tests compare against.
Once installed, `import chronos` works from anywhere in the environment, with
no path to set. The tiny dummy checkpoint the tests load is committed under
`tests/fixtures/`, so nothing else needs downloading for the tests that use it.

```bash
cd "$TT_METAL_HOME"   # or wherever your tt-metal checkout is
source python_env/bin/activate
uv pip install -r models/experimental/chronos_forecast/requirements.txt
```

Use `uv pip`, not bare `pip`: `python_env` is a uv environment with no `pip` of
its own, so `pip` can resolve to a different environment and install there.
If the package is missing or the wrong version, the tests stop with an error
that prints the command above.

The demo and the real-checkpoint accuracy tests also need the Chronos-2
weights; tests that need them skip or error without them:

```bash
uv pip install -U "huggingface_hub[cli]"
hf download amazon/chronos-2 --local-dir models/experimental/chronos_forecast/weights/chronos-2
```

`weights/` is gitignored. The TTNN tests and benchmarks need Blackhole hardware
(the paper-shape numbers below were measured on p150a cards).

Pinned Chronos-2 copy: commit `10afa9ebe016e514f9d7dc1aa873f66af57e116b`, checked
against the `chronos-forecasting==2.3.2` release. See [reference/PROVENANCE.md](reference/PROVENANCE.md).



## Demo

Single `Chronos2Model.forward` (CPU):

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py
```

Chronos-1 tokenizer only:

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py --tokenizer-only --context-length 16
```

## TTNN trace

The fixed paper benchmark (`batch=1024`, context `2048`, forecast `64`) has a
device-resident path and an address-stable TTNN trace runner. It keeps the
embeddings, 12-layer encoder, and output head on device, specializes group
attention when every series has a unique group ID, and refreshes fixed input
slots between replays.

[optimizations.md](optimizations.md) logs every optimization on the branch,
ranked by single-chip time saved, with code links, the hardware limit each one
addresses, and how data-parallel scaling holds up.

Accuracy and lifecycle:

```bash
source python_env/bin/activate
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_modules.py \
  models/experimental/chronos_forecast/tests/pcc/test_trace.py
```

Paper-shape performance (20 replay and 20 end-to-end iterations):

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -s
```

## Data parallel (4, 8, 16, 32 chips)

Opening `TtChronos` on a mesh device runs it data parallel, with no chip-to-chip
traffic:

- Weights are copied to every chip.
- `upload_inputs` splits the series along dim 0, and `postprocess_output`
  concatenates the per-chip outputs.
- Time attention stays within a series, and group attention stays within a
  group, so each chip runs the single-chip model on its share.
- Unique-group batches are padded with dummy series to a multiple of the chip
  count.
- Grouped batches are packed into whole blocks with the block count padded to a
  multiple of the chip count, so no group spans two chips.

The PCC, accuracy and trace tests are parametrized over `1`, `4`, `(1, 8)`,
`(2, 8)` and `(4, 8)`. Shapes larger than the machine are skipped.

```bash
PYTHONPATH=. pytest -s \
  models/experimental/chronos_forecast/tests/pcc/test_paper_shape_pcc.py \
  models/experimental/chronos_forecast/tests/pcc/test_trace.py
PYTHONPATH=. pytest -s \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -k data_parallel
```

`TtChronosTraceRunner.stream(items, prepare)` prepares batch i+1 on the host
while replay i runs, for back-to-back batches.

Paper shape (1024 series, performance precision, L1-resident, unique groups) on
p150a cards, at `86be1507afa`:

Note: benchmarks start with inputs on the host and are transferred over PCIe gen4.

| Chips | Series per chip | Replay | Serial end-to-end | Streamed per batch | Streamed series/s |
|---|---|---|---|---|---|
| 1 | 1024 | 260 ms | 322 ms | 311 ms | 3,290 |
| 4 | 256 | 69 ms | 113 ms | 97 ms | 10,570 |

On 4 chips, replay is 3.77× faster than one chip (94% scaling efficiency). With
groups of 4, replay is 106 ms (3.61×) and a streamed batch takes 129 ms. The
remaining host cost is serial readback (about 20 ms) and upload (about 9 ms).
Per-chip replay times from single-chip runs put 8, 16 and 32 chips at about
32, 16 and 9 ms, so from 16 chips up the host limits throughput unless readback
also overlaps the replay.

### After the optimization round (`10ec7ff655e`)

Same shape and configuration (`performance_l1`), measured at `10ec7ff655e` on
`chronos-forecast-experi`:

| Chips | Series per chip | Replay | Serial end-to-end | Streamed per batch | Streamed series/s |
|---|---|---|---|---|---|
| 1 | 1024 | 153 ms | 199 ms | 188 ms | 5,440 |
| 4 | 256 | 35 ms | 82 ms | 54 ms | 18,900 |

The changes since `86be1507afa`:

- Sweep-tuned L1 matmul configs, with the RMSNorm gamma folded into the
  following linears.
- LoFi attention matmuls.
- Model-local `generic_op` kernels in `ops/`: RoPE, a bank-local residual add,
  the QKV head split fused with RoPE, and an RMSNorm that writes bfloat8_b.
- The input embedding and the output head run inside the L1 chunks.

See [optimizations.md](optimizations.md) for each change's measured saving and
code references.

On 4 chips, replay is 4.36× one chip at the full batch. A chip holding 256
series is slightly more efficient per series than one holding 1024; against a
single chip at 256 series, 4 chips reach 99%.

With groups of 4:

| Chips | Replay | Streamed per batch | Replay speedup |
|---|---|---|---|
| 1 | 269 ms | 300 ms | 1× |
| 4 | 68 ms | 86 ms | 3.95× |

On 4 chips the host now limits streaming. The 35 ms replay is followed by a
serial readback of about 17 ms and an upload of about 3 ms. Host prepare
(about 17 ms) overlaps the replay. Single-chip replay at 128, 64 and 32 series
takes 17.5, 8.8 and 6.1 ms, which is roughly where 8, 16 and 32 chips would
land on replay alone.

Blackhole Galaxy bring-up:

- Run the commands above there. The perf report prints the worker grid.
- The L1 chunk budgets (`_L1_CHUNK_TOKENS_*` in `tt/program_configs.py`) were
  measured on the p150a's 11x10 grid. `TtChronos` logs a warning on any other
  grid; if you see it, re-measure the budgets on one chip with
  `sweeps/sweep_l1_chunk.py`.


## Fast path for univariate series

`TtChronos.forward` runs a separate, faster graph (`tt/fast_path.py`) for the most common call:
independent univariate series (no shared `group_ids`, no covariates), context up to 512 and up to 4
output patches. Everything else (groups, covariates, longer contexts) runs through the encoder
modules as before. The fast path also accepts missing values (NaN or `context_mask`) and runs data
parallel when `TtChronos` is built on a 1xN mesh. Nothing changes for callers:

```python
import ttnn
from models.experimental.chronos_forecast.tt.model import TtChronos
from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision

# two command queues and a trace region enable the traced 2-CQ path (it falls back without them)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 << 20, num_command_queues=2)
model = TtChronos.from_torch_model(mesh, reference, TtChronosPrecision.performance())  # bfp8_b weights
quantiles = model.forward(context, num_output_patches=4)  # (B, 512) -> (B, 21, 64)
model.uses_fast_path(context, num_output_patches=4)       # True; from_torch_model(..., fast_path=False) opts out
```

What it does differently:

- **Group attention folded away.** With one series per group each query sees only itself, so the
  sublayer is exactly `x + W_o W_v RMSNorm(x)`; the two linears are folded on the host into one.
- **Batch folded into M** for every matmul, activations block-sharded in L1, and per-shape program
  configs (1D/2D multicast, wide K blocks) for the large path; batch <= 4 keeps a separate
  latency graph.
- **Token packing.** At large batch up to six 37-token series share one padded sequence (block-diagonal
  SDPA mask), so padding to the tile size costs far less.
- **Fused split-heads + RoPE** op (`ops/split_rope`) on the large path.
- **Execution.** One metal trace per graph key with persistent inputs; multi-pass calls stream
  through two command queues and overlap host prep / readback with the replay.
- **Precision.** `TtChronosPrecision()` runs it in bf16: bf16 weights and activations, fp32
  accumulation, HiFi4 except HiFi2 for the large-path attention output, FF and SDPA. A preset with
  bf8 weights (e.g. `performance()`) runs it in bfp8_b: bfp8_b weights, HiFi2 matmuls, with LoFi for
  the folded group-attention linear and the FF down projection on the large path only.

Measured with `tests/perf/test_fast_path_perf.py` on Blackhole p150 (context 512, horizon 64,
host arrays in, host forecasts out, steady state):

| chips | precision | b1 | b4 | b64 | b256 | b1024 | b4096 |
|---|---|---|---|---|---|---|---|
| 1 | bf16 | 438/s, p50 2.29 ms | 1,168/s, 3.42 ms | 7,403/s | 8,314/s | 9,826/s | |
| 1 | bfp8_b | 485/s, p50 2.06 ms | 1,215/s, 3.29 ms | 7,870/s | 9,633/s | 11,736/s | |
| 4 | bf16 | | | 11,407/s | | 29,975/s | 35,591/s |
| 4 | bfp8_b | | | 11,950/s | | 32,738/s | 38,733/s |

Accuracy against the reference model (`tests/test_fast_path.py`): max per-row quantile NRMSE
<= 0.021 at both precisions for batch 1 to 1,024 on one and four chips (gate 0.04), including
missing values, short histories and constant series; traced repeats are bit-identical.

Same host, end-to-end timing at context 512 / horizon 64 against the encoder path through the trace
runner (`TtChronosTraceRunner`, performance preset at bfp8_b, default preset at bf16):

| | encoder path (trace runner) | fast path |
|---|---|---|
| p50 latency, batch <= 4, bfp8_b | 6.2 ms | 2.0 ms |
| 1 chip, b1024, bfp8_b / bf16 | 10,929-11,200 / 7,470-7,490 series/s | 11,877-12,048 / 9,964-10,066 series/s |
| 4 chips, b1024, bfp8_b / bf16 | 22,064 / 17,887 series/s | 33,762 / 29,997 series/s |
| 4 chips, b4096, bf16 | 22,496 series/s | 36,125 series/s |
| max row NRMSE vs FP32 reference, bfp8_b | 0.032 (0.040 on 4 chips) | 0.025 (0.036 on 4 chips) |
| device memory, bfp8_b | 141 MB | 111 MB |

Known limits: the fast path covers contexts up to 512 and horizons up to 64, the envelope it was
validated on; longer ones use the encoder path. One sample with extreme values (arcsinh stress)
exceeds 0.04 NRMSE at batch 1. Tested on Blackhole p150 only. At bfp8_b the encoder path uses about
5-10% less energy per series at b256 and b1024 on one chip (LoFi in every FF and attention matmul).

```bash
# accuracy, 1 chip and 1x4 mesh (needs weights/chronos-2)
pytest models/experimental/chronos_forecast/tests/test_fast_path.py
# timing tables above
pytest -s models/experimental/chronos_forecast/tests/perf/test_fast_path_perf.py
```

## Running the tests

Install the requirements from [Setup](#setup) first; without them the tests
fail at import.

From the tt-metal repo root:

```bash
source python_env/bin/activate
```

Core Python/unit tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/test_*.py \
  -s
```

TTNN module PCC tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_modules.py \
  -s
```

TTNN trace PCC tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_trace.py \
  -s
```

Eager paper-shape performance test:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward.py \
  -s \
  --timeout=3600
```

Trace-replay performance test (paper shape):

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -s \
  --timeout=3600
```

Single-chip, unique groups only:

```bash
PYTHONPATH=. pytest -s \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -k "performance_l1 and not groups"
```

`TtChronosTraceRunner` is TT Metal trace capture/replay, not a resident
persistent compute kernel. A true persistent kernel would require porting the
full transformer into unified device kernels under slow dispatch.

## References
- Paper: [Chronos](https://arxiv.org/abs/2403.07815)
- Chronos-2: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- Upstream: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
