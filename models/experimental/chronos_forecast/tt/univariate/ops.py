# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Chronos-2 TTNN port: constants, host prep / weight folding, the fused split + RoPE op and
the design notes of the device graph (model.py) and its execution (runtime.py).

Host (non-learned, FP32): instance norm + arcsinh, left NaN padding, patching,
patch features, masks, RoPE tables, final sinh/unscale and row slicing.
Device (all learned compute): input ResidualBlock, REG token, 12 encoder
blocks (time attention with RoPE, group attention, FFN), final RMSNorm and the
quantile ResidualBlock head.

Reference semantics: chronos-forecasting 2.3.2 `chronos/chronos2/{model,layers}.py`.
Group attention is specialised for the univariate mode where every row
is its own group: each query sees exactly one key (itself), so softmax == 1 and
the sublayer is exactly `x + W_o W_v RMSNorm(x)`, folded on the host into one
linear.

Execution: the device graph depends only on (B, S_pad) and the packed layout (below).
One metal trace is kept alive for the current graph key with persistent feature/mask
input buffers; a new key runs eagerly once (compiles programs, gives the result),
releases the old trace and captures a new one. CHRONOS2_TRACE=0 forces the eager path.
A traced call whose key mask equals the one already held in the persistent mask
buffer skips building, tilizing and uploading the SDPA mask (SDPA requires the
mask's query dim to match Q, so a [B, 1, 1, S_pad] broadcast mask is not accepted).
CHRONOS2_MASK_CACHE=0 always rebuilds and uploads it.

Batching and devices: `device` is a MeshDevice (1x1 on one card, 1xN for data
parallel). Weights are replicated to every card, each pass's rows are sharded
along dim 0 and outputs are concatenated back in row order. One device pass
holds at most _MAX_DEVICE_BATCH series per card (_MAX_DEVICE_BATCH_SHARDED on the
sharded large-M path below; CHRONOS2_MAX_BATCH overrides): the blocked matmul
configs keep a per-core output block of Mt / grid_y tile rows in L1, which
overflows beyond that. Larger calls run sequential passes of that
fixed size through the same trace; the last pass and any batch not divisible by
the card count are padded by repeating the last row, and padded rows are
dropped. Rows never interact, so padding does not change the real rows.

Multi-pass streaming: with two command queues (DEVICE_OPTIONS num_command_queues=2)
the passes after the first stream through device staging buffers allocated with
the trace: CQ1 writes pass i+1's features (and its mask, when it changed) into
staging buffers and reads back pass i's output copy, while CQ0 copies the staging
buffers into the trace inputs, replays the trace and copies its output into one
of two output staging buffers; record/wait events order the queues. Single-pass
calls (B <= _MAX_DEVICE_BATCH per card, including B <= 4) keep the single-queue
path. CHRONOS2_2CQ=0 disables streaming. CHRONOS2_HOST_OVERLAP=1 (opt-in; measured
and rejected as default) instead overlaps only the host tilize of pass i+1 with
the replay of pass i on one queue; otherwise passes run serially.

Host-device pipeline (token-packed path only, so never at B <= 4; CHRONOS2_PIPELINE=0
disables it): the call is cut into chunks of one packed pass (k * Bp series). Chunk i+1's
host prep (instance norm, arcsinh, patching, packing, tilize) runs while pass i replays;
once the trace is live every pass, the first included, is enqueued on the 2-CQ stream
without blocking, and chunk i's readback, sinh and unscale run while pass i+1 replays.
Every host step is per row and padding repeats the same rows as the unpipelined path, so
the output is the same. Without a live 2-CQ trace (new graph key) pass 0 runs blocking
(eager run + capture) and the rest stream.

Row-major feature upload (packed v2 path only, so never at B <= 4; on by default only on
a multi-card mesh, num_devices > 1; CHRONOS2_RM_UPLOAD=auto|1|0, 1 forces it on one card
too, 0 disables it): each pass's features are converted on the host to row-major bf16 (no
host tilize) and tilized on the device by to_layout at the start of the sharded graph,
inside the captured trace, so the persistent trace input and the 2-CQ staging input are
row-major. The embed activations are bf16 in every preset, so the device sees the same
values (A/B outputs bitwise equal). On a 1x4 mesh the host tilize of the sharded features
was a large part of the end-to-end time (b4096: device replays 64% of it);
same-process A/B +19% / +28% items/s at b1024 / b4096 on 1x4. One
card is device-bound and the extra on-device tilize lost 1-4% there, so a single card
keeps the host tilize by default.

RoPE: by default the QKV output is split into heads by
split_query_key_value_and_split_heads and q/k are rotated by
ttnn.experimental.rotary_embedding with a shared [1, 1, S_pad, d_kv] cos/sin cache.
CHRONOS2_SPLIT_ROPE=1 (opt-in; measured and rejected as default: faster in
scratch A/B but bringup max NRMSE +50% on the masked-tail case, docs/PERF_ROUND2.md)
instead splits and rotates with one custom ttnn.generic_op (kernels in
models/experimental/chronos2/tt/kernels, found by a path relative to this file):
with d_kv = 64 = two tiles, HF rotate_half is a swap of the two tiles, so
q/k = x * cos + swap(x) * (-sin, sin) per tile, and v is copied through. It is
used only on one card with bf16 attention activations and d_kv == 64 when the
kernel files are found. CHRONOS2_ROPE_FUSED=0 restores
the earlier path: rotate_half folded into a second "rotated" QKV weight and
applied as qkv * cos + rot * sin (one extra linear and three elementwise ops
per layer).

Matmul program configs: every encoder linear (QKV, attention output, folded
group attention, FF up with ReLU fused, FF down) uses a 2D multicast config with
the batch folded into M (fuse_batch) on a grid sized to divide the tile counts.
At small M that 2D grid is only (divisor of Nt) x (divisor of Mt) cores, e.g.
12 x 2 = 24 cores for QKV and FF up at Mt = 2, so when a 1D in0-multicast config
(all of M per core, Nt / per_core_N cores) covers more cores it is used instead
(CHRONOS2_MM_WIDE=0 keeps the 2D config). The wide-K, small-N ResidualBlock
output linears (embed wo 3072->768, head wo 3072->352) use a 1D in0-multicast
config: one output tile column per core over all of M. At large M (Mt >
_MAX_1D_MT, i.e. more than 8 series per card at S_pad 64; never at B <= 4) the
1D config cannot hold M per core, and the embed / head linears (wh, wo, wr) get
a blocked 2D multicast config sized from the padded shapes, with bias and ReLU
fused and the per-core L1 output / in0 blocks bounded by out_block_h
(_MAX_2D_BLOCK_TILES). Under the TTNN auto config these ran at ~2 TFLOP/s and
took ~25% of a 32-series pass; the blocked config measured +56% / +48% items/s
at B = 256 / 1024. CHRONOS2_MM_CONFIG=0
falls back to the TTNN auto-selected configs.

Large-M head configs (sharded large-M path only, B > 4; CHRONOS2_HEAD_LARGE=0 disables
it): on packed passes the head runs on the future rows only, Mt = 8 .. 16 per card (not
above _MAX_1D_MT), where wh / wr took the TTNN auto config (104 / 11 cores) and wo the 1D
config (11 cores, one output column each): 6.9% (bfp8_b) / 6.3% (bf16) of b1024 device
time at 1-5% FPU (device profile). From Mt >= _HEAD_LARGE_MIN_MT the
three head linears use the blocked 2D config above (12 x 8 cores for wh, 11 x 8 for wr /
wo at Mt 16), same dtypes and HiFi4 fidelity. The latency path (B <= 4) never uses it.

Sharded large-M path (per-card Mt > _MAX_1D_MT, bf16 attention activations and
residual, d_kv == 64, split + RoPE kernels present; never at B <= 4;
CHRONOS2_SHARD_LARGE=0 disables it): passes of up to _MAX_DEVICE_BATCH_SHARDED
series per card, padded so Mt is a multiple of 8. The encoder residual stream is
BLOCK sharded in L1 on gx x gy cores (gx = largest divisor of the d_model tile
count <= grid x, gy = largest divisor of Mt <= grid y); the residual adds and
rms_norm run sharded (LayerNormShardedMultiCoreProgramConfig); every encoder linear
uses a 2D multicast fuse-batch config on that grid (sharded in0 with in0_block_w =
shard width, sharded output for the d_model-wide outputs, L1 interleaved output
with out_block_h bounded to _MAX_2D_BLOCK_TILES for QKV and the FF hidden); q/k/v
come from the fused split + RoPE kernel and q/k/v, SDPA and the head concat stay
in L1. Measured +42.8% / +44.4% items/s at B = 256 / 1024 over the interleaved
32-series path, NRMSE vs that path 1.1e-3 / 1.2e-3 (128-series passes overflow L1 circular buffers). On a 1xN mesh every card runs this
single-card path on its own batch shard (weights replicated, no chip-to-chip traffic):
the path choice, Mt padding, shard / program configs and token-packing pass sizes are
computed from the per-card batch.
Wide K blocks for QKV and FF up (sharded large-M path only; CHRONOS2_LARGE_IBW=0 restores
the sharded norm feeding them): the norm before QKV and before FF up converts the sharded
residual to L1 interleaved and runs rms_norm there, so the two linears read an interleaved
in0 with the widest K block whose per-core in0 block fits _LARGE_IBW_IN0_TILES (in0_block_w
6 at b1024 instead of the 4-tile shard width). Op probes at the b1024 pass shapes (outputs bitwise equal): sharded_to_interleaved 8.6 us + interleaved
rms_norm 16.8 us vs sharded rms_norm 21.9 us; QKV 86.8 -> 73.7 us, FF up 112.4 -> 94.8 us
per layer (device profile: QKV + FF up 33% of b1024 device time).

Large-path SDPA (sharded large-M path only, B > 4; CHRONOS2_LARGE_SDPA=0 restores the
TTNN default chunking at the HiFi4 / fp32 config): one q / k chunk per (packed) sequence
(largest multiple of 32 dividing S_pad, <= _LARGE_SDPA_MAX_CHUNK) at HiFi2 without fp32
dest accumulation. With no program config SDPA was 40% of a packed 96-series bfp8_b pass
(374 us / layer, device profile). Same-process A/B on the packed path: bfp8_b +17% items/s at B = 256 and 1024, B = 256 max row
NRMSE vs FP64 0.0152 (base 0.0157); bf16 +22% at B = 256, NRMSE 0.0133 (base 0.0120).
LoFi failed (NRMSE 0.042; on the packed bfp8_b path at B = 1024, max row NRMSE vs FP64
0.0807 vs 0.0357 base).

Token packing (sharded large-M path only, B > 4; CHRONOS2_PACK=0 disables it): a
series has S = ctx patches + REG + future patches tokens (37 at ctx 512 / horizon 64),
padded to S_pad = 64, so 42% of every encoder matmul row is padding. Packing puts k
series into one attention sequence of Sp = round_up(k * S, 32) rows (k = 6, Sp 224 at
S 37) with a block-diagonal SDPA mask: each series' queries see only its own valid keys
and the trailing pad rows form their own block. The device graph is the sharded graph
at (B', Sp), passes of up to _PACK_MAX_MT tile rows per card. It is used only when it
cuts the padded rows of the call to <= _PACK_MAX_ROW_RATIO of the unpacked rows
(B >= ~128 at S 37). Measured b1024 +16% items/s (4542 -> 5280) and b256 max NRMSE vs
FP64 0.0120 vs 0.0168 unpacked.
Series per sequence: k is the one that gives the call the fewest padded rows (rows x
(1 + Sp / 5000) for the SDPA cost growing with Sp), but the density choice (_choose_pack)
is kept unless the fewest-rows k saves more than _PACK_SWITCH. At S 37 this also packs
small calls (b16: k 2, Sp 96; b64: k 4, Sp 160; b256 / b1024 keep k 6, Sp 224). Same-process
A/B vs density-only packing: bfp8_b +22.6% / +58% items/s at
B = 16 / 64, bf16 +19.5% / +64%; max row rel rms vs that
path <= 9.6e-3.
Packed layout v2 (default; CHRONOS2_PACK_V2=0 restores v1, series laid out one after
another): the k * H future tokens of the k series come first (rows 0 .. k*H-1, segment
j's horizon h at row j*H + h), then each series' context + REG tokens; every row carries
its position within its own series in the per-row RoPE cos/sin table (RoPE scores depend
only on position differences inside a series, and the mask keeps series apart), so the
result equals per-series attention. Two consequences: (1) the quantile head runs on, and
the host reads back, only the first round_up(k*H, 32) rows of each sequence (32 of 224 at
S 37: 7x less head work, output bytes and host untilize); (2) the block-diagonal part of
the mask is constant for the layout, so it lives on the device ([B', 1, Sp, Sp], built once
per trace) and each pass uploads only a per-key row [B', 1, 1, Sp] (0 / -inf, one tile row
instead of Sp / 32) that one broadcast add combines with it inside the graph. Measured
before the change (bfp8_b): b256 42.8 ms end to end vs 23.7 ms
of device replays; per pass the host mask build took ~4 ms and the output readback 1.5 ms.

Activation placement: the [M, d_model] tensors each encoder sublayer re-reads
are L1 interleaved: the residual stream (embed add and the 3 adds per layer),
the per-layer rms_norm outputs and the wo / wg / FF-down outputs. QKV, the
heads, SDPA, the FF-up hidden, the final norm and the head stay DRAM
interleaved. At the 32-series pass (M = 2048 rows) this measured 5.8% (bfp8_b)
and 7.7% (bf16) lower forecast latency, bitwise equal output
(docs/PERF_ROUND2.md). CHRONOS2_ACT_L1=0 keeps every activation in DRAM.

Matmuls, RMSNorm and SDPA run with fp32 dest accumulation at HiFi4, except the
encoder linears under the bfp8_b preset, which default to HiFi2 (groups qkv, wo,
wg, ff_up, ff_down): with 8-bit block-float weights HiFi2 measured no accuracy
loss and 2-5% lower latency at B >= 4 (docs/PERF.md). Under bf16 weights a HiFi2
sweep, a block-sharded rms_norm and a K-heavy 1D config for FF down were measured
and rejected at small batch. CHRONOS2_MM_FIDELITY (e.g. "ff_down=HiFi2,ff_up=LoFi")
replaces the preset's per-group fidelity table under any precision mode; groups it
leaves unset run at HiFi4, so CHRONOS2_MM_FIDELITY="" restores all-HiFi4.
On the sharded large-M path only (B > 4, compute-bound encoder linears) the bf16
preset runs wo, ff_up and ff_down at HiFi2 (_DEFAULT_LARGE_MM_FIDELITY, applied over
the table above): same-process A/B +6.2% / +7.0% items/s at B = 256 / 1024, B = 256
max row NRMSE vs FP64 0.01204 vs 0.01200 (all five groups at HiFi2 gave +9.5% / +11.5% at max NRMSE 0.01503 and is not the default).
The bfp8_b preset runs wg and ff_down at LoFi there (the other groups stay HiFi2): perf
suite on the wide-K-block graph vs HiFi2: J/item b256 / b1024
0.01433 / 0.01211 vs 0.01614 / 0.01261, items/s 9956 / 12437 vs 9849 / 12071, max row
NRMSE 0.0249 vs 0.0233 (on the earlier sharded-norm graph the same pair gave 0.0337). LoFi on ff_up, qkv, wo or SDPA raised NRMSE further (0.0469 ff_up + ff_down,
0.0674 qkv, 0.0435 wo, 0.0807 SDPA) and is not used.
CHRONOS2_LARGE_MM_FIDELITY (same syntax) replaces that large-path table; "" keeps
the small-batch table there. B <= 4 calls never use it.
CHRONOS2_W_BFP8 (opt-in, e.g. "ff_up,ff_down") stores the weights of the listed
encoder linear groups as BFLOAT8_B, and CHRONOS2_W_BFP4 (same syntax) as
BFLOAT4_B; a group may appear in only one of the two. Activations stay in the
precision mode's dtype and unset groups keep the mode's weight dtype.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

import ttnn

TILE = 32
NEG_INF = float("-inf")

# ~130 ops per forward; a captured trace needs well under 8 MiB.
# CQ0 runs compute; CQ1 streams multi-pass inputs/outputs.
DEVICE_OPTIONS = {"trace_region_size": 32 * 1024 * 1024, "num_command_queues": 2}

_DTYPES = {
    "fp32": ttnn.float32,
    "bf16": ttnn.bfloat16,
    "bfp8_b": ttnn.bfloat8_b,
    "bfp4_b": ttnn.bfloat4_b,
}

_MAX_IN0_BLOCK_W = 8
# 1D in0-mcast: every core holds all of M, so keep M and the in0 block small.
_MAX_1D_MT = 16
_MAX_1D_IN0_BLOCK_W = 32
_MAX_1D_IN0_TILES = 64
# large-M 2D mcast (embed / head linears): per-core output and in0 L1 blocks <= this many tiles
_MAX_2D_BLOCK_TILES = 64
# sharded large-M path, QKV / FF up on an L1-interleaved norm output: per-core in0 block <= this many tiles
_LARGE_IBW_IN0_TILES = 96
# sharded large-M path head: blocked 2D configs from this many tile rows per card (packed passes)
_HEAD_LARGE_MIN_MT = 8
# series per card per device pass (S_pad 64: 64 FF-up blocks overflow L1 at B=64, 1.68 MB > 1.5 MB)
_MAX_DEVICE_BATCH = 32
# sharded large-M path: series per card per pass (128 overflows L1 circular buffers at the wo matmul)
_MAX_DEVICE_BATCH_SHARDED = 64
# sharded large-M path: passes are padded so Mt is a multiple of this many tile rows
_SHARD_MT_STEP = 8
# latency calls (at most this many series per pass) always take the interleaved path,
# at any S_pad: never the sharded large-M path, its pass padding or token packing
_LATENCY_MAX_BATCH = 4
# token packing: at most this many series per packed sequence, packed sequences of at most
# _PACK_MAX_SP rows, passes of at most _PACK_MAX_MT tile rows (16 sequences of 224 rows measured)
_PACK_MAX_K = 8
_PACK_MAX_SP = 256
_PACK_MAX_MT = 112
# token packing is used only when it cuts the call's padded rows to at most this fraction
_PACK_MAX_ROW_RATIO = 0.75
# token packing: the fewest-padded-rows k replaces the density choice only when it saves more than this
_PACK_SWITCH = 0.10
# sharded large-M path SDPA: q / k chunk = the whole (packed) sequence up to this many rows
_LARGE_SDPA_MAX_CHUNK = 256

# encoder linear groups that CHRONOS2_MM_FIDELITY / CHRONOS2_W_BFP8 / CHRONOS2_W_BFP4 may override
MM_GROUPS = ("qkv", "wo", "wg", "ff_up", "ff_down")
_FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
# per-preset default encoder linear fidelity (CHRONOS2_MM_FIDELITY, if set, replaces it)
_DEFAULT_MM_FIDELITY = {
    "bfp8_b": "qkv=HiFi2,wo=HiFi2,wg=HiFi2,ff_up=HiFi2,ff_down=HiFi2",
}
# per-preset fidelity overrides on the sharded large-M path only (B > _LATENCY_MAX_BATCH),
# applied over the table above (CHRONOS2_LARGE_MM_FIDELITY, if set, replaces it)
_DEFAULT_LARGE_MM_FIDELITY = {
    "bf16": "wo=HiFi2,ff_up=HiFi2,ff_down=HiFi2",
    "bfp8_b": "wg=LoFi,ff_down=LoFi",
}

# fused split-heads + RoPE kernels, in kernels/ next to this file
_SPLIT_ROPE_KERNELS = ("reader_split_rope.cpp", "writer_split_rope.cpp", "compute_split_rope.cpp")


def _split_rope_kernel_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
    return d if all(os.path.isfile(os.path.join(d, k)) for k in _SPLIT_ROPE_KERNELS) else None


def _round_up(x, m):
    return (x + m - 1) // m * m


def _default_use_trace():
    return os.environ.get("CHRONOS2_TRACE", "1") != "0"


def _num_command_queues(device):
    """2 when the device was opened with a second command queue, else 1. The device API
    exposes no queue count, so this records an (empty) event on CQ1."""
    try:
        ttnn.record_event(device, 1)
    except Exception:  # noqa: BLE001 - opened with one queue
        return 1
    return 2


def _parse_fidelity_ladder(spec, env="CHRONOS2_MM_FIDELITY"):
    """'group=Fidelity,...' -> {group: Fidelity}; empty spec -> {} (all HiFi4)."""
    out = {}
    for item in filter(None, (s.strip() for s in (spec or "").split(","))):
        group, _, fid = item.partition("=")
        group, fid = group.strip(), fid.strip()
        if group not in MM_GROUPS or fid not in _FIDELITIES:
            raise ValueError(f"{env}: bad entry {item!r} " f"(groups {MM_GROUPS}, fidelities {_FIDELITIES})")
        out[group] = fid
    return out


def _parse_groups(spec, env="CHRONOS2_W_BFP8"):
    """'group,...' -> frozenset of encoder linear groups named in `env`; empty -> none."""
    out = set()
    for group in filter(None, (s.strip() for s in (spec or "").split(","))):
        if group not in MM_GROUPS:
            raise ValueError(f"{env}: bad group {group!r} (groups {MM_GROUPS})")
        out.add(group)
    return frozenset(out)


def _largest_divisor_at_most(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _subblock(block_h, block_w, max_tiles):
    """Largest (h, w) output subblock; h > 1 only when w spans the whole block."""
    best = (1, 1)
    for h in range(1, max_tiles + 1):
        for w in range(1, max_tiles // h + 1):
            if block_h % h or block_w % w or (h > 1 and w != block_w):
                continue
            if (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def _in0_block_w_1d(Mt, Kt):
    """Largest in0 block width dividing Kt with the per-core in0 block <= _MAX_1D_IN0_TILES tiles."""
    return max(d for d in range(1, min(Kt, _MAX_1D_IN0_BLOCK_W) + 1) if Kt % d == 0 and Mt * d <= _MAX_1D_IN0_TILES)


def _pad_rows(a, n):
    """Append n copies of the last row (n >= 0)."""
    return a if n == 0 else np.concatenate([a, np.repeat(a[-1:], n, axis=0)], axis=0)


def _choose_pack(S):
    """(k, Sp): series per packed sequence and its padded length, by row density k * S / Sp
    discounted for the SDPA cost growing with Sp; k == 1 means packing does not apply."""
    best = (0.0, 1, _round_up(S, TILE))
    for k in range(2, _PACK_MAX_K + 1):
        Sp = _round_up(k * S, TILE)
        if Sp > _PACK_MAX_SP:
            break
        score = k * S / Sp / (1 + Sp / 5000)
        if score > best[0]:
            best = (score, k, Sp)
    return best[1], best[2]


def _pack_layout_v2(k, S, H, Sp):
    """Packed layout v2 (module docstring): for packed row r < k * S, the series slot J[r] and the
    token index T[r] within that series (futures first, then context + REG blocks); seg [Sp]
    (series slot per row, k for the trailing pad rows) and pos [Sp] (RoPE position per row)."""
    n1 = S - H
    J = np.concatenate([np.repeat(np.arange(k), H), np.repeat(np.arange(k), n1)])
    T = np.concatenate([np.tile(np.arange(n1, S), k), np.tile(np.arange(n1), k)])
    seg = np.full(Sp, k, np.int64)
    seg[: k * S] = J
    pos = np.zeros(Sp, np.float32)
    pos[: k * S] = T
    return {
        "key": ("v2", k, S, H, Sp),
        "J": J,
        "T": T,
        "seg": seg,
        "pos": pos,
        "rows": _round_up(k * H, TILE),
        "k": k,
        "H": H,
    }


def _load_state_dict(weights_path):
    """A checkpoint directory or .safetensors file, or an already loaded state dict."""
    if isinstance(weights_path, dict):
        return {k: v.detach().to(torch.float32) for k, v in weights_path.items()}
    from safetensors.torch import load_file

    path = weights_path
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    return {k: v.to(torch.float32) for k, v in load_file(path).items()}


# ----------------------------------------------------------------------------
# Fused split-heads + RoPE (custom generic_op)
# ----------------------------------------------------------------------------
def _tile_cb(index, num_pages, cores):
    page = 2 * TILE * TILE  # bf16 tile
    return ttnn.CBDescriptor(
        total_size=num_pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page)],
    )


def split_rope(
    qkv, cos, sin, *, num_heads, kernel_dir, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """q, k, v [B, H, S_pad, 64] bf16 from qkv [B, S_pad, 3 * H * 64] bf16 (tile layout), with HF RoPE
    applied to q and k. cos / sin: [1, 1, S_pad, 64] bf16. A work unit is (batch, tile row, head);
    units are split evenly over the compute grid. The cos / sin tiles of a tile row are shared by its
    heads, so each core reads them once per run of consecutive units on the same tile row and the
    compute kernel keeps them in its circular buffer until the row changes (reuse within a core, not
    a NoC multicast)."""
    d_kv = 2 * TILE
    width = tuple(qkv.padded_shape)[-1]
    if width != 3 * num_heads * d_kv or qkv.dtype != ttnn.bfloat16:
        raise ValueError(f"split_rope: needs bf16 qkv of width 3 * {num_heads} * {d_kv}, got {width} {qkv.dtype}")
    wt = width // TILE
    rows = math.prod(qkv.padded_shape) // (TILE * width)
    s_pad = int(tuple(cos.padded_shape)[-2])
    rt = s_pad // TILE
    batch = rows // rt
    device = qkv.device()
    shape = ttnn.Shape([batch, num_heads, s_pad, d_kv])
    q, k, v = (
        ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config) for _ in range(3)
    )
    _, cores, group_1, _, n1, n2 = ttnn.split_work_to_cores(
        device.compute_with_storage_grid_size(), rows * num_heads, True
    )
    g1 = group_1.num_cores()
    work, start = [], 0
    for i, core in enumerate(ttnn.corerange_to_cores(cores, row_wise=True)):
        n = n1 if i < g1 else n2
        work.append(((core.x, core.y), n, start))
        start += n
    xargs = list(ttnn.TensorAccessorArgs(qkv).get_compile_time_args())
    targs = list(ttnn.TensorAccessorArgs(cos).get_compile_time_args())
    oargs = list(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    xa, ca, sa = qkv.buffer_address(), cos.buffer_address(), sin.buffer_address()
    qa, ka, va = q.buffer_address(), k.buffer_address(), v.buffer_address()
    ck = compute_kernel_config
    fid = ck.math_fidelity if ck is not None else ttnn.MathFidelity.HiFi4
    acc = ck.fp32_dest_acc_en if ck is not None else True

    def kern(name, ct, rt_args, config):
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(kernel_dir, name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt_args,
            config=config,
        )

    kernels = [
        kern(
            "reader_split_rope.cpp",
            [rt, num_heads, wt, len(xargs), *xargs, *targs],
            [(c, [xa, ca, sa, n, s]) for c, n, s in work],
            ttnn.ReaderConfigDescriptor(),
        ),
        kern(
            "writer_split_rope.cpp",
            [rt, num_heads, *oargs],
            [(c, [qa, ka, va, n, s]) for c, n, s in work],
            ttnn.WriterConfigDescriptor(),
        ),
        kern(
            "compute_split_rope.cpp",
            [rt, num_heads],
            [(c, [n, s]) for c, n, s in work],
            ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=acc),
        ),
    ]
    # c0: q/k tile pairs, c1: cos/sin tiles (c0 c1 s0 s1), c2: v pass-through, c16: rotated out, c24: cross terms
    cbs = [
        _tile_cb(0, 4, cores),
        _tile_cb(1, 8, cores),
        _tile_cb(2, 4, cores),
        _tile_cb(16, 4, cores),
        _tile_cb(24, 2, cores),
    ]
    ttnn.generic_op([qkv, cos, sin, q, k, v], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return q, k, v


class Chronos2Config:
    """Architecture values read from the checkpoint config (never hardcoded)."""

    def __init__(self, cfg: dict):
        cc = cfg["chronos_config"]
        self.d_model = int(cfg["d_model"])
        self.d_ff = int(cfg["d_ff"])
        self.d_kv = int(cfg["d_kv"])
        self.num_heads = int(cfg["num_heads"])
        self.num_layers = int(cfg["num_layers"])
        self.eps = float(cfg["layer_norm_epsilon"])
        self.rope_theta = float(cfg.get("rope_theta", 10000.0))
        self.act = cfg.get("dense_act_fn", "relu")
        self.reg_token_id = int(cfg.get("reg_token_id", 1))
        self.patch = int(cc["input_patch_size"])
        self.stride = int(cc["input_patch_stride"])
        self.out_patch = int(cc["output_patch_size"])
        self.max_output_patches = int(cc["max_output_patches"])
        self.context_length = int(cc["context_length"])
        self.time_scale = float(cc.get("time_encoding_scale", cc["context_length"]))
        self.quantiles = list(cc["quantiles"])
        self.use_arcsinh = bool(cc.get("use_arcsinh", False))
        self.use_reg_token = bool(cc.get("use_reg_token", False))
        if self.act != "relu":
            raise NotImplementedError(f"activation {self.act!r} not supported")
        if self.patch != self.stride or self.patch != self.out_patch:
            raise NotImplementedError("only non-overlapping patches with equal in/out size are supported")

    @classmethod
    def from_any(cls, config, weights_path):
        if config is None or (isinstance(config, dict) and "chronos_config" not in config):
            p = weights_path if os.path.isdir(weights_path) else os.path.dirname(weights_path)
            with open(os.path.join(p, "config.json")) as f:
                config = json.load(f)
        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)
        return cls(config)


# ----------------------------------------------------------------------------
# Host preprocessing (FP32, no learned weights)
# ----------------------------------------------------------------------------
def prepare_inputs(cfg: Chronos2Config, past_values, past_observed_mask, prediction_length):
    """Returns (features [B,S_pad,F_pad] f32, key_mask [B,S_pad] bool, loc, scale, n_ctx, n_out, S)."""
    v = np.asarray(past_values, dtype=np.float32)
    m = np.asarray(past_observed_mask, dtype=np.float32)
    if v.ndim != 2 or v.shape != m.shape:
        raise ValueError("past_values/past_observed_mask must both be [B, T]")
    if v.shape[1] > cfg.context_length:
        v = v[:, -cfg.context_length :]
        m = m[:, -cfg.context_length :]
    B, T = v.shape
    obs = m > 0
    x = np.where(obs, v, np.nan).astype(np.float32)
    cnt = obs.sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        loc = np.where(cnt > 0, np.nansum(x, -1, keepdims=True) / np.maximum(cnt, 1), 0.0).astype(np.float32)
        var = np.nansum((x - loc) ** 2, -1, keepdims=True) / np.maximum(cnt, 1)
        scale = np.where(cnt > 0, np.sqrt(var), 1.0).astype(np.float32)
    scale = np.where(scale == 0, np.float32(1e-5), scale).astype(np.float32)
    z = (x - loc) / scale
    if cfg.use_arcsinh:
        z = np.arcsinh(z)
    P = cfg.patch
    pad = (-T) % P
    z = np.concatenate([np.full((B, pad), np.nan, np.float32), z.astype(np.float32)], axis=1)
    om = np.concatenate([np.zeros((B, pad), np.float32), obs.astype(np.float32)], axis=1)
    N = (T + pad) // P
    zp = z.reshape(B, N, P)
    mp = om.reshape(B, N, P)
    zp = np.where(mp > 0, zp, 0.0).astype(np.float32)
    ctx_time = (np.arange(-N * P, 0, dtype=np.float32) / np.float32(cfg.time_scale)).reshape(N, P)

    H = int(math.ceil(prediction_length / cfg.out_patch))
    H = min(H, cfg.max_output_patches)
    fut_time = (np.arange(0, H * P, dtype=np.float32) / np.float32(cfg.time_scale)).reshape(H, P)

    n_reg = 1 if cfg.use_reg_token else 0
    S = N + n_reg + H
    S_pad = _round_up(S, TILE)
    F = 3 * P
    F_pad = _round_up(F + 1, TILE)  # +1 column: REG indicator
    feats = np.zeros((B, S_pad, F_pad), np.float32)
    feats[:, :N, 0:P] = ctx_time[None]
    feats[:, :N, P : 2 * P] = zp
    feats[:, :N, 2 * P : 3 * P] = mp
    if n_reg:
        feats[:, N, F] = 1.0
    feats[:, N + n_reg : S, 0:P] = fut_time[None]
    key_mask = np.zeros((B, S_pad), bool)
    key_mask[:, :N] = mp.sum(-1) > 0
    key_mask[:, N:S] = True
    return feats, key_mask, loc, scale, N, H, S


def token_counts(cfg: Chronos2Config, T, prediction_length):
    """(n_ctx, n_out, S) that prepare_inputs returns for a context of T steps, without the data."""
    T = min(int(T), cfg.context_length)
    N = -(-T // cfg.patch)
    H = min(int(math.ceil(prediction_length / cfg.out_patch)), cfg.max_output_patches)
    return N, H, N + (1 if cfg.use_reg_token else 0) + H


def rope_tables(cfg: Chronos2Config, S_pad, positions=None):
    """cos/sin [S_pad, d_kv] in FP32 (HF default RoPE, rotate_half layout); row r at position
    positions[r] (default r)."""
    d = cfg.d_kv
    inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.int64).astype(np.float32) / d))
    pos = np.arange(S_pad, dtype=np.float32) if positions is None else np.asarray(positions, np.float32)
    freqs = np.outer(pos, inv_freq.astype(np.float32)).astype(np.float32)
    emb = np.concatenate([freqs, freqs], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


# ----------------------------------------------------------------------------
# Host weight folding (FP32/FP64, exact algebra)
# ----------------------------------------------------------------------------
def _rotate_half_matrix(d):
    """R such that x @ R == rotate_half(x) for a row vector x of length d."""
    R = np.zeros((d, d), np.float64)
    h = d // 2
    for j in range(h):
        R[j + h, j] = -1.0
        R[j, j + h] = 1.0
    return R


def fold_weights(cfg: Chronos2Config, sd: dict):
    """Folded host weights in (in, out) layout, float64 numpy."""
    f64 = lambda k: sd[k].double().numpy()
    D, Hh, dk = cfg.d_model, cfg.num_heads, cfg.d_kv
    inner = Hh * dk
    R = np.kron(np.eye(Hh), _rotate_half_matrix(dk))
    W = {}

    # input patch embedding with REG folded in through an indicator feature
    P = cfg.patch
    F = 3 * P
    F_pad = _round_up(F + 1, TILE)
    pe = "input_patch_embedding."
    wh = np.zeros((F_pad, cfg.d_ff))
    wh[:F] = f64(pe + "hidden_layer.weight").T
    bh = f64(pe + "hidden_layer.bias")
    wr = np.zeros((F_pad, D))
    wr[:F] = f64(pe + "residual_layer.weight").T
    bo = f64(pe + "output_layer.bias") + f64(pe + "residual_layer.bias")
    if cfg.use_reg_token:
        # REG row: hidden = relu(b_h - big) == 0, out = bo + wr[F] == shared[reg]
        wh[F] = -(np.abs(bh).max() + 1.0) * 64.0
        wr[F] = f64("shared.weight")[cfg.reg_token_id] - bo
    W["emb_wh"], W["emb_bh"] = wh, bh
    W["emb_wo"], W["emb_bo"] = f64(pe + "output_layer.weight").T, bo
    W["emb_wr"] = wr

    layers = []
    for i in range(cfg.num_layers):
        p = f"encoder.block.{i}.layer."
        g0 = f64(p + "0.layer_norm.weight")[:, None]
        wq = g0 * f64(p + "0.self_attention.q.weight").T
        wk = g0 * f64(p + "0.self_attention.k.weight").T
        wv = g0 * f64(p + "0.self_attention.v.weight").T
        wqkv = np.concatenate([wq, wk, wv], 1)
        wrot = np.concatenate([wq @ R, wk @ R, np.zeros((D, inner))], 1)
        wo = f64(p + "0.self_attention.o.weight").T
        g1 = f64(p + "1.layer_norm.weight")[:, None]
        wg = (g1 * f64(p + "1.self_attention.v.weight").T) @ f64(p + "1.self_attention.o.weight").T
        g2 = f64(p + "2.layer_norm.weight")[:, None]
        wi = g2 * f64(p + "2.mlp.wi.weight").T
        wff = f64(p + "2.mlp.wo.weight").T
        layers.append(dict(wqkv=wqkv, wrot=wrot, wo=wo, wg=wg, wi=wi, wff=wff))
    W["layers"] = layers

    gf = f64("encoder.final_layer_norm.weight")[:, None]
    oe = "output_patch_embedding."
    W["head_wh"] = gf * f64(oe + "hidden_layer.weight").T
    W["head_bh"] = f64(oe + "hidden_layer.bias")
    W["head_wo"] = f64(oe + "output_layer.weight").T
    W["head_bo"] = f64(oe + "output_layer.bias") + f64(oe + "residual_layer.bias")
    W["head_wr"] = gf * f64(oe + "residual_layer.weight").T
    return W


# ----------------------------------------------------------------------------
# Device precision presets
# ----------------------------------------------------------------------------
PRECISION_PRESETS = {
    # group -> (weight dtype, activation dtype); "residual" is the residual-stream dtype
    "bf16": {
        "embed": ("bf16", "bf16"),
        "attn": ("bf16", "bf16"),
        "ffn": ("bf16", "bf16"),
        "head": ("bf16", "bf16"),
        "residual": "bf16",
    },
    "fp32": {
        "embed": ("fp32", "fp32"),
        "attn": ("fp32", "fp32"),
        "ffn": ("fp32", "fp32"),
        "head": ("fp32", "fp32"),
        "residual": "fp32",
    },
    "bfp8_b": {
        "embed": ("bf16", "bf16"),
        "attn": ("bfp8_b", "bf16"),
        "ffn": ("bfp8_b", "bf16"),
        "head": ("bf16", "bf16"),
        "residual": "bf16",
    },
}
