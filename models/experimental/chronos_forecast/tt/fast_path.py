# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fast path behind ``TtChronos.forward`` for independent univariate series (each series its own
group, no covariates), the common forecasting call. ``TtChronos`` routes a call here when
``supports`` holds and runs everything else through the encoder modules. Tuned for latency at
batch <= 4 and throughput at large batch, on one chip or data parallel over a 1xN mesh.

- fast_path_ops.py: design notes and measurements (module docstring), constants, host prep,
  weight folding and the precision presets;
- fast_path_runtime.py: TracedRunner, the execution mixin (eager / traced passes, 2-CQ
  streaming, pass sizing, token packing, host-device pipeline);
- fast_path.py (this file): the device graph and TtChronosFastPath;
- ops/split_rope: the fused split-heads + RoPE op of the large-batch path.

Row-major feature upload (fast_path_ops docstring) is on by default only on a multi-chip
mesh (CHRONOS2_RM_UPLOAD=auto): there the host tilize bounds the packed passes, +16% / +30%
items/s at b1024 / b4096 on 1x4 in bf16 (bfp8_b +19% / +28%), while one chip is device-bound
and measured -2.5% (bf16) / -0.8% (bfp8_b) mean items/s at b1024 with the extra on-device
tilize. CHRONOS2_RM_UPLOAD=1 forces it on, 0 off.

FF intermediate in bfp8_b (bfp8_b preset, sharded B>4 path only): the ff_up output (relu
fused) that feeds ff_down is written in bfp8_b instead of bf16, halving the L1 bytes of the
widest activation (3072 columns); the profile showed LoFi ff_down at 22.6% FPU, slower than
HiFi2 ff_up. Off by default (it did not lower J/item measurably); CHRONOS2_LARGE_FF_BF8=1
enables it; bf16 mode and B<=4 are unchanged.
"""
from __future__ import annotations

import os

import numpy as np

import ttnn

from models.experimental.chronos_forecast.ops.split_rope import split_rope
from models.experimental.chronos_forecast.tt import fast_path_ops as _ops
from models.experimental.chronos_forecast.tt.fast_path_runtime import TracedRunner

# the shared names (constants, host prep, folds, presets) used by the graph below
globals().update({k: v for k, v in vars(_ops).items() if not k.startswith("__")})


class Chronos2TTBackend(TracedRunner):
    precision_policy = {
        "mode": "bf16",
        "weights": "bf16",
        "activations": "bf16",
        "accumulation": "fp32",
        "exceptions": [],
    }

    def __init__(self, weights_path, config, device, *, precision="bf16", groups=None, use_trace=None):
        if precision not in PRECISION_PRESETS:
            raise NotImplementedError(
                f"precision {precision!r} is not supported by this TTNN backend "
                f"(supported: {sorted(PRECISION_PRESETS)}; installed TTNN has no native fp16)"
            )
        self.cfg = Chronos2Config.from_any(config, weights_path)
        self.device = device
        self.precision = precision
        # data parallel over a 1xN MeshDevice: replicate weights, shard rows on dim 0
        get_n = getattr(device, "get_num_devices", None)
        self.num_devices = int(get_n()) if get_n is not None else 1
        multi = self.num_devices > 1
        self._rep = ttnn.replicate_tensor_to_mesh_mapper(device) if multi else None
        self._shard = ttnn.shard_tensor_to_mesh_mapper(device, dim=0) if multi else None
        self._concat = ttnn.concat_mesh_to_tensor_composer(device, dim=0) if multi else None
        self.groups = dict(PRECISION_PRESETS[precision])
        if groups:
            self.groups.update(groups)
        self.ck = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.mm_fidelity = _parse_fidelity_ladder(
            os.environ.get("CHRONOS2_MM_FIDELITY", _DEFAULT_MM_FIDELITY.get(precision, ""))
        )
        # sharded large-M path (B > 4 only): the table above with the large-path overrides on top
        self.large_mm_fidelity = dict(self.mm_fidelity)
        self.large_mm_fidelity.update(
            _parse_fidelity_ladder(
                os.environ.get("CHRONOS2_LARGE_MM_FIDELITY", _DEFAULT_LARGE_MM_FIDELITY.get(precision, "")),
                "CHRONOS2_LARGE_MM_FIDELITY",
            )
        )
        fid_ck = lambda f: ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, f),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.mm_ck = {g: fid_ck(f) for g, f in self.mm_fidelity.items()}
        self.mm_ck_large = {g: fid_ck(f) for g, f in self.large_mm_fidelity.items()}
        # sharded large-M path SDPA (B > 4 only): one chunk per sequence, HiFi2, no fp32 dest acc
        self.large_sdpa = os.environ.get("CHRONOS2_LARGE_SDPA", "1") != "0"
        self.sdpa_ck_large = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # sharded large-M path (B > 4 only): QKV / FF up read an L1-interleaved norm output with
        # wide K blocks (ops.py docstring)
        self.large_ibw = os.environ.get("CHRONOS2_LARGE_IBW", "1") != "0"
        # sharded large-M path (B > 4 only), bfp8_b preset only: FF intermediate in bfp8_b (opt-in)
        self.large_ff_bf8 = precision == "bfp8_b" and os.environ.get("CHRONOS2_LARGE_FF_BF8", "0") == "1"
        self.w_bfp8 = _parse_groups(os.environ.get("CHRONOS2_W_BFP8", ""), "CHRONOS2_W_BFP8")
        self.w_bfp4 = _parse_groups(os.environ.get("CHRONOS2_W_BFP4", ""), "CHRONOS2_W_BFP4")
        if self.w_bfp8 & self.w_bfp4:
            raise ValueError(
                f"groups {sorted(self.w_bfp8 & self.w_bfp4)} are in both " "CHRONOS2_W_BFP8 and CHRONOS2_W_BFP4"
            )
        self.precision_policy = self._policy()
        self.mem = ttnn.DRAM_MEMORY_CONFIG
        # residual stream, rms_norm outputs and wo / wg / FF-down outputs (CHRONOS2_ACT_L1=0: DRAM)
        self.act_l1 = os.environ.get("CHRONOS2_ACT_L1", "1") != "0"
        self.act_mem = ttnn.L1_MEMORY_CONFIG if self.act_l1 else self.mem
        self._grid = device.compute_with_storage_grid_size()
        self.use_mm_configs = os.environ.get("CHRONOS2_MM_CONFIG", "1") != "0"
        self.use_mm_wide = os.environ.get("CHRONOS2_MM_WIDE", "1") != "0"
        # blocked 2D head configs on the sharded large-M path (module docstring)
        self.head_large = os.environ.get("CHRONOS2_HEAD_LARGE", "1") != "0"
        self.rope_fused = os.environ.get("CHRONOS2_ROPE_FUSED", "1") != "0"
        bf16_attn = self._act("attn") == ttnn.bfloat16 and self.cfg.d_kv == 2 * TILE
        # custom split + RoPE (opt-in): one card, bf16 attention activations, d_kv == two tiles, kernels present
        self.split_rope_dir = None
        if os.environ.get("CHRONOS2_SPLIT_ROPE", "0") != "0" and self.rope_fused and not multi and bf16_attn:
            self.split_rope_dir = _split_rope_kernel_dir()
        # sharded large-M path (per-card Mt > _MAX_1D_MT only; per chip on a mesh): same conditions
        # plus a bf16 residual stream
        self.large_rope_dir = None
        if (
            os.environ.get("CHRONOS2_SHARD_LARGE", "1") != "0"
            and self.rope_fused
            and bf16_attn
            and self.use_mm_configs
            and _DTYPES[self.groups["residual"]] == ttnn.bfloat16
        ):
            self.large_rope_dir = _split_rope_kernel_dir()
        default_batch = _MAX_DEVICE_BATCH_SHARDED if self.large_rope_dir is not None else _MAX_DEVICE_BATCH
        self.max_batch = int(os.environ.get("CHRONOS2_MAX_BATCH", default_batch))
        if self.max_batch < 1:
            raise ValueError("CHRONOS2_MAX_BATCH must be >= 1")
        # token packing on the sharded large-M path (B > 4 only); v2 layout (futures first, device mask)
        self.pack = os.environ.get("CHRONOS2_PACK", "1") != "0" and self.large_rope_dir is not None
        self.pack_v2 = os.environ.get("CHRONOS2_PACK_V2", "1") != "0"
        self._pack_layout = None  # packed v2 layout of the call in flight (graph key, RoPE rows, mask)
        self.mask_cache = os.environ.get("CHRONOS2_MASK_CACHE", "1") != "0"
        self.host_overlap = os.environ.get("CHRONOS2_HOST_OVERLAP", "0") != "0"
        # host-device pipeline on the packed path (module docstring)
        self.pipeline = os.environ.get("CHRONOS2_PIPELINE", "1") != "0"
        # row-major bf16 feature upload, tilized on the device (packed v2 path only): default on a
        # multi-card mesh only (module docstring)
        rm = os.environ.get("CHRONOS2_RM_UPLOAD", "auto")
        self.rm_upload = self._act("embed") == ttnn.bfloat16 and (multi if rm == "auto" else rm != "0")
        self.two_cq = os.environ.get("CHRONOS2_2CQ", "1") != "0" and _num_command_queues(device) >= 2
        self._mm_cache = {}
        W = fold_weights(self.cfg, _load_state_dict(weights_path))
        self._upload(W)
        self._shape_cache = {}
        if use_trace is None:
            use_trace = _default_use_trace()
        self.use_trace = bool(use_trace)
        self._trace = None  # live trace state for one graph key

    # -- setup ---------------------------------------------------------------
    def _policy(self):
        g = self.groups
        act = "bf16" if self.precision == "bfp8_b" else self.precision
        exc = []
        for name in ("embed", "attn", "ffn", "head"):
            w, a = g[name]
            if w != self.precision or a != act:
                exc.append(f"{name} group: weights {w}, activations {a}")
        if g["residual"] != act:
            exc.append(f"residual stream kept in {g['residual']}")
        if act != "fp32":
            exc.append("quantile head output linear/add written in fp32 before host readback")
        for grp, fid in self.mm_fidelity.items():
            if fid != "HiFi4":
                exc.append(f"encoder {grp} linears at math fidelity {fid}")
        for grp, fid in self.large_mm_fidelity.items():
            if fid != self.mm_fidelity.get(grp, "HiFi4"):
                exc.append(
                    f"encoder {grp} linears at math fidelity {fid} on the sharded large-batch "
                    f"path (more than {_LATENCY_MAX_BATCH} series per pass)"
                )
        if self.large_sdpa:
            exc.append(
                "time-attention SDPA at math fidelity HiFi2 without fp32 dest accumulation on the "
                f"sharded large-batch path (more than {_LATENCY_MAX_BATCH} series per pass)"
            )
        if self.large_ff_bf8:
            exc.append(
                "ffn group: FF intermediate activation (ff_up output into ff_down) in bfp8_b on the "
                f"sharded large-batch path (more than {_LATENCY_MAX_BATCH} series per pass)"
            )
        for grp in sorted(self.w_bfp8):
            exc.append(f"encoder {grp} linear weights in bfp8_b")
        for grp in sorted(self.w_bfp4):
            exc.append(f"encoder {grp} linear weights in bfp4_b")
        exc.append(
            "host FP32 non-learned prep/post: instance-norm loc/scale, arcsinh, patching, "
            "RoPE cos/sin tables, sinh + unscale of head outputs"
        )
        return {
            "mode": self.precision,
            "weights": self.precision,
            "activations": act,
            "accumulation": "fp32 dest accumulation (fp32_dest_acc_en, packer_l1_acc), HiFi4 "
            "unless an exception lists a lower fidelity, for matmul, rms_norm and SDPA",
            "exceptions": exc,
        }

    def _w(self, arr, dtype_key):
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if t.ndim == 1:
            t = t.reshape(1, -1)
        return ttnn.from_torch(
            t,
            dtype=_DTYPES[dtype_key],
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.mem,
            mesh_mapper=self._rep,
        )

    def _upload(self, W):
        g = self.groups
        ew, hw = g["embed"][0], g["head"][0]
        bias_dt = lambda w: "fp32" if w == "fp32" else "bf16"
        self.emb = {
            k: self._w(W["emb_" + k], ew if k.startswith("w") else bias_dt(ew)) for k in ("wh", "bh", "wo", "bo", "wr")
        }
        aw, fw = g["attn"][0], g["ffn"][0]
        wdt = lambda grp, base: ("bfp4_b" if grp in self.w_bfp4 else "bfp8_b" if grp in self.w_bfp8 else base)
        self.layers = []
        for L in W["layers"]:
            self.layers.append(
                {
                    "wqkv": self._w(L["wqkv"], wdt("qkv", aw)),
                    "wrot": None if self.rope_fused else self._w(L["wrot"], wdt("qkv", aw)),
                    "wo": self._w(L["wo"], wdt("wo", aw)),
                    "wg": self._w(L["wg"], wdt("wg", aw)),
                    "wi": self._w(L["wi"], wdt("ff_up", fw)),
                    "wff": self._w(L["wff"], wdt("ff_down", fw)),
                }
            )
        self.head = {
            k: self._w(W["head_" + k], hw if k.startswith("w") else bias_dt(hw)) for k in ("wh", "bh", "wo", "bo", "wr")
        }

    def _act(self, group):
        return _DTYPES[self.groups[group][1]]

    def _layout_key(self):
        L = self._pack_layout
        return None if L is None else L["key"]

    def _make_shape_consts(self, B, S_pad):
        """(cos, sin, block mask) for a per-card batch B; the RoPE rows follow the packed v2
        layout in flight (if any), whose constant block-diagonal mask [B, 1, S_pad, S_pad] is the
        third entry (None otherwise)."""
        L = self._pack_layout
        cos, sin = rope_tables(self.cfg, S_pad, None if L is None else L["pos"])
        ad = self._act("attn")
        bd = None
        if L is not None:
            seg = L["seg"]
            blk = np.where(seg[:, None] == seg[None, :], 0.0, NEG_INF).astype(np.float32)
            bd = ttnn.from_torch(
                torch.from_numpy(np.broadcast_to(blk, (B, 1, S_pad, S_pad)).copy()),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.mem,
                mesh_mapper=self._rep,
            )
        if self.rope_fused:
            # shared [1, 1, S_pad, d_kv] cache for ttnn.experimental.rotary_embedding / split_rope
            mk4 = lambda a: ttnn.from_torch(
                torch.from_numpy(a.reshape(1, 1, S_pad, -1).copy()),
                dtype=ad,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=self.mem,
                mesh_mapper=self._rep,
            )
            return mk4(cos), mk4(sin), bd
        inner = self.cfg.num_heads * self.cfg.d_kv
        cos3 = np.concatenate([np.tile(cos, (1, 2 * self.cfg.num_heads)), np.ones((S_pad, inner), np.float32)], 1)
        sin3 = np.concatenate([np.tile(sin, (1, 2 * self.cfg.num_heads)), np.zeros((S_pad, inner), np.float32)], 1)
        mk = lambda a: ttnn.from_torch(
            torch.from_numpy(np.broadcast_to(a, (B, S_pad, a.shape[-1])).copy()),
            dtype=ad,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.mem,
            mesh_mapper=self._rep,
        )
        return mk(cos3), mk(sin3), bd

    def _shape_consts(self, B, S_pad):
        key = (B, S_pad, self._layout_key())
        if key not in self._shape_cache:
            self._shape_cache[key] = self._make_shape_consts(B, S_pad)
        return self._shape_cache[key]

    # -- device graph ----------------------------------------------------------
    def _mm_config(self, x, w, activation=None):
        """Encoder linear config, batch folded into M: the 2D multicast config on a grid sized
        to divide (Mt, Nt), or the wider 1D config when it covers more cores.
        activation="relu" is fused into the matmul output."""
        if not self.use_mm_configs:
            return None
        shp = tuple(x.shape)
        M = int(np.prod(shp[:-1]))
        K, N = int(w.shape[-2]), int(w.shape[-1])
        key = (M, K, N, activation)
        if key not in self._mm_cache:
            Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
            gx = _largest_divisor_at_most(Nt, self._grid.x)
            gy = _largest_divisor_at_most(Mt, self._grid.y)
            fused = None
            if activation == "relu":
                fused = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
            elif activation is not None:
                raise NotImplementedError(f"fused activation {activation!r}")
            pc = self._mm_wide_config(Mt, Kt, Nt, gx * gy, fused)
            if pc is None:
                pm, pn = Mt // gy, Nt // gx
                sh, sw = _subblock(pm, pn, 4)  # fp32 dest acc: 4 dst tiles
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    in0_block_w=_largest_divisor_at_most(Kt, _MAX_IN0_BLOCK_W),
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    per_core_M=pm,
                    per_core_N=pn,
                    transpose_mcast=False,
                    fused_activation=fused,
                    fuse_batch=True,
                )
            self._mm_cache[key] = pc
        return self._mm_cache[key]

    def _mm_wide_config(self, Mt, Kt, Nt, cores_2d, fused):
        """Small-M grid widening: 1D in0-multicast config holding all of M on each core with
        the fewest output tile columns per core that fit the grid. None unless M is small
        enough and it uses more cores than the 2D config (cores_2d)."""
        if not self.use_mm_wide or Mt > _MAX_1D_MT:
            return None
        grid_cores = self._grid.x * self._grid.y
        pn = min(d for d in range(1, Nt + 1) if Nt % d == 0 and Nt // d <= grid_cores)
        if Nt // pn <= cores_2d:
            return None
        sh, sw = _subblock(Mt, pn, 4)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self._grid.x, self._grid.y),
            in0_block_w=_in0_block_w_1d(Mt, Kt),
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=Mt,
            per_core_N=pn,
            fuse_batch=True,
            fused_activation=fused,
            mcast_in0=True,
        )

    def _mm1d_config(self, x, w):
        """1D in0-multicast config for the wide-K, small-N output linears: each core owns
        one output tile column over all of M and streams K. None (auto config) when M is
        too large to keep on every core."""
        if not self.use_mm_configs:
            return None
        M = int(np.prod(tuple(x.padded_shape)[:-1]))
        K, N = int(w.padded_shape[-2]), int(w.padded_shape[-1])
        key = ("1d", M, K, N)
        if key not in self._mm_cache:
            Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
            pc = None
            if Mt <= _MAX_1D_MT and Nt <= self._grid.x * self._grid.y:
                sh, sw = _subblock(Mt, 1, 4)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(self._grid.x, self._grid.y),
                    in0_block_w=_in0_block_w_1d(Mt, Kt),
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    per_core_M=Mt,
                    per_core_N=1,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            self._mm_cache[key] = pc
        return self._mm_cache[key]

    def _mm_large_config(self, x, w, activation=None):
        """Large-M (Mt > _MAX_1D_MT) 2D multicast config for the embed / head linears, from the
        padded shapes, batch folded into M, activation="relu" fused. When the per-core output
        block (per_core_M x per_core_N) or in0 block (per_core_M x in0_block_w) exceeds
        _MAX_2D_BLOCK_TILES tiles, out_block_h bounds both."""
        M = int(np.prod(tuple(x.padded_shape)[:-1]))
        K, N = int(w.padded_shape[-2]), int(w.padded_shape[-1])
        key = ("2d_large", M, K, N, activation)
        if key not in self._mm_cache:
            Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
            gx = _largest_divisor_at_most(Nt, self._grid.x)
            gy = _largest_divisor_at_most(Mt, self._grid.y)
            fused = None
            if activation == "relu":
                fused = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
            elif activation is not None:
                raise NotImplementedError(f"fused activation {activation!r}")
            pm, pn = Mt // gy, Nt // gx
            ibw = _largest_divisor_at_most(Kt, _MAX_IN0_BLOCK_W)
            kw = {}
            obh = pm
            if pm * pn > _MAX_2D_BLOCK_TILES or pm * ibw > _MAX_2D_BLOCK_TILES:
                obh = max(
                    d
                    for d in range(1, pm + 1)
                    if pm % d == 0 and d * pn <= _MAX_2D_BLOCK_TILES and d * ibw <= _MAX_2D_BLOCK_TILES
                )
                kw = dict(out_block_h=obh, out_block_w=pn)
            sh, sw = _subblock(obh, pn, 4)  # fp32 dest acc: 4 dst tiles
            self._mm_cache[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=ibw,
                out_subblock_h=sh,
                out_subblock_w=sw,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=fused,
                fuse_batch=True,
                **kw,
            )
        return self._mm_cache[key]

    def _mm_sharded_config(self, x, w, Mt, gx, gy, out_sharded, activation=None, in0_cap=_MAX_2D_BLOCK_TILES):
        """Sharded large-M path: 2D multicast fuse-batch config on the residual shard grid
        (gx x gy). A block-sharded in0 streams one shard width per block; an interleaved in0
        uses the widest K block (<= _MAX_IN0_BLOCK_W) with per_core_M x in0_block_w <=
        in0_cap tiles. An interleaved output bounds the per-core L1 block with out_block_h."""
        Kt, Nt = int(x.padded_shape[-1]) // TILE, int(w.padded_shape[-1]) // TILE
        in_sharded = x.memory_config().is_sharded()
        key = ("sharded", Mt, Kt, Nt, activation, in_sharded, out_sharded, gx, gy, in0_cap)
        if key not in self._mm_cache:
            pm, pn = Mt // gy, Nt // gx
            if in_sharded:
                ibw = Kt // gx
            else:
                ibw = max(
                    d for d in range(1, min(Kt, _MAX_IN0_BLOCK_W) + 1) if Kt % d == 0 and (pm * d <= in0_cap or d == 1)
                )
            obh, kw = pm, {}
            if not out_sharded and pm * pn > _MAX_2D_BLOCK_TILES:
                obh = max(d for d in range(1, pm + 1) if pm % d == 0 and (d * pn <= _MAX_2D_BLOCK_TILES or d == 1))
                kw = dict(out_block_h=obh, out_block_w=pn)
            sh, sw = _subblock(obh, pn, 4)  # fp32 dest acc: 4 dst tiles
            fused = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if activation == "relu" else None
            self._mm_cache[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=ibw,
                out_subblock_h=sh,
                out_subblock_w=sw,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=fused,
                fuse_batch=True,
                **kw,
            )
        return self._mm_cache[key]

    def _sdpa_large_config(self, S_pad):
        """Sharded large-M path SDPA program config: one q / k chunk per (packed) sequence (the
        largest multiple of 32 dividing S_pad, at most _LARGE_SDPA_MAX_CHUNK)."""
        key = ("sdpa_large", S_pad)
        if key not in self._mm_cache:
            chunk = max(
                d * TILE for d in range(1, min(S_pad, _LARGE_SDPA_MAX_CHUNK) // TILE + 1) if S_pad % (d * TILE) == 0
            )
            self._mm_cache[key] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self._grid, q_chunk_size=chunk, k_chunk_size=chunk
            )
        return self._mm_cache[key]

    def _linear(
        self,
        x,
        w,
        *,
        bias=None,
        dtype=None,
        activation=None,
        program_config=None,
        compute_kernel_config=None,
        memory_config=None,
    ):
        if program_config is None and self.use_mm_configs:
            # large M: the TTNN auto config runs these at ~2 TFLOP/s; small M keeps the auto config
            if int(np.prod(tuple(x.padded_shape)[:-1])) // TILE > _MAX_1D_MT:
                program_config = self._mm_large_config(x, w, activation)
                activation = None
        return ttnn.linear(
            x,
            w,
            bias=bias,
            dtype=dtype,
            activation=activation,
            program_config=program_config,
            memory_config=memory_config or self.mem,
            compute_kernel_config=compute_kernel_config or self.ck,
        )

    def _mm(self, x, w, *, dtype, group, activation=None, memory_config=None):
        """Encoder linear: blocked program config (activation fused) or TTNN auto config,
        at the group's math fidelity (preset default or CHRONOS2_MM_FIDELITY, else HiFi4)."""
        pc = self._mm_config(x, w, activation)
        return self._linear(
            x,
            w,
            dtype=dtype,
            program_config=pc,
            activation=None if pc is not None else activation,
            compute_kernel_config=self.mm_ck.get(group),
            memory_config=memory_config,
        )

    def _rms(self, x, memory_config=None):
        # gammas are folded into the consumer linears
        return ttnn.rms_norm(
            x, epsilon=self.cfg.eps, memory_config=memory_config or self.mem, compute_kernel_config=self.ck
        )

    def _rope(self, t, cos, sin):
        r = ttnn.experimental.rotary_embedding(t, cos, sin, memory_config=self.mem, compute_kernel_config=self.ck)
        ttnn.deallocate(t)
        return r

    def _embed(self, feats, rd, am):
        e = self.emb
        h = self._linear(feats, e["wh"], bias=e["bh"], dtype=self._act("embed"), activation="relu")
        x = self._linear(h, e["wo"], bias=e["bo"], dtype=rd, program_config=self._mm1d_config(h, e["wo"]))
        ttnn.deallocate(h)
        r = self._linear(feats, e["wr"], dtype=rd)
        x = ttnn.add(x, r, dtype=rd, memory_config=am)
        ttnn.deallocate(r)
        return x

    def _head(self, x, large=False):
        """Final RMSNorm + quantile ResidualBlock. large=True (sharded large-M path only): the
        blocked 2D configs from Mt >= _HEAD_LARGE_MIN_MT (module docstring)."""
        xn = self._rms(x)
        ttnn.deallocate(x)
        hd = self.head
        big = (
            large
            and self.head_large
            and self.use_mm_configs
            and int(np.prod(tuple(xn.padded_shape)[:-1])) // TILE >= _HEAD_LARGE_MIN_MT
        )
        if big:
            h = self._linear(
                xn,
                hd["wh"],
                bias=hd["bh"],
                dtype=self._act("head"),
                program_config=self._mm_large_config(xn, hd["wh"], "relu"),
            )
        else:
            h = self._linear(xn, hd["wh"], bias=hd["bh"], dtype=self._act("head"), activation="relu")
        y = self._linear(
            h,
            hd["wo"],
            bias=hd["bo"],
            dtype=ttnn.float32,
            program_config=self._mm_large_config(h, hd["wo"]) if big else self._mm1d_config(h, hd["wo"]),
        )
        ttnn.deallocate(h)
        r = self._linear(
            xn, hd["wr"], dtype=ttnn.float32, program_config=self._mm_large_config(xn, hd["wr"]) if big else None
        )
        ttnn.deallocate(xn)
        y = ttnn.add(y, r, dtype=ttnn.float32)
        ttnn.deallocate(r)
        return y

    def _forward_device(self, feats, mask, cos3, sin3, bd=None):
        # mesh tensors report the per-card shape, so Mt and the path choice are per card
        shp = tuple(feats.padded_shape)
        Mt = int(np.prod(shp[:-1])) // TILE
        if self._use_sharded(int(shp[0]), int(shp[1])):
            return self._forward_device_sharded(feats, mask, cos3, sin3, Mt, bd)
        cfg = self.cfg
        rd = _DTYPES[self.groups["residual"]]
        aa, fa = self._act("attn"), self._act("ffn")
        am = self.act_mem
        x = self._embed(feats, rd, am)
        for L in self.layers:
            # time self-attention (RoPE on q/k: custom split + RoPE op, fused rotary op, or folded
            # rotate_half weights)
            xn = self._rms(x, am)
            qkv = self._mm(xn, L["wqkv"], dtype=aa, group="qkv")
            if not self.rope_fused:
                rot = self._mm(xn, L["wrot"], dtype=aa, group="qkv")
                qkv = ttnn.add(ttnn.mul(qkv, cos3), ttnn.mul(rot, sin3))
                ttnn.deallocate(rot)
            ttnn.deallocate(xn)
            if self.split_rope_dir is not None:
                q, k, v = split_rope(
                    qkv,
                    cos3,
                    sin3,
                    num_heads=cfg.num_heads,
                    kernel_dir=self.split_rope_dir,
                    compute_kernel_config=self.ck,
                    memory_config=self.mem,
                )
                ttnn.deallocate(qkv)
            else:
                q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                    qkv, num_heads=cfg.num_heads, transpose_key=False, memory_config=self.mem
                )
                ttnn.deallocate(qkv)
                if self.rope_fused:
                    q = self._rope(q, cos3, sin3)
                    k = self._rope(k, cos3, sin3)
            a = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=False, scale=1.0, compute_kernel_config=self.ck
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            a = ttnn.transformer.concatenate_heads(a, memory_config=self.mem)
            o = self._mm(a, L["wo"], dtype=rd, group="wo", memory_config=am)
            ttnn.deallocate(a)
            x = ttnn.add(x, o, dtype=rd, memory_config=am)
            ttnn.deallocate(o)
            # group self-attention, one key per query -> folded linear
            xn = self._rms(x, am)
            o = self._mm(xn, L["wg"], dtype=rd, group="wg", memory_config=am)
            ttnn.deallocate(xn)
            x = ttnn.add(x, o, dtype=rd, memory_config=am)
            ttnn.deallocate(o)
            # feed-forward
            xn = self._rms(x, am)
            h = self._mm(xn, L["wi"], dtype=fa, group="ff_up", activation="relu")
            ttnn.deallocate(xn)
            o = self._mm(h, L["wff"], dtype=rd, group="ff_down", memory_config=am)
            ttnn.deallocate(h)
            x = ttnn.add(x, o, dtype=rd, memory_config=am)
            ttnn.deallocate(o)
        return self._head(x)

    def _forward_device_sharded(self, feats, mask, cos3, sin3, Mt, bd=None):
        """Sharded large-M encoder (see module docstring): BLOCK-sharded L1 residual stream on
        gx x gy cores, sharded add / rms_norm, linears on the shard grid, q/k/v / SDPA in L1.
        Mt is the per-card tile row count. With a packed v2 layout, bd is the constant
        block-diagonal mask, `mask` the per-key row, and the head runs on the future rows only.
        Row-major features (rm_upload) are tilized here, inside the captured trace.
        With large_ibw the norms before QKV and FF up run on an L1-interleaved copy of the
        residual, so those linears take wide K blocks (ops.py docstring)."""
        cfg = self.cfg
        rd = _DTYPES[self.groups["residual"]]
        aa, fa = self._act("attn"), self._act("ffn")
        # FF intermediate dtype (bfp8_b with large_ff_bf8, module docstring)
        fh = ttnn.bfloat8_b if self.large_ff_bf8 else fa
        l1 = ttnn.L1_MEMORY_CONFIG
        own_mask = bd is not None
        if own_mask:
            mask = ttnn.add(bd, mask, memory_config=self.mem)
        if feats.layout == ttnn.ROW_MAJOR_LAYOUT:
            ft = ttnn.to_layout(feats, ttnn.TILE_LAYOUT, memory_config=self.mem)
            x = self._embed(ft, rd, l1)
            ttnn.deallocate(ft)
        else:
            x = self._embed(feats, rd, l1)
        Dt = int(x.padded_shape[-1]) // TILE
        gx = _largest_divisor_at_most(Dt, self._grid.x)
        gy = _largest_divisor_at_most(Mt, self._grid.y)
        smc = ttnn.create_sharded_memory_config(
            shape=(Mt // gy * TILE, Dt // gx * TILE),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        xs = ttnn.to_memory_config(x, smc)
        ttnn.deallocate(x)
        x = xs
        lnpc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            subblock_w=Dt // gx,
            block_h=Mt // gy,
            block_w=Dt // gx,
            inplace=False,
        )
        sdpa_kw = {"compute_kernel_config": self.ck}
        if self.large_sdpa:
            sdpa_kw = {
                "compute_kernel_config": self.sdpa_ck_large,
                "program_config": self._sdpa_large_config(int(mask.padded_shape[-1])),
            }
        wide = self.large_ibw
        in0_cap = _LARGE_IBW_IN0_TILES if wide else _MAX_2D_BLOCK_TILES

        def rms(t):
            return ttnn.rms_norm(
                t, epsilon=cfg.eps, program_config=lnpc, memory_config=smc, compute_kernel_config=self.ck
            )

        def rms_wide(t):
            # L1-interleaved norm output for the wide-K-block QKV / FF up (sharded otherwise)
            if not wide:
                return rms(t)
            ti = ttnn.sharded_to_interleaved(t, l1)
            y = ttnn.rms_norm(ti, epsilon=cfg.eps, memory_config=l1, compute_kernel_config=self.ck)
            ttnn.deallocate(ti)
            return y

        def mm(t, w, dtype, group, out_mc, activation=None, cap=_MAX_2D_BLOCK_TILES):
            # large-path fidelity table (self.mm_ck_large), see the module docstring
            pc = self._mm_sharded_config(t, w, Mt, gx, gy, out_mc.is_sharded(), activation, cap)
            return ttnn.linear(
                t,
                w,
                dtype=dtype,
                program_config=pc,
                memory_config=out_mc,
                compute_kernel_config=self.mm_ck_large.get(group) or self.ck,
            )

        for L in self.layers:
            # time self-attention: fused split + RoPE, SDPA and head concat in L1
            xn = rms_wide(x)
            qkv = mm(xn, L["wqkv"], aa, "qkv", l1, cap=in0_cap)
            ttnn.deallocate(xn)
            q, k, v = split_rope(
                qkv,
                cos3,
                sin3,
                num_heads=cfg.num_heads,
                kernel_dir=self.large_rope_dir,
                compute_kernel_config=self.ck,
                memory_config=l1,
            )
            ttnn.deallocate(qkv)
            a = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=False, scale=1.0, memory_config=l1, **sdpa_kw
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            a = ttnn.transformer.concatenate_heads(a, memory_config=l1)
            o = mm(a, L["wo"], rd, "wo", smc)
            ttnn.deallocate(a)
            x = ttnn.add(x, o, dtype=rd, memory_config=smc)
            ttnn.deallocate(o)
            # group self-attention, one key per query -> folded linear
            xn = rms(x)
            o = mm(xn, L["wg"], rd, "wg", smc)
            ttnn.deallocate(xn)
            x = ttnn.add(x, o, dtype=rd, memory_config=smc)
            ttnn.deallocate(o)
            # feed-forward (hidden L1 interleaved)
            xn = rms_wide(x)
            h = mm(xn, L["wi"], fh, "ff_up", l1, activation="relu", cap=in0_cap)
            ttnn.deallocate(xn)
            o = mm(h, L["wff"], rd, "ff_down", smc)
            ttnn.deallocate(h)
            x = ttnn.add(x, o, dtype=rd, memory_config=smc)
            ttnn.deallocate(o)
        if own_mask:
            ttnn.deallocate(mask)
        xi = ttnn.sharded_to_interleaved(x, l1)
        ttnn.deallocate(x)
        if self._pack_layout is not None:
            # packed v2: the future rows of every sequence are its first `rows` rows
            shp = tuple(xi.padded_shape)
            xf = ttnn.slice(xi, [0, 0, 0], [shp[0], self._pack_layout["rows"], shp[2]])
            ttnn.deallocate(xi)
            xi = xf
        return self._head(xi, large=True)

    # -- public API --------------------------------------------------------------
    def forecast(self, past_values, past_observed_mask, prediction_length):
        cfg = self.cfg
        h_req = int(prediction_length)
        if not 1 <= h_req <= cfg.max_output_patches * cfg.out_patch:
            raise ValueError(f"prediction_length {h_req} outside 1..{cfg.max_output_patches * cfg.out_patch}")
        if self.pipeline and self.pack and self.pack_v2:
            v = np.asarray(past_values, dtype=np.float32)
            m = np.asarray(past_observed_mask, dtype=np.float32)
            if v.ndim == 2 and v.shape == m.shape:
                _, H, S = token_counts(cfg, v.shape[1], h_req)
                plan = self._pack_plan(v.shape[0], S, _round_up(S, TILE))
                if plan is not None:
                    return {"quantiles": self._forecast_pipelined(v, m, h_req, S, H, *plan)}
        feats, key_mask, loc, scale, N, H, S = prepare_inputs(cfg, past_values, past_observed_mask, h_req)
        B = feats.shape[0]
        n_reg = 1 if cfg.use_reg_token else 0
        plan = self._pack_plan(B, S, feats.shape[1])
        if plan is not None:
            fut_rows = self._run_packed(feats, key_mask, S, H, *plan)
        else:
            Bp = self._pass_size(B, feats.shape[1])
            total = _round_up(B, Bp)
            feats, key_mask = _pad_rows(feats, total - B), _pad_rows(key_mask, total - B)
            outs = self._run_passes(feats, key_mask, Bp)
            out = (outs[0] if len(outs) == 1 else np.concatenate(outs, axis=0))[:B]
            fut_rows = out[:, N + n_reg : N + n_reg + H]
        return {"quantiles": self._postprocess(fut_rows, loc, scale, h_req)}


# Envelope the fast path is validated on; longer contexts / horizons take the encoder path.
MAX_CONTEXT = 512
MAX_OUTPUT_PATCHES = 4


def supports(config: dict) -> bool:
    """Checkpoint geometry the fast path implements: RoPE heads of 64 (two tiles), tile-aligned
    widths, ReLU feed-forward and equal non-overlapping input / output patches."""
    cc = config.get("chronos_config", {})
    return (
        int(config.get("d_kv", 0)) == 2 * TILE
        and int(config.get("d_model", 0)) % TILE == 0
        and int(config.get("d_ff", 0)) % TILE == 0
        and config.get("dense_act_fn", "relu") == "relu"
        and cc.get("input_patch_size") == cc.get("input_patch_stride") == cc.get("output_patch_size")
    )


class TtChronosFastPath(Chronos2TTBackend):
    """Built by ``TtChronos`` from the reference model's state dict and config. precision:
    "bf16" or "bfp8_b" (bfp8_b weights, LoFi wg / ff_down on the large path). A device opened with
    DEVICE_OPTIONS (two command queues, 32 MiB trace region) gets the traced 2-CQ path; otherwise
    it runs eagerly or on one queue."""

    def forward(self, context, context_mask=None, num_output_patches=1):
        """Quantile forecasts (B, Q, num_output_patches * output_patch_size), host float, the
        ``TtChronos.forward`` layout. ``context`` (B, T) may hold NaN for missing values."""
        import torch

        context = torch.as_tensor(context, dtype=torch.float32)
        if context_mask is None:
            context_mask = ~torch.isnan(context)
        horizon = num_output_patches * self.cfg.out_patch
        quantiles = self.forecast(context.numpy(), torch.as_tensor(context_mask).float().numpy(), horizon)
        return torch.from_numpy(np.ascontiguousarray(quantiles["quantiles"].transpose(0, 2, 1))).float()
