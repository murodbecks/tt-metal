# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Chronos-2 TTNN port: execution of the device graph (model.py) — eager and traced passes,
2-CQ streaming, pass sizing, token packing and the host-device pipeline. Design notes and
measurements: the ops.py module docstring."""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch

import ttnn

from . import ops as _ops

TILE, NEG_INF = _ops.TILE, _ops.NEG_INF
_round_up, _pad_rows, _choose_pack = _ops._round_up, _ops._pad_rows, _ops._choose_pack
_pack_layout_v2, prepare_inputs = _ops._pack_layout_v2, _ops.prepare_inputs
_LATENCY_MAX_BATCH, _MAX_1D_MT, _SHARD_MT_STEP = _ops._LATENCY_MAX_BATCH, _ops._MAX_1D_MT, _ops._SHARD_MT_STEP
_PACK_MAX_K, _PACK_MAX_SP, _PACK_MAX_MT = _ops._PACK_MAX_K, _ops._PACK_MAX_SP, _ops._PACK_MAX_MT
_PACK_MAX_ROW_RATIO, _PACK_SWITCH = _ops._PACK_MAX_ROW_RATIO, _ops._PACK_SWITCH


class TracedRunner:
    """Execution mixin of Chronos2TTBackend: needs device, num_devices, _shard / _concat mesh
    mappers, the feature flags set in its __init__ and _forward_device / _make_shape_consts /
    _shape_consts / _layout_key."""

    # -- execution: eager and traced ---------------------------------------------
    def _to_host(self, t):
        return ttnn.to_torch(t, mesh_composer=self._concat).float().numpy()

    def _run_eager(self, hf, hm, B, S_pad):
        # While a trace is live, replays overwrite its free intermediate regions, so the
        # eager path must not leave persistent buffers behind: no RoPE caching then.
        transient = self._trace is not None
        Bd = B // self.num_devices
        cos3, sin3, bd = self._make_shape_consts(Bd, S_pad) if transient else self._shape_consts(Bd, S_pad)
        tf = ttnn.to_device(hf, self.device, memory_config=self.mem)
        tm = ttnn.to_device(hm, self.device, memory_config=self.mem)
        y = self._forward_device(tf, tm, cos3, sin3, bd)
        out = self._to_host(y)
        ttnn.deallocate(y)
        ttnn.deallocate(tf)
        ttnn.deallocate(tm)
        if transient:
            ttnn.deallocate(cos3)
            ttnn.deallocate(sin3)
            if bd is not None:
                ttnn.deallocate(bd)
        return out

    def release_trace(self):
        """Release the live trace and every buffer it captured (persistent inputs, output,
        RoPE, block mask, 2-CQ staging buffers)."""
        st, self._trace = self._trace, None
        if st is None:
            return
        if st.get("trace_id") is not None:
            ttnn.release_trace(self.device, st["trace_id"])
        for k in ("out", "tf", "tm", "cos3", "sin3", "bd", "sf", "sm"):
            if st.get(k) is not None:
                ttnn.deallocate(st[k])
        for t in st.get("so") or ():
            ttnn.deallocate(t)

    close = release_trace

    def _capture(self, hf, hm, B, S_pad, mask_sig=None):
        """New graph key: release the old trace, allocate persistent buffers, run eagerly
        once (program compile + this call's result), then capture the trace."""
        self.release_trace()
        # free eager-path RoPE / mask buffers so no other shape's tensors sit between captures
        for consts in self._shape_cache.values():
            for t in consts:
                if t is not None:
                    ttnn.deallocate(t)
        self._shape_cache = {}
        st = {
            "key": (B, S_pad, self._layout_key()),
            "trace_id": None,
            "out": None,
            "mask_sig": mask_sig,
            "in_free": None,
        }
        st["cos3"], st["sin3"], st["bd"] = self._make_shape_consts(B // self.num_devices, S_pad)
        st["tf"] = ttnn.to_device(hf, self.device, memory_config=self.mem)
        st["tm"] = ttnn.to_device(hm, self.device, memory_config=self.mem)
        self._trace = st
        y = self._forward_device(st["tf"], st["tm"], st["cos3"], st["sin3"], st["bd"])
        out = self._to_host(y)
        if self.two_cq:
            # 2-CQ staging (allocated before capture so replays never overwrite them):
            # input staging written by CQ1, two output buffers read by CQ1
            st["sf"] = ttnn.to_device(hf, self.device, memory_config=self.mem)
            st["sm"] = ttnn.to_device(hm, self.device, memory_config=self.mem)
            st["so"] = [y, ttnn.clone(y, memory_config=self.mem)]
        else:
            ttnn.deallocate(y)
        set_misses = getattr(self.device, "set_program_cache_misses_allowed", None)
        try:
            if set_misses is not None:
                set_misses(False)
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            try:
                st["out"] = self._forward_device(st["tf"], st["tm"], st["cos3"], st["sin3"], st["bd"])
            finally:
                ttnn.end_trace_capture(self.device, tid, cq_id=0)
                st["trace_id"] = tid
        except Exception as exc:  # eager fallback, keep serving
            warnings.warn(f"chronos2: trace capture failed ({exc!r}); falling back to eager execution")
            self.release_trace()
            self.use_trace = False
        finally:
            if set_misses is not None:
                set_misses(True)
        return out

    def _replay(self, hf, hm, mask_sig):
        """Upload into the live trace's persistent inputs and enqueue a non-blocking replay."""
        st = self._trace
        ttnn.copy_host_to_device_tensor(hf, st["tf"], cq_id=0)
        if hm is not None:
            ttnn.copy_host_to_device_tensor(hm, st["tm"], cq_id=0)
            st["mask_sig"] = mask_sig
        ttnn.execute_trace(self.device, st["trace_id"], cq_id=0, blocking=False)

    def _run_traced(self, hf, hm, B, S_pad, mask_sig=None):
        """hm is None when the persistent mask buffer already holds this call's mask."""
        st = self._trace
        if st is None or st["key"] != (B, S_pad, self._layout_key()):
            return self._capture(hf, hm, B, S_pad, mask_sig)
        self._replay(hf, hm, mask_sig)
        return self._to_host(st["out"])

    def _mask_is_live(self, B, S_pad, mask_sig):
        st = self._trace
        return (
            self.use_trace
            and mask_sig is not None
            and st is not None
            and st["key"] == (B, S_pad, self._layout_key())
            and st.get("mask_sig") == mask_sig
        )

    def _host_mask(self, key_mask, B, S_pad):
        """bool key mask [B, S_pad] -> per-key additive mask [B, 1, S_pad, S_pad]; int16 packed
        segment map (seg >= 0: valid key of segment seg, -(seg + 1): masked key of segment seg)
        -> block-diagonal additive mask (a query sees only valid keys of its own segment);
        uint8 key validity (packed v2) -> per-key row [B, 1, 1, S_pad], added on the device to
        the layout's constant block-diagonal mask."""
        if key_mask.dtype == np.uint8:
            add_mask = np.where(key_mask > 0, 0.0, NEG_INF).astype(np.float32)[:, None, None, :]
        elif key_mask.dtype == bool:
            add_mask = np.where(key_mask[:, None, None, :], 0.0, NEG_INF).astype(np.float32)
            add_mask = np.broadcast_to(add_mask, (B, 1, S_pad, S_pad)).copy()
        else:
            seg = np.where(key_mask >= 0, key_mask, -key_mask - 1)
            ok = (key_mask[:, None, :] >= 0) & (seg[:, :, None] == seg[:, None, :])
            add_mask = np.where(ok, 0.0, NEG_INF).astype(np.float32)[:, None]
        return ttnn.from_torch(
            torch.from_numpy(add_mask), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._shard
        )

    def _host_pass(self, feats, key_mask, live_sig=None):
        """Host tensors of one pass: the features (tilized, or row-major bf16 on the packed v2
        path with rm_upload, tilized on the device) and the SDPA mask, or None for the mask
        when live_sig (signature held in the persistent mask buffer) already matches."""
        B, S_pad, _ = feats.shape
        mask_sig = key_mask.tobytes() if self.mask_cache else None
        if self.rm_upload and self._pack_layout is not None:
            hf = ttnn.from_torch(
                torch.from_numpy(feats).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._shard,
            )
        else:
            hf = ttnn.from_torch(
                torch.from_numpy(feats), dtype=self._act("embed"), layout=ttnn.TILE_LAYOUT, mesh_mapper=self._shard
            )
        live = mask_sig is not None and mask_sig == live_sig
        hm = None if live else self._host_mask(key_mask, B, S_pad)
        return hf, hm, mask_sig

    def _run_pass(self, feats, key_mask):
        """One device pass over every card: rows divide evenly by the card count."""
        B, S_pad, _ = feats.shape
        mask_sig = key_mask.tobytes() if self.mask_cache else None
        live_sig = mask_sig if self._mask_is_live(B, S_pad, mask_sig) else None
        hf, hm, mask_sig = self._host_pass(feats, key_mask, live_sig)
        if self.use_trace:
            return self._run_traced(hf, hm, B, S_pad, mask_sig)
        return self._run_eager(hf, hm, B, S_pad)

    def _enqueue_2cq(self, hf, hm, mask_sig, slot):
        """CQ1 writes the staging inputs; CQ0 copies them into the trace inputs, replays and
        copies the output into staging slot `slot`. Returns the CQ0 event for that copy."""
        st = self._trace
        if st["in_free"] is not None:
            ttnn.wait_for_event(1, st["in_free"])  # previous pass has consumed the staging inputs
        ttnn.copy_host_to_device_tensor(hf, st["sf"], cq_id=1)
        if hm is not None:
            ttnn.copy_host_to_device_tensor(hm, st["sm"], cq_id=1)
        written = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, written)
        ttnn.copy(st["sf"], st["tf"])
        if hm is not None:
            ttnn.copy(st["sm"], st["tm"])
            st["mask_sig"] = mask_sig
        st["in_free"] = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, st["trace_id"], cq_id=0, blocking=False)
        ttnn.copy(st["out"], st["so"][slot])
        return ttnn.record_event(self.device, 0)

    def _read_2cq(self, event, slot):
        ttnn.wait_for_event(1, event)
        t = ttnn.from_device(self._trace["so"][slot], cq_id=1)
        return ttnn.to_torch(t, mesh_composer=self._concat).float().numpy()

    def _run_passes_2cq(self, feats, key_mask, Bp):
        """Passes 1.. of a multi-pass call: pass k+1 is prepared, written (CQ1) and enqueued
        (CQ0) before pass k's output is read back on CQ1. Output slots alternate, and slot
        k % 2 is read (blocking) before pass k + 2 is enqueued into it."""
        st = self._trace
        starts = list(range(Bp, feats.shape[0], Bp))

        def enqueue(k):
            i = starts[k]
            hf, hm, sig = self._host_pass(feats[i : i + Bp], key_mask[i : i + Bp], st["mask_sig"])
            return self._enqueue_2cq(hf, hm, sig, k % 2)

        outs, event = [], enqueue(0)
        for k in range(len(starts)):
            nxt = enqueue(k + 1) if k + 1 < len(starts) else None
            outs.append(self._read_2cq(event, k % 2))
            event = nxt
        return outs

    def _run_passes(self, feats, key_mask, Bp):
        """Sequential passes of Bp rows. Once the trace for (Bp, S_pad) is live, passes
        1.. stream over two command queues (or, opt-in on one queue, overlap the host prep
        of pass i+1 with the replay of pass i); otherwise they run serially."""
        total, S_pad = feats.shape[0], feats.shape[1]
        outs = [self._run_pass(feats[:Bp], key_mask[:Bp])]
        st = self._trace
        if total == Bp:
            return outs
        live = self.use_trace and st is not None and st["key"] == (Bp, S_pad, self._layout_key())
        if live and self.two_cq and st.get("so") is not None:
            return outs + self._run_passes_2cq(feats, key_mask, Bp)
        if not (live and self.host_overlap):
            return outs + [self._run_pass(feats[i : i + Bp], key_mask[i : i + Bp]) for i in range(Bp, total, Bp)]
        nxt = self._host_pass(feats[Bp : 2 * Bp], key_mask[Bp : 2 * Bp], st["mask_sig"])
        for i in range(Bp, total, Bp):
            self._replay(*nxt)
            j = i + Bp
            if j < total:
                # mask_sig is final once this replay's upload is enqueued
                nxt = self._host_pass(feats[j : j + Bp], key_mask[j : j + Bp], st["mask_sig"])
            outs.append(self._to_host(st["out"]))
        return outs

    def _use_sharded(self, B, S_pad):
        """Sharded large-M path for a per-card pass of B series of S_pad rows: Mt > _MAX_1D_MT and
        B > _LATENCY_MAX_BATCH, so latency calls keep the interleaved graph at any S_pad."""
        return self.large_rope_dir is not None and B > _LATENCY_MAX_BATCH and B * S_pad // TILE > _MAX_1D_MT

    def _pass_size(self, B, S_pad, max_batch=None):
        """Rows per device pass (all cards): the whole call when it fits, else fixed passes of
        max_batch (default self.max_batch) series per card. On the sharded large-M path each
        card's share is padded so its Mt is a multiple of _SHARD_MT_STEP (keeps gy = 8+ grid
        rows); B <= 4 never takes that path."""
        nd = self.num_devices
        mb = self.max_batch if max_batch is None else max_batch
        if B > nd * mb:
            return nd * mb
        Bc = -(-B // nd)
        if self._use_sharded(Bc, S_pad):
            step = _SHARD_MT_STEP * TILE // math.gcd(_SHARD_MT_STEP * TILE, S_pad)
            Bc = min(_round_up(Bc, step), mb)
        return nd * Bc

    def _pack_plan(self, B, S, S_pad):
        """(k, Sp, Bp) when token packing applies to a call of B series (B > 4, packed passes
        on the sharded path per card, padded rows <= _PACK_MAX_ROW_RATIO x unpacked), else
        None. k gives the fewest padded rows (module docstring); the density choice
        _choose_pack(S) is kept unless that saves more than _PACK_SWITCH."""
        if not self.pack or B <= _LATENCY_MAX_BATCH:
            return None
        base_rows = _round_up(B, self._pass_size(B, S_pad)) * S_pad
        k0, _ = _choose_pack(S)
        cands = {}
        for k in range(2, _PACK_MAX_K + 1):
            Sp = _round_up(k * S, TILE)
            if Sp > _PACK_MAX_SP:
                break
            nb = -(-B // k)
            Bp = self._pass_size(nb, Sp, max(1, _PACK_MAX_MT * TILE // Sp))
            if not self._use_sharded(Bp // self.num_devices, Sp):
                continue
            rows = _round_up(nb, Bp) * Sp
            if rows > _PACK_MAX_ROW_RATIO * base_rows:
                continue
            cands[k] = (rows * (1 + Sp / 5000), Sp, Bp)
        if not cands:
            return None
        k = min(cands, key=lambda j: cands[j][0])
        if k0 in cands and cands[k][0] > (1 - _PACK_SWITCH) * cands[k0][0]:
            k = k0
        return k, cands[k][1], cands[k][2]

    def _run_packed(self, feats, key_mask, S, H, k, Sp, Bp):
        """Token packing (module docstring): k series per sequence of Sp rows with a
        block-diagonal mask; returns the per-series future-token outputs [B, H, C]."""
        B, _, F = feats.shape
        nb = -(-B // k)
        total = _round_up(nb, Bp)
        if not self.pack_v2:
            f = _pad_rows(feats[:, :S], nb * k - B).reshape(nb, k * S, F)
            km = _pad_rows(key_mask[:, :S], nb * k - B).reshape(nb, k * S)
            seg = np.repeat(np.arange(k, dtype=np.int16), S)
            pf = np.zeros((nb, Sp, F), np.float32)
            pf[:, : k * S] = f
            ks = np.full((nb, Sp), k, np.int16)  # trailing pad rows: their own (valid) segment k
            ks[:, : k * S] = np.where(km, seg[None], -(seg[None] + 1))
            outs = self._run_passes(_pad_rows(pf, total - nb), _pad_rows(ks, total - nb), Bp)
            out = (outs[0] if len(outs) == 1 else np.concatenate(outs, axis=0))[:nb]
            return out[:, : k * S].reshape(nb * k, S, out.shape[-1])[:B, S - H : S]
        lay = _pack_layout_v2(k, S, H, Sp)
        J, T = lay["J"], lay["T"]
        fs = _pad_rows(feats[:, :S], nb * k - B).reshape(nb, k, S, F)
        kms = _pad_rows(key_mask[:, :S], nb * k - B).reshape(nb, k, S)
        pf = np.zeros((total, Sp, F), np.float32)
        pf[:nb, : k * S] = fs[:, J, T]
        valid = np.ones((total, Sp), np.uint8)  # trailing pad rows and pad sequences: valid keys
        valid[:nb, : k * S] = kms[:, J, T]
        if total > nb:
            pf[nb:] = pf[nb - 1]
            valid[nb:] = valid[nb - 1]
        self._pack_layout = lay
        try:
            outs = self._run_passes(pf, valid, Bp)
        finally:
            self._pack_layout = None
        out = (outs[0] if len(outs) == 1 else np.concatenate(outs, axis=0))[:nb]
        return out[:, : k * H].reshape(nb * k, H, out.shape[-1])[:B]

    def _postprocess(self, fut_rows, loc, scale, h_req):
        """Future-token head outputs [B, H, C] -> quantile forecasts [B, h_req, Q] (host FP32)."""
        cfg = self.cfg
        B, H = fut_rows.shape[:2]
        Q, P = len(cfg.quantiles), cfg.out_patch
        fut = fut_rows[:, :, : Q * P].reshape(B, H, Q, P)
        fut = fut.transpose(0, 2, 1, 3).reshape(B, Q, H * P)[:, :, :h_req].astype(np.float32)
        if cfg.use_arcsinh:
            fut = np.sinh(fut)
        fut = fut * scale[:, :, None] + loc[:, :, None]
        return np.ascontiguousarray(fut.transpose(0, 2, 1), dtype=np.float32)

    def _forecast_pipelined(self, v, m, h_req, S, H, k, Sp, Bp):
        """Host-device pipeline (module docstring) on the packed v2 path: chunks of one pass
        (k * Bp series) are prepared and packed exactly as _run_packed does for that pass,
        enqueued while the previous pass replays, and post-processed while the next replays."""
        cfg = self.cfg
        B, n_ser = v.shape[0], k * Bp
        starts = list(range(0, B, n_ser))
        lay = _pack_layout_v2(k, S, H, Sp)
        J, T = lay["J"], lay["T"]

        def prep(c):
            i = starts[c]
            feats, key_mask, loc, scale, _, _, _ = prepare_inputs(cfg, v[i : i + n_ser], m[i : i + n_ser], h_req)
            n, F = feats.shape[0], feats.shape[2]
            nb = -(-n // k)
            fs = _pad_rows(feats[:, :S], nb * k - n).reshape(nb, k, S, F)
            kms = _pad_rows(key_mask[:, :S], nb * k - n).reshape(nb, k, S)
            pf = np.zeros((Bp, Sp, F), np.float32)
            pf[:nb, : k * S] = fs[:, J, T]
            valid = np.ones((Bp, Sp), np.uint8)  # trailing pad rows and pad sequences: valid keys
            valid[:nb, : k * S] = kms[:, J, T]
            if Bp > nb:
                pf[nb:] = pf[nb - 1]
                valid[nb:] = valid[nb - 1]
            return pf, valid, (n, nb, loc, scale)

        def post(out, meta):
            n, nb, loc, scale = meta
            return self._postprocess(out[:nb, : k * H].reshape(nb * k, H, out.shape[-1])[:n], loc, scale, h_req)

        def streaming():
            st = self._trace
            return (
                self.use_trace
                and self.two_cq
                and st is not None
                and st.get("so") is not None
                and st["key"] == (Bp, Sp, self._layout_key())
            )

        def enqueue(c, p):
            hf, hm, sig = self._host_pass(p[0], p[1], self._trace["mask_sig"])
            return self._enqueue_2cq(hf, hm, sig, c % 2)

        res, n_chunks = [], len(starts)
        self._pack_layout = lay
        try:
            cur, c0 = prep(0), 0
            if not streaming():
                # new graph key (eager run + capture) or no 2-CQ trace: pass 0 blocks
                res.append(post(self._run_pass(cur[0], cur[1]), cur[2]))
                c0 = 1
                if not streaming():
                    for c in range(1, n_chunks):
                        cur = prep(c)
                        res.append(post(self._run_pass(cur[0], cur[1]), cur[2]))
                    c0 = n_chunks
                elif c0 < n_chunks:
                    cur = prep(c0)
            if c0 < n_chunks:
                event = enqueue(c0, cur)
                for c in range(c0, n_chunks):
                    nxt = None
                    if c + 1 < n_chunks:
                        p = prep(c + 1)  # host prep of the next chunk overlaps this replay
                        nxt = enqueue(c + 1, p)
                    res.append(post(self._read_2cq(event, c % 2), cur[2]))
                    if nxt is not None:
                        cur = p
                    event = nxt
        finally:
            self._pack_layout = None
        return res[0] if len(res) == 1 else np.concatenate(res, axis=0)
