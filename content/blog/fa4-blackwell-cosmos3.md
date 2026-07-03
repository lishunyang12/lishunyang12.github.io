---
slug: "/blog/fa4-blackwell-cosmos3"
date: "2026-07-03"
title: "A SageAttention-Class Kernel for Datacenter Blackwell"
description: "FP8-quality attention without the quality loss: a mixed fp8-QKᵀ / bf16-P·V kernel built inside FlashAttention 4's CuTe-DSL for B300/B200, validated end-to-end on Cosmos3 in vLLM-Omni — plus the investigation that showed why stock FP8 attention cannot work for video DiTs."
---

Low-bit attention is the main speed lever left for long-video diffusion on datacenter Blackwell (SM100/SM103): at 189 frames, Cosmos3's self-attention runs over ~86k-token sequences where FlashAttention 4's FP8 kernel is 1.49x faster than cuDNN. But stock FP8 attention visibly degrades video DiT output, and SageAttention — the community's answer on consumer GPUs — has no kernels for sm_100. This post documents how we got FP8-class speed at bf16-class quality on a B300: a **mixed-precision kernel (fp8 QKᵀ, bf16 P·V) built inside FA4's own CuTe-DSL source**, plus the Hadamard input conditioning that closes the last gap. Model: **Cosmos3-Nano**, vLLM-Omni, single B300; all comparisons same-seed at official generation configs.

<div class="tldr">

**One-minute version.**

- **Quality at the bf16 floor, no tricks**: same-seed SSIM vs cuDNN — t2v @189f **0.970**, i2v **0.963** (floor 0.968), t2v @33f **0.950** (floor 0.975), v2v **0.811 — identical to its bf16 floor (0.8115)** — from step 0, with no bf16-warmup schedule. Stock FP8 scores 0.73–0.87 on the same tests.
- **Speed kept**: 1.24–1.25x over cuDNN kernel-level at Cosmos3 shapes, ~1.1x end-to-end at 189 frames. The Hadamard conditioning is free (fused into an already-compiled quantization block).
- **Why it works**: an error decomposition showed FA4's FP8 kernel carries an ~11–16% *intrinsic* error on diffuse attention. Building a mixed fp8-QKᵀ/bf16-P·V kernel (~90 lines inside FA4's CuTe-DSL) eliminated it — and the forensics that followed found the root cause: **a one-line bug**. FA4's pipelined softmax computes P̃ against a delayed running max, and with `max_offset=8` (P̃×256) e4m3 has only 1.75x headroom for the overshoot — wide-range logits (video DiTs, σ≥1) silently clip P. LLM-sharp attention never grows the max mid-sequence, so nobody saw it.
- **The endgame**: with the fix (`max_offset` 8→4) plus Hadamard input conditioning, **full FP8 runs at its native 1.49x with floor-level quality** — i2v 0.965, t2v 0.951, matching the mixed kernel while beating its speed. The fix is a one-constant upstream patch.

</div>

## The result

Same seed, official Cosmos3 configs (35 steps for t2v/i2v at 33 frames; 35 steps at 189 frames; v2v 50 steps / 121 frames conditioned on the official reference clip). SSIM against the cuDNN baseline; the bf16-FA4 column is the same-trajectory noise floor.

| Task | stock FP8 | mixed + Hadamard (1.24x) | **fixed FP8 + Hadamard (1.49x)** | bf16 floor |
|---|---|---|---|---|
| t2v · 33f | 0.872 | 0.950 | **0.951** | 0.975 |
| i2v · 33f | 0.733 | 0.963 | **0.965** | 0.968 |
| t2v · 189f (long video) | — | — | **0.970** | — |
| v2v · 121f · 50 steps (reference-conditioned) | — | — | **0.8115** | 0.8115 |

| Speed | cuDNN | FA4 bf16 | mixed + Hadamard | **fixed FP8 + Hadamard** |
|---|---|---|---|---|
| kernel @21.5k tokens | 1.00x | 1.03x | 1.25x | **1.49x** |
| kernel @86k tokens | 1.00x | 1.04x | 1.24x | **1.49x** |
| e2e @189f t2v | 2.45–2.55 s/step | 2.36 | ~2.29 (~1.1x) | **2.00 (1.28x)** |

Attention error on real Cosmos3 activations (relative to the bf16 kernel): stock FP8 **0.237** → mixed **0.065** → mixed+Hadamard **0.032**. Zero Dynamo recompiles in the compiled pipeline; no `fp8_start_step` schedule anywhere — the quality comes from the kernel, not from giving back speed on early steps.

<figure>
<div class="gal">
<div><video src="/videos/fa4/i2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN (baseline)</div></div>
<div><video src="/videos/fa4/i2v_fa4_fp8.mp4" controls muted loop playsinline></video><div class="c">stock FP8 · 0.733</div></div>
<div><video src="/videos/fa4/i2v_fa4_mixed_had.mp4" controls muted loop playsinline></video><div class="c">mixed + Hadamard · 0.963</div></div>
<div><video src="/videos/fa4/i2v_fa4_fp8fixed_had.mp4" controls muted loop playsinline></video><div class="c">fixed FP8 + Hadamard · 0.965</div></div>
</div>
<figcaption>Fig. 1 — i2v (the most quantization-sensitive short task), same seed, official config. bf16 floor: 0.968.</figcaption>
</figure>

<figure>
<div class="gal">
<div><video src="/videos/fa4/t2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN (baseline)</div></div>
<div><video src="/videos/fa4/t2v_fa4_fp8.mp4" controls muted loop playsinline></video><div class="c">stock FP8 · 0.872</div></div>
<div><video src="/videos/fa4/t2v_fa4_mixed.mp4" controls muted loop playsinline></video><div class="c">mixed kernel · 0.910</div></div>
<div><video src="/videos/fa4/t2v_fa4_mixed_had.mp4" controls muted loop playsinline></video><div class="c">mixed + Hadamard · 0.950</div></div>
</div>
<figcaption>Fig. 2 — t2v, same seed, official config. bf16 floor: 0.975.</figcaption>
</figure>

<figure>
<div class="gal">
<div><video src="/videos/fa4/t2v189_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN · 189 frames · 35 steps</div></div>
<div><video src="/videos/fa4/t2v189_mixed_had.mp4" controls muted loop playsinline></video><div class="c">mixed + Hadamard · SSIM 0.970</div></div>
<div><video src="/videos/fa4/v2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN · v2v · 121f / 50 steps</div></div>
<div><video src="/videos/fa4/v2v_mixed_had.mp4" controls muted loop playsinline></video><div class="c">mixed + Hadamard · SSIM 0.811 (= bf16 floor)</div></div>
</div>
<figcaption>Fig. 3 — the showcase regimes. Left pair: long-video t2v at the full official config (86k-token attention, where the speedup pays). Right pair: v2v continuation conditioned on the official reference clip — the most reference-conditioned task, hence the hardest quality test for quantized attention. Its lower absolute SSIM (0.811) is pure 50-step trajectory divergence: the bf16-FA4 control lands at the identical 0.8115, i.e. the mixed kernel contributes no measurable error on top of the same-seed kernel-noise floor.</figcaption>
</figure>

## Why stock FP8 attention fails on video DiTs

We started where everyone does: FA4's FP8 path with per-(batch, kv-head) e4m3 scales and smooth-K. Kernel-level it is 1.16–1.49x over cuDNN at DiT shapes, and end-to-end it delivered 1.22–1.28x at 189 frames. But same-seed quality was unacceptable — i2v SSIM 0.733 — and every input-side remedy we measured failed to fix it:

| Error source (real Cosmos3 activations, rel. error vs bf16 kernel) | |
|---|---|
| Input quantization alone (quantize-dequantize through the bf16 kernel) | 0.050–0.067 |
| Input quant at SageAttention-style 128-token blocks | 0.065 (no help) |
| Input quant with Hadamard rotation | 0.038 |
| **Real FP8 kernel, same scales** | **0.237** |
| Real FP8 kernel on pure random gaussian inputs | 0.111 @1k → 0.158 @10k tokens |

The decomposition is unambiguous: input scales contribute ~0.06 of the 0.24 — **the FP8 kernel itself carries an 11–16% intrinsic relative error on diffuse attention, even on exactly-fp8-representable inputs with descale=1**. Structural forensics then pinned the mechanism precisely. One-hot, graded two-hot, and uniform-attention probes are all *exact* — no packing, transfer, or accumulation defect. The error switches on with **logit dynamic range**: 0.027 at logit σ=0.25, 0.14 at σ=1, 0.36 at σ=2, while a true-fp8-math simulation stays flat at 0.02. The culprit: FA4's pipelined softmax computes P̃ against a *delayed* running max, so P̃ transiently exceeds 1 — and with `max_offset=8` (P̃×256), e4m3 saturates at 448, leaving only **1.75x headroom**. Wide-range logits clip P silently. Video DiTs live at σ≈1; LLM attention is sharp and never grows the max mid-sequence, which is why FP8 attention "works" there and why the failure went unnoticed — helped by test suites using absolute tolerances that silently admit ~12% relative error at diffuse-output magnitudes (ours since fixed to a relative bound).

The fix is one constant — `max_offset` 8→4 (28x headroom, best value on real activations) — worth an upstream patch on its own: it takes stock full-FP8 from 0.237 to 0.069 on real activations, and to **0.044 with Hadamard inputs, at the unchanged 1.49x kernel speed**.

Step scheduling (bf16 early steps) recovers some compounding — 0.733 → 0.840 at best — and an elaborate input-side stack (Hadamard + percentile scales + chunked per-block scales with LSE merging) cuts per-call error 30% yet **does not improve videos at all**. The ceiling was in the kernel; the fix had to be too.

## The kernel surgery

SageAttention's essential design — low-precision QKᵀ, high-precision P·V — turned out to be reachable inside FA4's own sm100 kernel, because the CuTe-DSL source is already dtype-parameterized per operand: the PV tiled-MMA and P's tmem layout key on `v_dtype`. Passing fp8 Q/K with bf16 V *almost* works out of the box. The gaps we closed (~90 lines, all no-ops for same-dtype kernels):

- Interface: accept the mixed combination, key `is_fp8` on Q's dtype, include V's dtype in the compile cache key, and view only actual-fp8 tensors as uint8 in the call path.
- Two P-path sites keyed on `q_dtype` where they meant "P's dtype" (the tmem store repetition and the register recast) — invisible until the two diverge.
- The fp8 register tuning and emulated-exp2 path gated to pure-fp8 (P is bf16 in mixed mode).
- The real bug: K/V share one smem pool by byte-aliasing V onto K's stages. With fp8-K and bf16-V the two pick **different swizzle families** at head_dim ≥ 96, and the load warp faults. Fix: in mixed mode V gets its own smem region after the K stages (per-stage budget K+V instead of max) — found with compute-sanitizer after eliminating six other hypotheses.

On top of the kernel, the vLLM-Omni backend adds a **Hadamard rotation on Q/K before quantization**, fused into the already-torch.compiled quant block: exact for QKᵀ (HHᵀ = I), no output correction needed since V is untouched, and it halves the remaining input error (0.065 → 0.032) by spreading per-channel outliers before amax scaling. This is FA3's fp8 "incoherent processing", finally paying off once the kernel stopped dominating the error budget.

Backend integration mirrors the compile-safety rules we learned the hard way: quantization lives in one `torch.compile`d function, the kernel sits behind a custom op (Dynamo cannot trace CuTe-DSL launchers), and nothing in forward mutates Python state — an innocuous per-call int counter guards-churns the regionally-compiled DiT block into silent eager fallback, which at one point made FP8 measure *slower* than bf16.

## Benchmark notes

Three measurement lessons, condensed (details in the earlier revision of this post, preserved in git history):

1. **Never A/B on tqdm's displayed `s/it`** — it is EMA-smoothed, so a ~40s first-step compile outlier biases every later reading against whichever config pays more one-time JIT; and a naive `s/it` grep also swallows the single-iteration VAE-decode bar. Use unsmoothed elapsed deltas — steady s/step = (elapsed@N/N − elapsed@1/N)/(N−1) — or segment profiler traces by attention-kernel count. Both methods agree here.
2. **Verify the GPU is idle** with per-process `nvidia-smi pmon` tagging; neighbor contention inflated early numbers by up to 84%, nonlinearly.
3. **Profiler traces are the ground truth** for where time goes: they showed FP8 removing 12.3% of total GPU work (and even the GEMMs speeding up ~10% from freed clock headroom), FA4's steady host cost at a negligible 72µs/call, and the one-time ~2–4s CuTe-DSL JIT that Inductor-style caches amortize away.

## User guide

Requirements: sm_100/sm_103, CUDA 13 torch, `pip install --pre flash-attn-4` — plus, for the mixed mode today, the patched flash-attention branch (upstreaming in progress; the wheel alone runs the bf16 backend).

**bf16 — drop-in, zero risk** (this is what vLLM-Omni [PR #4858](https://github.com/vllm-project/vllm-omni/pull/4858) ships):

```bash
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN_4 python examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Nano --num-frames 189 --num-inference-steps 35 --seed 42 ...
```

**Mixed fp8-QK / bf16-PV + Hadamard — the recommended lossy mode** (experimental flags today; a proper `extra` knob lands with the stacked PR):

```bash
OMNI_FA4_MIXED=1 OMNI_FA4_MIXED_HAD=1 \
python ... --diffusion-attention-config '{"default":{"backend":"FLASH_ATTN_4","extra":{"fp8":true}}}'
```

Recommendations: long video (≥100 frames) → mixed+Hadamard, ~1.1x e2e at floor-level quality, including reference-conditioned tasks (i2v 0.963; v2v matches the bf16 floor exactly). Short clips or high USP degrees → plain bf16 (the win compresses to noise). Stock full-FP8 remains available where throughput matters more than fidelity. Automatic safeguards unchanged: short-KV cross-attention, head_dim > 128, and masked/varlen calls always stay bf16; smooth-K always on.

## Developer guide & roadmap

Invariants for anyone extending this: (1) forward stays Dynamo-guard-stable — no per-call Python state; step-dependent decisions enter the graph as bools flipped from outside the compiled region; (2) JIT launchers stay behind custom ops; (3) quality claims are validated same-seed against the bf16 floor with relative error bounds — absolute tolerances hide diffuse-attention failures.

Remaining kernel roadmap, in ROI order:

- **INT8 QKᵀ** — s8×s8→s32 runs at the same 2x rate as fp8 on tcgen05; ~7 effective mantissa bits vs e4m3's 3; the 32-bit accumulator means the tmem plumbing is untouched, and the existing descale-folded-into-softmax-scale mechanism carries over. Closes the last ~0.02 SSIM at zero speed cost.
- **fp8 P·V with per-block P scales** (Sage2's trick) — recovers stock FP8's 1.49x kernel speed while keeping P precision; medium-heavy (touches the online-softmax correction path).
- **Upstreaming** — the mixed-mode patches are a clean per-operand-dtype generalization of FA4's sm100 kernel; the vLLM-Omni `extra` knob follows as a stacked PR once the kernel side is released.

Raw tables, reproduction commands, and the kernel microbenchmark live in [PR #4858](https://github.com/vllm-project/vllm-omni/pull/4858); the full investigation trail (error decompositions, the quality-ceiling experiments, all intermediate SSIM sweeps) is in this post's git history.
