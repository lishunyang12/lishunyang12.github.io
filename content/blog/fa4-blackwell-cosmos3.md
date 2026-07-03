---
slug: "/blog/fa4-blackwell-cosmos3"
date: "2026-07-03"
title: "FlashAttention 4 on B300: a Diffusion Attention Backend for vLLM-Omni"
description: "Bringing the FLASH_ATTN_4 backend to vLLM-Omni (PR #4858): kernel and end-to-end latency, profiler analysis, the FP8 quality trade-off, video results, plus a user guide and a developer guide — with two torch.compile traps and one measurement trap along the way."
---

Diffusion attention in vLLM-Omni on datacenter Blackwell (SM100/SM103) has had one main road: cuDNN via PyTorch SDPA. FlashAttention 4 ships JIT-compiled CuTe-DSL kernels with native Blackwell support and — the real prize — an FP8 path that SDPA cannot reach: cuDNN has FP8 FMHA in the library, but PyTorch SDPA does not expose it. This post is the full record of bringing the `FLASH_ATTN_4` backend into vLLM-Omni ([PR #4858](https://github.com/vllm-project/vllm-omni/pull/4858)): model **Cosmos3-Nano**, single **NVIDIA B300** (SM103), tasks t2v and i2v.

<div class="tldr">

**One-minute version.**

- **FA4 FP8 is 1.22–1.28x end-to-end** at 189 frames (t2v 2.55 → 2.00 s/step, i2v 2.36 → 1.91); FA4 bf16 is 1.03–1.08x. The gain scales with sequence length and shrinks to ~6% at 33 frames (~15k tokens).
- **FP8 costs quality**: same-seed i2v SSIM 0.73 against a 0.97 bf16 floor. `fp8_start_step` (bf16 for the first N denoising steps) recovers most of it — i2v 0.73 → 0.84, t2v 0.87 → 0.95 — while keeping ~2/3 of the speedup.
- Three engineering traps shaped the campaign: tqdm's smoothed rate readout measured FA4 as *slower*; one Python int counter in the attention module silently dropped the whole DiT block out of torch.compile; and `fp8_start_step` was a silent no-op on Cosmos3 until the pipeline learned to publish its step index.

</div>

## Video results

Same seed (42), official 35-step config, 1280x720, 33 frames. Within each row only the attention backend changes, so every visible difference is attributable to the backend. bf16 is visually indistinguishable from the cuDNN baseline; full-FP8 shows the quality drop that motivated the step schedule, and `fp8_start_step 12` recovers most of it.

<figure>
<div class="gal">
<div><video src="/videos/fa4/t2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN (baseline)</div></div>
<div><video src="/videos/fa4/t2v_fa4_bf16.mp4" controls muted loop playsinline></video><div class="c">FA4 bf16 · SSIM 0.975</div></div>
<div><video src="/videos/fa4/t2v_fa4_fp8.mp4" controls muted loop playsinline></video><div class="c">FA4 FP8 · SSIM 0.872</div></div>
<div><video src="/videos/fa4/t2v_fp8_start12.mp4" controls muted loop playsinline></video><div class="c">FP8 start 12 · SSIM 0.940</div></div>
</div>
<figcaption>Fig. 1 — t2v, foundry molten-metal pour (official Cosmos3 example prompt). SSIM vs the cuDNN baseline.</figcaption>
</figure>

<figure>
<div class="gal">
<div><video src="/videos/fa4/i2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN (baseline)</div></div>
<div><video src="/videos/fa4/i2v_fa4_bf16.mp4" controls muted loop playsinline></video><div class="c">FA4 bf16 · SSIM 0.968</div></div>
<div><video src="/videos/fa4/i2v_fa4_fp8.mp4" controls muted loop playsinline></video><div class="c">FA4 FP8 · SSIM 0.733</div></div>
<div><video src="/videos/fa4/i2v_fp8_start12.mp4" controls muted loop playsinline></video><div class="c">FP8 start 12 · SSIM 0.831</div></div>
</div>
<figcaption>Fig. 2 — i2v, dual robot-arm demo conditioned on the official reference image. Reference-conditioned tasks are the most FP8-sensitive.</figcaption>
</figure>

Every video above comes from one command template — only the backend selector changes between columns. Prompts, negative prompt, and the i2v reference image are the official Cosmos3 example assets (from [cosmos3-playground](https://github.com/lishunyang12/cosmos3-playground)); generation parameters are the paper defaults with only the frame count changed.

```bash
# Column 1 — cuDNN baseline (t2v; i2v adds --image i2v_robot.jpg and its own prompt)
python examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Nano \
  --prompt "$(cat prompt_t2v.json)" --negative-prompt "$(cat neg_prompts.json)" \
  --height 720 --width 1280 --fps 24 \
  --num-frames 33 --num-inference-steps 35 --guidance-scale 6.0 --seed 42 \
  --output t2v_cudnn.mp4

# Column 2 — FA4 bf16: identical flags, one env var
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN_4 python ...   # same flags as above
```

```python
# Columns 3 and 4 — FA4 FP8: identical flags, attention config on Omni
omni = Omni(model="nvidia/Cosmos3-Nano",
            diffusion_attention_config={"default": {
                "backend": "FLASH_ATTN_4",
                "extra": {"fp8": True},                      # column 3: FP8 from step 0
                # "extra": {"fp8": True, "fp8_start_step": 12},  # column 4
            }})
```

## Kernel level: bf16 is parity, FP8 is the whole point

Dense non-causal attention, head_dim 128, median of 50 iterations, idle GPU. Speedups vs cuDNN.

| Shape | cuDNN | FA4 bf16 | FA4 FP8 kernel | FP8 speedup |
|---|---|---|---|---|
| B1 H24 S14336 (HV-1.5-like) | 1462 µs | 1410 µs | 1164 µs | 1.26x |
| B1 H40 S16384 (Wan2.2-like) | 5741 µs | 5599 µs | 4960 µs | 1.16x |
| B1 H32 S21504 (Cosmos3, USP=4 per-rank) | 5037 µs | 4908 µs | 3565 µs | 1.41x |
| B1 H40 S32768 | 27632 µs | 27334 µs | 19521 µs | 1.42x |
| B1 H32 S86016 (Cosmos3, full sequence) | 80753 µs | 78770 µs | 54490 µs | 1.48x |

bf16 ties cuDNN at every shape (1.00–1.03x) — at hdim128 the sm100 tile space has exactly two legal configs, and the default is already optimal. **The case for FA4 on DC Blackwell is the FP8 path.**

FP8's quantization overhead (per-call amax + smooth-K + cast) is ~3 extra memory sweeps of Q/K/V in eager mode — enough to lose the kernel gain below ~32k tokens. Handing the quantization block to `torch.compile` (Inductor fuses smooth-K + amax + cast) made it **7–15x faster** (16k: 4716 → 635 µs; 86k: 18589 → 1221 µs), which moved the FP8 crossover down to every Cosmos3 deployment shape.

## End-to-end: the speedup scales with sequence length

Steady-state seconds per denoising step (first-step compile/JIT warmup excluded; idle GPU verified by per-process `nvidia-smi pmon` tagging).

<div class="chartbox">
<svg viewBox="0 0 720 258" xmlns="http://www.w3.org/2000/svg" class="chart" role="img" aria-label="189-frame end-to-end per-step latency">
  <text x="16" y="24" class="axlab" font-weight="700">t2v · 189f</text>
  <text x="112" y="54" class="tick" text-anchor="end">cuDNN</text>
  <rect x="120" y="40" width="549" height="20" fill="#6C7A89" rx="2"/>
  <text x="664" y="54" class="tick" text-anchor="end" fill="#fff">2.55 s</text>
  <text x="112" y="80" class="tick" text-anchor="end">FA4 bf16</text>
  <rect x="120" y="66" width="508" height="20" fill="#1B5A8E" rx="2"/>
  <text x="623" y="80" class="tick" text-anchor="end" fill="#fff">2.36 s · 1.08x</text>
  <text x="112" y="106" class="tick" text-anchor="end">FA4 FP8</text>
  <rect x="120" y="92" width="431" height="20" fill="#76b900" rx="2"/>
  <text x="546" y="106" class="tick" text-anchor="end" fill="#fff">2.00 s · 1.28x</text>
  <text x="16" y="148" class="axlab" font-weight="700">i2v · 189f</text>
  <text x="112" y="178" class="tick" text-anchor="end">cuDNN</text>
  <rect x="120" y="164" width="518" height="20" fill="#6C7A89" rx="2"/>
  <text x="633" y="178" class="tick" text-anchor="end" fill="#fff">2.36–2.45 s</text>
  <text x="112" y="204" class="tick" text-anchor="end">FA4 bf16</text>
  <rect x="120" y="190" width="489" height="20" fill="#1B5A8E" rx="2"/>
  <text x="604" y="204" class="tick" text-anchor="end" fill="#fff">2.27 s · 1.04–1.08x</text>
  <text x="112" y="230" class="tick" text-anchor="end">FA4 FP8</text>
  <rect x="120" y="216" width="411" height="20" fill="#76b900" rx="2"/>
  <text x="526" y="230" class="tick" text-anchor="end" fill="#fff">1.91 s · 1.24–1.28x</text>
</svg>
</div>

| Configuration | cuDNN | FA4 bf16 | FA4 FP8 |
|---|---|---|---|
| t2v · 189f · 12 steps | 2.55 s/step | 2.36 (1.08x) | 2.00 (**1.28x**) |
| i2v 704x1280 · 189f | 2.36–2.45 | 2.27 (1.04–1.08x) | 1.91 (**1.24–1.28x**) |
| t2v · 189f · profiler on (trace-derived) | 2.78 | 2.70 (1.03x) | 2.28 (1.22x) |
| t2v · 33f · 35 steps (official config) | 267 ms/step | 260 (1.03x) | 252 (1.06x) |
| i2v · 33f · 35 steps | 268 | 263 (1.02x) | 254 (1.06x) |

Quantization is O(S) while attention is O(S²), so FP8's net win stretches with length: 1.22–1.28x at 189 frames (~86k tokens), ~6% at 33 frames (~15k). Corollary: higher USP degrees shrink per-rank sequences back into the overhead-dominated regime — FP8's home turf is long videos and low parallelism.

## Profiler deep-dive

torch-profiler captures (t2v, 189f, clean-verified), GPU kernel time summed over one profiled window per config:

| GPU time bucket | cuDNN | FA4 bf16 | FA4 FP8 |
|---|---|---|---|
| attention kernels | 12.69 s | 12.14 s | 9.28 s |
| quantize / cast | 2.18 s | 2.18 s | 2.97 s |
| GEMM (nvjet / cutlass) | 7.94 s | 7.91 s | 7.16 s |
| elementwise + norm + other | 3.47 s | 3.50 s | 3.65 s |
| **total GPU busy** | **26.28 s** | **25.73 s** | **23.06 s** |

| Host-side metric | cuDNN | FA4 bf16 | FA4 FP8 |
|---|---|---|---|
| attention op host cost, p50 | 28 µs | 72 µs | 63 µs |
| attention op host cost, max | 32 ms | 1.84 s (one-time JIT) | 1.81 s (one-time JIT) |
| trace-derived steady s/step | 2.78 | 2.70 | 2.28 |

Four takeaways:

1. **FP8 removes 12.3% of all GPU work** (26.28 → 23.06 s) — 3.4 s of attention time minus 0.8 s of extra quantization kernels.
2. **Even the GEMMs speed up ~10% under FP8** (7.94 → 7.16 s, identical launch counts). The cheaper attention kernel frees power/thermal headroom and the GEMMs clock higher — a second-order win you only see at the trace level.
3. **FA4's steady host cost is negligible**: 72 µs/call vs cuDNN's 28 µs, ~3.6 ms/step against 2+ s steps. The scary multi-millisecond *average* is one 1.84 s first-call CuTe-DSL JIT compile skewing the mean — a one-time cost, amortized by Inductor/JIT caches on subsequent runs.
4. The trace-derived per-step walls independently confirm the elapsed-time method (row 3 of the e2e table) — two measurement paths, same ordering.

## The measurement trap that almost killed the backend

The first e2e verdict was "FA4 bf16 is **4% slower** than cuDNN" — and it was pure artifact, twice over:

1. **tqdm's displayed `s/it` is EMA-smoothed.** The ~40 s first-step torch.compile outlier decays as 0.7^k into every later reading — still ~0.8 s of bias at step 12 — and it systematically penalizes whichever config pays more one-time warmup (FA4 pays an extra 2–4 s of CuTe-DSL JIT).
2. The log also contains a **single-iteration VAE-decode progress bar** (38–58 s/it) that a naive `s/it` grep swallows into the median.

Re-measuring with two independent methods — (a) unsmoothed progress-bar elapsed deltas, steady s/step = (elapsed@N/N − elapsed@1/N) / (N−1), and (b) profiler-trace timelines segmented by attention-kernel count (81 calls/step) — flipped the verdict: bf16 from −4% to +3–8%, FP8 from +2% to +22–28%. Both methods agree.

**Lesson: never A/B on a smoothed rate readout, especially when warmup costs are asymmetric.**

## Two torch.compile traps

**Trap 1: a Python int counter in the attention module.** The pipeline regionally compiles the DiT block (36 blocks). The backend originally kept `self._fp8_calls += 1` in forward; Dynamo guards module int attributes *by value*, so every step produced a new guard, eight steps hit `recompile_limit`, and the whole block silently fell back to eager. The symptom was maximally misleading — **FP8 measured slower than bf16** (~0.3 s/step worse) — and the root cause had nothing to do with quantization. Deleting the counter fixed it; the backend's forward is now guard-stable.

**Trap 2: `fp8_start_step` was a silent no-op — and the fix has a shape.** The option runs the first N steps in bf16 (early steps set global composition and are the most quantization-sensitive), then switches to FP8. The original implementation read `denoise_step_idx` from the ForwardContext — but the Cosmos3 pipeline never published that index, so the knob did nothing (three different start values produced bit-identical videos). The fix has two halves:

- The Cosmos3 pipeline now publishes `set_forward_context_denoise_step_idx(i)` in each of its denoise loops (and resets to `None` after), mirroring HunyuanImage3.
- The backend does **not** read the step integer in forward — that would re-create Trap 1 (int guards specialize per value; one recompile per step). Instead the ForwardContext exposes a listener registry, and the callback flips a plain bool on each impl *outside* the compiled region; a bool guard has exactly two compiled variants. The default is conservative: with `fp8_start_step > 0` and no step ever published, FP8 stays off — a pipeline without the hook degrades to bf16 speed, never to silent quality loss.

## The FP8 quality account

Same seed, official config (35 steps, 1280x720, 33 frames), SSIM vs the cuDNN baseline. bf16's 0.97 is the same-trajectory kernel-noise floor — the practical reference point.

| Configuration | t2v SSIM | i2v SSIM |
|---|---|---|
| FA4 bf16 (floor) | 0.975 | 0.968 |
| FP8 from step 0 | 0.872 | 0.733 |
| FP8 start 5 / 35 | 0.911 | 0.804 |
| FP8 start 12 / 35 | 0.940 | 0.831 |
| FP8 start 18 / 35 | 0.949 | 0.840 |

Three observations:

1. **i2v suffers visibly more than t2v** — reference-conditioned tasks accumulate error against a fixed image, so quantization noise has nowhere to hide.
2. `fp8_start_step` recovers quality monotonically, and **the earliest steps carry the most weight** (start 5 already claws back nearly half the gap) — confirming the early-step-sensitivity hypothesis. The speed cost is linear and mild: at 189 frames FP8 saves ~0.55 s/step, so start 12/35 keeps roughly 2/3 of the speedup.
3. The curve flattens (i2v: 5→12 gains +0.027, 12→18 only +0.009). The schedule can only reclaim the *compounding* of early-step error; the residual is a floor — and the next section shows why no input-side technique moves it.

## Where the FP8 error really lives — and the quality ceiling

The obvious next move was finer-grained quantization (SageAttention-style blockwise scales). Before building it, we decomposed the error on **captured real Cosmos3 activations** (relative error vs the bf16 kernel, GQA 32/8, ~10k KV tokens):

| Error source | rel. error |
|---|---|
| Input quantization alone (current per-head scales, quantize-dequantize through the bf16 kernel) | 0.050–0.067 |
| Input quant at SageAttention-style 128-token blocks | 0.065 (no help) |
| Input quant with Hadamard rotation (FA3-style incoherent processing) | 0.038 |
| **Real FP8 kernel, same scales** | **0.237** |
| Real FP8 kernel on pure random gaussian inputs | 0.111 @ 1k → 0.158 @ 10k tokens |
| Best input-side stack (Hadamard + smooth-K + per-chunk K/V scales + LSE-merged chunked calls) | 0.166 |

The decomposition overturns the granularity hypothesis: input scales contribute only ~0.06 of the 0.24, and **the FP8 kernel itself carries an ~11–16% intrinsic relative error on diffuse attention — even on perfect gaussian inputs**. The mechanism is e4m3's 3-bit mantissa on the softmax matrix P: video DiT attention is *diffuse* (probabilities ~1e-4 across ~10k tokens), the regime where FP8 P-rounding is weakest. LLM attention is sharp, which is why FP8 attention "works" there — the FA3-on-Hopper accumulation story, in a new costume.

We then built the best input-side stack anyway and ran it end-to-end (videos below): it cuts per-call error 30% (0.237 → 0.166) yet **does not improve video quality** — i2v SSIM 0.839 vs 0.831 for plain FP8 at the same schedule, t2v actually drops (0.925 vs 0.940), and from step 0 it is markedly worse (0.588 vs 0.733). Mean per-call error is not what the diffusion trajectory cares about; the stack's error is more *structured* (chunk-coherent attention shifts, rotated-basis noise), and structured error compounds worse.

**Conclusion: the practical quality ceiling for FA4 FP8 on DiT today is plain per-head quantization + `fp8_start_step` — the shipped configuration.** Going past it requires kernel-level changes: a higher-precision P path (fp16-P·V or INT8-QKᵀ, exactly SageAttention's design — whose sm_120 kernels do not run on datacenter sm_100). We are filing the gaussian-input repro upstream; a relative-tolerance fix to our own unit test is also in order, since an absolute 5e-3 bound at diffuse magnitudes silently admits ~12% relative kernel error.

<figure>
<div class="gal">
<div><video src="/videos/fa4/i2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN (baseline)</div></div>
<div><video src="/videos/fa4/i2v_fp8_start12.mp4" controls muted loop playsinline></video><div class="c">FP8 start 12 · SSIM 0.831</div></div>
<div><video src="/videos/fa4/i2v_fp8_start18.mp4" controls muted loop playsinline></video><div class="c">FP8 start 18 · SSIM 0.840</div></div>
<div><video src="/videos/fa4/i2v_fp8_best_start12.mp4" controls muted loop playsinline></video><div class="c">max-quality stack, start 12 · SSIM 0.839</div></div>
</div>
<figcaption>Fig. 3 — the quality ceiling, i2v. The elaborate input-side stack (rightmost) lands on top of plain FP8 with the same step schedule: the FP8 kernel's intrinsic diffuse-attention error is the binding constraint, not our quantization.</figcaption>
</figure>

## Breaking the ceiling: a mixed-precision kernel (fp8 QKᵀ + bf16 P·V)

If the binding constraint is in-kernel, change the kernel. SageAttention's essential design — low-precision QKᵀ, high-precision P·V — turned out to be reachable *inside* FA4's own CuTe-DSL sm100 kernel, because the code is already dtype-parameterized per operand (the PV tiled-MMA and the P tmem layout key on `v_dtype`). We built the mixed mode on a branch of flash-attention: fp8 Q/K with per-head descales feed the QKᵀ MMA; P is converted to bf16 and the P·V MMA runs in bf16 against unquantized V. About 90 changed lines, all no-ops for existing same-dtype kernels.

Two of the bugs en route are worth recording. Several P-path sites keyed on `q_dtype` where they meant "P's dtype" (the tmem store repetition, the register recast) — invisible until the two diverge. And the K/V shared-smem design byte-aliases V onto K's stages, which is unsound when their swizzle families differ: at head_dim ≥ 96, fp8-K and bf16-V pick different swizzle atoms and the load warp faults. Fix: in mixed mode V gets its own smem region after the K stages (budgeted K+V per stage instead of max).

Adding a Hadamard rotation on Q/K before quantization (exact for QKᵀ, no output correction needed since V is untouched, fused into the compiled quant block at negligible cost) then halves the remaining input-quantization error. The combined result:

| Metric | full FP8 | mixed | **mixed + Hadamard** |
|---|---|---|---|
| rel. error, real Cosmos3 activations | 0.237 | 0.065 | **0.032** |
| kernel latency @21.5k–86k vs cuDNN | 1.49x | 1.24–1.25x | ~1.24x |
| e2e @189f t2v vs cuDNN | 1.22–1.28x | ~1.08–1.12x | ~1.1x |
| **e2e SSIM, i2v, step 0, no schedule** | 0.733 | 0.926 | **0.963** (floor 0.968) |
| **e2e SSIM, t2v, step 0, no schedule** | 0.872 | 0.910 | **0.950** (floor 0.975) |

i2v lands 0.005 below the bf16 same-seed floor — visually-indistinguishable territory, with no step schedule and no latency give-back beyond the P·V precision itself. Zero Dynamo recompiles in the compiled pipeline. The remaining roadmap inside the kernel: INT8 QKᵀ (s8×s8→s32 runs at the same 2x rate; ~7 effective mantissa bits vs e4m3's 3; the 32-bit accumulator keeps the tmem plumbing unchanged) to close the last gap, and Sage2-style fp8 P·V with per-block P scales to recover full-FP8's 1.49x kernel speed.

<figure>
<div class="gal">
<div><video src="/videos/fa4/i2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN (baseline)</div></div>
<div><video src="/videos/fa4/i2v_fa4_fp8.mp4" controls muted loop playsinline></video><div class="c">full FP8 · SSIM 0.733</div></div>
<div><video src="/videos/fa4/i2v_fa4_mixed.mp4" controls muted loop playsinline></video><div class="c">mixed kernel · SSIM 0.926</div></div>
<div><video src="/videos/fa4/i2v_fa4_mixed_had.mp4" controls muted loop playsinline></video><div class="c">mixed + Hadamard · SSIM 0.963</div></div>
</div>
<figcaption>Fig. 4 — breaking the ceiling on i2v: no step schedule, same seed, official config. bf16 floor is 0.968. t2v analog: 0.872 → 0.910 → 0.950 (video: <a href="/videos/fa4/t2v_fa4_mixed_had.mp4">t2v_fa4_mixed_had</a>).</figcaption>
</figure>

Status: the kernel changes live on a flash-attention branch (upstreaming planned — they are a clean per-operand-dtype generalization); the vLLM-Omni backend knob will follow as a stacked PR once the kernel side is released.

## User guide

**Requirements**: datacenter Blackwell GPU (sm_100/sm_103), CUDA 13 torch build, and:

```bash
pip install --pre flash-attn-4
```

**bf16 — drop-in, zero risk.** Same quality band as cuDNN (SSIM 0.97 same-seed), never slower, 3–8% faster on long sequences. One env var:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN_4 \
python examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Nano --num-frames 189 --num-inference-steps 35 --seed 42 ...
```

**FP8 — the speed lever, with a schedule.** Configure through the attention config; `fp8_start_step` at roughly N/3 of your step count is the current sweet spot:

```python
omni = Omni(
    model="nvidia/Cosmos3-Nano",
    diffusion_attention_config={"default": {
        "backend": "FLASH_ATTN_4",
        "extra": {"fp8": True, "fp8_start_step": 12},   # for 35 steps
    }},
)
```

**Recommendations by workload:**

| Workload | Recommendation |
|---|---|
| Long t2v (≥ 100 frames), throughput matters | FP8 + `fp8_start_step ≈ N/3` — ~2/3 of a 1.22–1.28x speedup |
| i2v / reference-conditioned | Eyeball first; FP8 is most visible here. bf16 if in doubt |
| Short clips (≤ 33 frames) or high USP degree | Plain bf16 — FP8's win compresses to noise |
| Zero quality tolerance | FA4 bf16 (or stay on cuDNN; they are equivalent) |

**Automatic safeguards** you do not need to configure: short-KV calls (text cross-attention) always stay bf16, head dims above 128 stay bf16, smooth-K is always on, and masked/varlen calls never quantize.

## Developer guide

For anyone extending the backend or wiring FP8 scheduling into another pipeline — three invariants and one recipe.

**Invariant 1: forward must stay Dynamo-guard-stable.** No per-call Python state mutation (counters, caches keyed by step) in anything the DiT block's compiled region can see. Int attributes guard by value; one new value per step means one recompile per step, `recompile_limit` is 8, and the failure mode is a *silent* eager fallback that shows up as an inexplicable slowdown.

**Invariant 2: step-dependent decisions enter the graph as bools.** The step index lives on the ForwardContext; backends register a listener (`register_denoise_step_listener`) and flip a bool attribute on their impls from outside the compiled region. Two bool states = two compiled variants, and switching between them is guard dispatch, not recompilation.

**Invariant 3: keep JIT launchers behind custom ops.** The CuTe-DSL entry points are wrapped as `torch.library.custom_op`s with fake kernels — Dynamo cannot trace the cutlass JIT launcher, and the op boundary is what lets the quantization block compile while the kernel call stays opaque.

**Recipe: enabling `fp8_start_step` for a new pipeline** is one hook — publish the step index from your denoise loop:

```python
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx

for i, t in enumerate(timesteps):
    set_forward_context_denoise_step_idx(i)
    ...
set_forward_context_denoise_step_idx(None)   # reset after the loop
```

Without the hook nothing breaks: FP8 simply stays off when `fp8_start_step > 0` (conservative default — bf16 speed, not silent quality loss). Cosmos3 and HunyuanImage3 publish today.

**Testing**: `tests/diffusion/attention/test_flash_attn4.py` covers SDPA parity, cross-attention shapes, masked/varlen, FP8-vs-bf16 tolerance (incl. GQA scales), the short-KV and head-256 fallbacks, the step gate, and platform selection/fallback.

**Benchmarking**: measure steady s/step from unsmoothed progress-bar elapsed deltas or trace-timeline segmentation — never from tqdm's displayed `s/it` (see the measurement trap above). Verify the GPU is idle with per-process `nvidia-smi pmon` tagging; neighbor contention inflated our early numbers by up to 84%.

Full reproduction commands, the kernel microbenchmark script, and raw tables live in [PR #4858](https://github.com/vllm-project/vllm-omni/pull/4858).
