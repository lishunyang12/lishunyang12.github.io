---
slug: "/blog/fa4-blackwell-cosmos3"
date: "2026-07-03"
title: "FP8 Attention on B300: Videos, Accuracy, Latency"
description: "Cosmos3 video generation with FlashAttention-4 FP8 attention on a single B300 — same-seed videos, SSIM accuracy, and kernel/end-to-end latency for every mode."
---

All results: **Cosmos3-Nano** on a single **NVIDIA B300** (sm_103) in vLLM-Omni, official generation configs, same seed (42) within each task, idle-GPU verified. Baseline is cuDNN (the platform default). **FP8 (fixed) + Hadamard** is the recommended mode: FlashAttention-4 FP8 with the `max_offset` saturation fix ([Dao-AILab/flash-attention#2694](https://github.com/Dao-AILab/flash-attention/pull/2694)) and Hadamard-rotated Q/K/V. **Mixed + Hadamard** (fp8 QKᵀ, bf16 P·V) is the conservative alternative. Code: [vllm-omni#4858](https://github.com/vllm-project/vllm-omni/pull/4858) (bf16 backend), [lishunyang12/flash-attention#1](https://github.com/lishunyang12/flash-attention/pull/1) (kernel work).

## Accuracy (same-seed SSIM vs cuDNN baseline)

| Task · config | stock FP8 | mixed + Hadamard | **FP8 (fixed) + Hadamard** | FA4 bf16 (floor) |
|---|---|---|---|---|
| t2v · 33f · 35 steps | 0.872 | 0.950 | **0.951** | 0.975 |
| i2v · 33f · 35 steps | 0.733 | 0.963 | **0.965** | 0.968 |
| t2v · 189f · 35 steps | — | 0.970 | — | — |
| v2v · 121f · 50 steps | — | 0.8115 | — | 0.8115 |

Attention-kernel relative error on real Cosmos3 activations (vs the bf16 kernel): stock FP8 **0.237** → mixed + Hadamard **0.032** → FP8 (fixed) + Hadamard **0.044**. The v2v row's lower absolute SSIM is step-count trajectory divergence, not quantization: the bf16 control lands at the identical 0.8115.

## Latency

| | cuDNN | FA4 bf16 | mixed + Hadamard | **FP8 (fixed) + Hadamard** |
|---|---|---|---|---|
| kernel · 21.5k tokens | 5058 µs | 4901 µs (1.03x) | 4048 µs (1.25x) | **3402 µs (1.49x)** |
| kernel · 86k tokens | 81.4 ms | 78.3 ms (1.04x) | 65.4 ms (1.24x) | **54.5 ms (1.49x)** |
| e2e · 189f t2v, s/step | 2.45–2.55 | 2.36 (1.08x) | ~2.29 (~1.1x) | **2.00 (1.28x)** |
| e2e · 33f, ms/step | 267 | 260 (1.03x) | — | **252 (1.06x)** |

Steady-state per denoising step, first-step compile/JIT excluded, measured from unsmoothed elapsed timestamps and cross-checked against profiler-trace timelines. FP8's advantage scales with sequence length (quantization is O(S), attention O(S²)); 189 frames ≈ 86k tokens is its home turf.

## Videos

### i2v · official config · same seed

<figure>
<div class="gal">
<div><video src="/videos/fa4/i2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN · baseline</div></div>
<div><video src="/videos/fa4/i2v_fa4_fp8.mp4" controls muted loop playsinline></video><div class="c">stock FP8 · SSIM 0.733</div></div>
<div><video src="/videos/fa4/i2v_fa4_mixed_had.mp4" controls muted loop playsinline></video><div class="c">mixed + Hadamard · 0.963</div></div>
<div><video src="/videos/fa4/i2v_fa4_fp8fixed_had.mp4" controls muted loop playsinline></video><div class="c">FP8 (fixed) + Hadamard · 0.965</div></div>
</div>
<figcaption>Image-to-video, 33 frames, 35 steps, 1280x720. bf16 floor: 0.968.</figcaption>
</figure>

### t2v · official config · same seed

<figure>
<div class="gal">
<div><video src="/videos/fa4/t2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN · baseline</div></div>
<div><video src="/videos/fa4/t2v_fa4_fp8.mp4" controls muted loop playsinline></video><div class="c">stock FP8 · SSIM 0.872</div></div>
<div><video src="/videos/fa4/t2v_fa4_mixed_had.mp4" controls muted loop playsinline></video><div class="c">mixed + Hadamard · 0.950</div></div>
<div><video src="/videos/fa4/t2v_fa4_fp8fixed_had.mp4" controls muted loop playsinline></video><div class="c">FP8 (fixed) + Hadamard · 0.951</div></div>
</div>
<figcaption>Text-to-video, 33 frames, 35 steps, 1280x720. bf16 floor: 0.975.</figcaption>
</figure>

### Long video and v2v

<figure>
<div class="gal">
<div><video src="/videos/fa4/t2v189_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN · t2v 189f / 35 steps</div></div>
<div><video src="/videos/fa4/t2v189_mixed_had.mp4" controls muted loop playsinline></video><div class="c">quantized · SSIM 0.970</div></div>
<div><video src="/videos/fa4/v2v_cudnn.mp4" controls muted loop playsinline></video><div class="c">cuDNN · v2v 121f / 50 steps</div></div>
<div><video src="/videos/fa4/v2v_mixed_had.mp4" controls muted loop playsinline></video><div class="c">quantized · SSIM 0.811 (= bf16 floor)</div></div>
</div>
<figcaption>Left: long-video t2v at the full official config (~86k-token attention). Right: v2v continuation conditioned on the official reference clip.</figcaption>
</figure>
