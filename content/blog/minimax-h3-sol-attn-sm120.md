---
slug: "/blog/minimax-h3-sol-attn-sm120"
date: "2026-08-11"
title: "MiniMax-H3 Sol-Attn on SM120: Speed and Quality"
description: "Same-seed Dense cuDNN and Sol-Attn videos on 4×SM120, with preset parameters, latency, SSIM, PSNR, and LPIPS."
---

This page hosts the calibration snapshot attached in
[vllm-omni-rankings commit `bce9339`](https://github.com/lishunyang12/vllm-omni-rankings/commit/bce933957ed957dab7f743ef81afbde29db27fe9).
It compares an exact Dense cuDNN reference with three Sol-Attn configurations
for MiniMax-H3 FL2VA on four NVIDIA SM120 GPUs. All outputs use the same prompt,
seed, sampling settings, and BF16 model weights.

<div class="callout"><strong>Result:</strong> Recommended delivers 1.097× speedup with LPIPS 0.08563. Medium reaches 1.140× and still passes every quality gate. Aggressive reaches 1.205× but fails the LPIPS ≤ 0.20 gate.</div>

## Same-seed video and audio

<figure>
<div class="gal gal-2">
<div><video src="/videos/sol-attn-sm120/dense-cudnn.mp4" controls loop playsinline preload="metadata"></video><div class="c"><strong>Dense cuDNN</strong> · 142.49 s · reference</div></div>
<div><video src="/videos/sol-attn-sm120/sol-recommended.mp4" controls loop playsinline preload="metadata"></video><div class="c"><strong>Sol Recommended</strong> · 1.097× · LPIPS 0.08563</div></div>
<div><video src="/videos/sol-attn-sm120/sol-medium.mp4" controls loop playsinline preload="metadata"></video><div class="c"><strong>Sol Medium</strong> · 1.140× · LPIPS 0.11771</div></div>
<div><video src="/videos/sol-attn-sm120/sol-aggressive.mp4" controls loop playsinline preload="metadata"></video><div class="c"><strong>Sol Aggressive</strong> · 1.205× · LPIPS 0.22556</div></div>
</div>
<figcaption>Run 1 from each case, seed 1101. Every clip is 1344×768 H.264 with model-generated AAC audio; use the player controls to listen. Quality metrics compare each Sol-Attn clip against the Dense clip above.</figcaption>
</figure>

Prompt:

> At night, while their owner sleeps, three cats march into a bedroom playing tiny brass instruments, freeze, and quietly march out.

## Speed–quality result

| Configuration | Median generation | Speedup | Latency reduction | SSIM ↑ | PSNR ↑ | LPIPS ↓ | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Dense cuDNN | 142.49 s | 1.000× | 0.00% | 1.00000 | ∞ | 0.00000 | Reference |
| **Sol Recommended** | 129.91 s | 1.097× | 8.83% | 0.93344 | 33.28 | 0.08563 | **PASS** |
| **Sol Medium** | 124.95 s | 1.140× | 12.31% | 0.91780 | 31.49 | 0.11771 | **PASS** |
| Sol Aggressive | 118.29 s | 1.205× | 16.99% | 0.87586 | 27.00 | 0.22556 | **FAIL** |

The quality gates are SSIM ≥ 0.82, PSNR ≥ 20 dB, and LPIPS ≤ 0.20.
Latency is the median of three measured generation runs after one warmup; video
encoding is excluded. SSIM and PSNR use all decoded frames. LPIPS uses up to 16
uniformly sampled frames resized to 256×256.

## Exact configurations

These labels are benchmark presets, not built-in vLLM-Omni preset names.

| Parameter | Recommended | Medium | Aggressive |
|---|---:|---:|---:|
| `tau` | 1.0 | 1.5 | 2.0 |
| `dense_steps` | 10 | 8 | 5 |
| `thresh_type` | `diag` | `diag` | `diag` |
| `sink_tokens` / `sink_start` | 951 / 0 | 951 / 0 | 951 / 0 |
| `dense_layers` | `"0,1"` | `"0,1"` | `"0,1"` |
| `kv_splits` | 1 | 1 | 1 |

Higher `tau` selects fewer KV blocks for exact attention. Lower `dense_steps`
switches from Dense Attention to Sol-Attn earlier in the 20-step denoising
trajectory. Medium changes both controls from the Recommended point; Aggressive
pushes both further.

## Test setup and provenance

| Item | Setting |
|---|---|
| Model | MiniMax-H3 / FL2VA, BF16 |
| Hardware | 4× NVIDIA SM120 GPUs |
| Parallelism | TP4; text encoder TP4 |
| Output | 1344×768, 24 fps, requested duration 5.0 s |
| Sampling | 20 inference steps; `flow_shift=12.0`; `audio_flow_shift=3.0` |
| Repetitions | 1 warmup + 3 measured runs per configuration |
| Quality pair | Run 1, seed 1101, each Sol output vs Dense cuDNN |

The source commit contains all 12 MP4 files plus the exact worker configs,
timing JSON, and logs. The implementation is tracked in
[vLLM-Omni PR #5851](https://github.com/vllm-project/vllm-omni/pull/5851).

<div class="callout"><strong>Scope:</strong> this is a single-prompt, single-seed calibration snapshot, not a broad video-generation benchmark. It is useful for comparing these exact settings, but release claims should be revalidated across multiple prompts and seeds. The measurements also predate the final sink-query dense-recomputation change on the PR head.</div>
