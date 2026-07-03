---
slug: "/blog/fa4-blackwell-cosmos3"
date: "2026-07-03"
title: "FlashAttention 4 on B300：vLLM-Omni 扩散注意力后端实测"
description: "为 vLLM-Omni 引入 FLASH_ATTN_4 扩散注意力后端（PR #4858）：kernel 与端到端延迟、FP8 质量取舍、两个 torch.compile 陷阱与一个测量陷阱的完整记录。"
---

vLLM-Omni 在数据中心 Blackwell（SM100/SM103）上的扩散注意力此前只有 cuDNN（经 PyTorch SDPA）一条主路。FlashAttention 4 以 CuTe-DSL JIT kernel 原生支持 Blackwell，并带有 cuDNN 在 SDPA 里拿不到的 FP8 路径——cuDNN 库内有 FP8 FMHA，但 PyTorch SDPA 并未暴露。本文是 [PR #4858](https://github.com/vllm-project/vllm-omni/pull/4858) 引入 `FLASH_ATTN_4` 后端的完整实测记录：模型 **Cosmos3-Nano**，单卡 **NVIDIA B300**（SM103），任务 t2v 与 i2v。

<div class="tldr">

**一分钟版。**

- **FA4 FP8 端到端 1.22–1.28x**（189 帧：t2v 2.55 → 2.00 s/step，i2v 2.36 → 1.91）；FA4 bf16 为 1.03–1.08x。收益随序列长度增长，33 帧（~15k token）时收窄到 ~6%。
- **FP8 有质量代价**：同 seed 下 i2v SSIM 0.73（bf16 底线 0.97）。`fp8_start_step`（前 N 步 bf16、之后 FP8）可回收一大截：i2v 0.73 → 0.84，t2v 0.87 → 0.94+。
- 三个工程陷阱贯穿全程：tqdm 平滑均值把 FA4 测成了"更慢"；attention 模块里一个 Python int 计数器让整个 DiT block 掉回 eager；`fp8_start_step` 在 Cosmos3 上曾是静默 no-op。

</div>

## Kernel 层：bf16 打平，FP8 是全部的意义

稠密非因果 attention，head_dim 128，50 次迭代取中位，空闲 GPU。加速比相对 cuDNN。

| 形状 | cuDNN | FA4 bf16 | FA4 FP8 kernel | FP8 加速 |
|---|---|---|---|---|
| B1 H24 S14336（HV-1.5 类） | 1462 µs | 1410 µs | 1164 µs | 1.26x |
| B1 H40 S16384（Wan2.2 类） | 5741 µs | 5599 µs | 4960 µs | 1.16x |
| B1 H32 S21504（Cosmos3, USP=4 每 rank） | 5037 µs | 4908 µs | 3565 µs | 1.41x |
| B1 H40 S32768 | 27632 µs | 27334 µs | 19521 µs | 1.42x |
| B1 H32 S86016（Cosmos3, 完整序列） | 80753 µs | 78770 µs | 54490 µs | 1.48x |

bf16 与 cuDNN 全形状打平（1.00–1.03x）——sm100 的 tile 配置空间在 hdim128 下只有 (128,128)/(128,96) 两个合法选项，默认即最优，没有再调的余地。**FA4 在 DC Blackwell 上的意义就是 FP8 路径。**

FP8 的量化开销（per-call amax + smooth-K + cast）在 eager 下是 ~3 次额外的 Q/K/V 显存扫描，短序列会亏掉 kernel 收益。后端把量化块交给 `torch.compile`（Inductor 融合 smooth-K + amax + cast）后快了 **7–15 倍**（16k：4716 → 635 µs；86k：18589 → 1221 µs），FP8 端到端在所有 Cosmos3 部署形状上转正。

## 端到端：加速随序列长度伸缩

稳态每去噪步秒数（剔除首步 compile/JIT 预热；空闲 GPU 经逐进程 `nvidia-smi pmon` 标记核验）。

<div class="chartbox">
<svg viewBox="0 0 720 258" xmlns="http://www.w3.org/2000/svg" class="chart" role="img" aria-label="189 帧端到端每步耗时条形图">
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

| 配置 | cuDNN | FA4 bf16 | FA4 FP8 |
|---|---|---|---|
| t2v · 189f · 12 steps | 2.55 s/step | 2.36（1.08x） | 2.00（**1.28x**） |
| i2v 704x1280 · 189f | 2.36–2.45 | 2.27（1.04–1.08x） | 1.91（**1.24–1.28x**） |
| t2v · 189f · 开 profiler（trace 逐步墙钟） | 2.78 | 2.70（1.03x） | 2.28（1.22x） |
| t2v · 33f · 35 steps（官方推荐配置） | 267 ms/step | 260（1.03x） | 252（1.06x） |
| i2v · 33f · 35 steps | 268 | 263（1.02x） | 254（1.06x） |

量化是 O(S)、attention 是 O(S²)，所以 FP8 的净收益随序列伸缩：189 帧（~86k token）1.22–1.28x，33 帧（~15k）收窄到 ~6%。推论：USP 并行度越高、每 rank 序列越短，FP8 越不划算——它的主场是长视频与低并行度。

Profiler trace 还给了一个佐证：FP8 窗口内总 GPU busy 时间比 cuDNN 低 **12.3%**（23.06 vs 26.28 s），连 nvjet GEMM 都变快了 ~12%——更省电的 attention kernel 给 GEMM 让出了时钟余量。FA4 稳态 host 开销 72 µs/call（cuDNN SDPA 为 28 µs），在 2 s 级步长下可忽略。

## 测量陷阱：tqdm 差点杀掉这个后端

最初的结论是"FA4 bf16 比 cuDNN **慢 4%**"——差点就以此收场。它是两个叠加的测量伪影：

1. **tqdm 显示的 `s/it` 是 EMA 平滑值**。首步 ~40 s 的 torch.compile 离群点以 0.7^k 衰减渗进后面每一个读数，到第 12 步仍有 ~0.8 s 的偏置——并且系统性地惩罚预热更重的一方（FA4 多付 2–4 s 的一次性 CuTe-DSL JIT）。
2. 日志里还有一个**单迭代的 VAE decode 进度条**（38–58 s/it），朴素的 `s/it` grep 会把它一起吞进中位数。

改用两种独立方法后互相印证、结论翻转：(a) 进度条未平滑的耗时戳，稳态 s/step = (elapsed@N/N − elapsed@1/N) / (N−1)；(b) 按每步 attention kernel 数（81 个/步）切分 profiler trace 时间线取逐步墙钟。bf16 从 −4% 翻到 +3–8%，FP8 从 +2% 翻到 +22–28%。

**教训：不要用任何平滑后的速率读数做 A/B 基准，尤其当两侧预热成本不对称时。**

## 两个 torch.compile 陷阱

**陷阱一：attention 模块里的 Python int 计数器。** 管线对 DiT block 做 regional compile（36 个 block）。后端最初在 forward 里维护 `self._fp8_calls += 1`，Dynamo 对模块 int 属性按值建 guard——每步一个新值，每步一次重编译，8 次撞上 `recompile_limit`，整个 block 静默掉回 eager。症状极具迷惑性：**FP8 测出来比 bf16 还慢**（多付 ~0.3 s/step），根因却与量化毫无关系。删掉计数器、保持 forward guard-stable 后归零。

**陷阱二：`fp8_start_step` 的静默 no-op 与它的正确修法。** 该选项让前 N 步跑 bf16（早期步决定全局构图，对量化噪声最敏感）、之后切 FP8。最初实现从 ForwardContext 读 `denoise_step_idx`——但 Cosmos3 管线根本不发布这个索引，选项静默失效（三个不同 start 值生成了逐位相同的视频）。修复分两半：

- Cosmos3 管线在 4 个去噪循环里逐步发布 `set_forward_context_denoise_step_idx(i)`；
- 后端**不在 forward 里读步数整数**——那会重蹈陷阱一（int guard 按值特化，每步重编译）。改为 ForwardContext 提供监听器注册，回调在编译区域**之外**翻转各 impl 上的一个普通 bool；bool guard 只有两个编译变体。另外把默认值改保守：`fp8_start_step > 0` 且管线从未发布步数时，FP8 保持关闭——宁可退化成 bf16 的速度，不静默退化成质量损失。

## FP8 的质量账

同 seed、官方推荐配置（35 steps，1280x720，33 帧），SSIM 相对 cuDNN 基线。bf16 的 0.97 是"同一条轨迹的 kernel 噪声底线"，可作参照系。

| 配置 | t2v SSIM | i2v SSIM |
|---|---|---|
| FA4 bf16（底线） | 0.975 | 0.968 |
| FP8 全程（start 0） | 0.872 | 0.733 |
| FP8 start 5 / 35 | 0.911 | 0.804 |
| FP8 start 12 / 35 | 0.940 | 0.831 |
| FP8 start 18 / 35 | 0.949 | 0.840 |

三个观察：

1. **i2v 明显比 t2v 更受伤**——参考图条件化的任务在固定参照下积累误差，量化噪声无处遁形。
2. `fp8_start_step` 单调回收质量，且**前几步的权重最大**（start 5 已拿回近半差距），印证"早期步最敏感"。速度代价线性且低：189 帧下 FP8 每步省 ~0.55 s，start 12/35 仍保留 ~2/3 的加速。
3. 曲线在变平（i2v：5→12 收 +0.027，12→18 只收 +0.009）——调度只能回收"误差在轨迹里的复利"，后期步残余的量化噪声是地板。要再往上走需要更细粒度的量化尺度（per-block），但 FA4 kernel API 目前只收 per-(batch, kv-head) 的 descale，这条路暂时封死。

给出的实践建议：**t2v/长视频用 FP8 + `fp8_start_step≈N/3`，速度与质量各取一半；i2v 等参考条件化任务先眼验再上；对质量零妥协的场景直接 FA4 bf16——反正它也不比 cuDNN 慢了。**

## 复现

sm_100/sm_103 + CUDA 13 torch，`pip install --pre flash-attn-4`，空闲 GPU（邻居进程曾把我们的早期数据抬高 84%）。

```bash
# bf16：只差一个环境变量
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN_4 \
python examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Nano --num-frames 189 --num-inference-steps 12 --seed 42 ...
```

```python
# FP8 + 步进调度：走 attention 配置
omni = Omni(
    model="nvidia/Cosmos3-Nano",
    diffusion_attention_config={"default": {
        "backend": "FLASH_ATTN_4",
        "extra": {"fp8": True, "fp8_start_step": 12},
    }},
)
```

度量口径：稳态 s/step 取进度条未平滑耗时戳之差，或按 attention kernel 计数切分 profiler trace——**不要**用 tqdm 显示的 `s/it`。完整命令、kernel microbenchmark 脚本与全部原始表格见 [PR #4858](https://github.com/vllm-project/vllm-omni/pull/4858)。
