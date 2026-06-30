---
slug: "/blog/full-duplex-session"
date: "2026-06-30"
title: "From Requests to Sessions: Full-Duplex Inference in vLLM-Omni"
description: "Why a request-oriented engine can't serve Doubao / GPT-4o-style realtime voice, and how vLLM-Omni introduces a session primitive that converges three full-duplex models — MiniCPM-o 4.5, JoyVL, and PersonaPlex — onto one piece of infrastructure."
---

Full-duplex voice is the "Doubao / Gemini Live / GPT-4o voice" experience: no push-to-talk, no turn button. You can interrupt the model mid-sentence, it listens and speaks at the same time, a single conversation context stays alive for minutes, and barge-in lands under 300–400 ms. A new class of speech models — MiniCPM-o 4.5, Nemotron VoiceChat, SoulX-Duplug, Moshi, PersonaPlex — are **full-duplex by design**: they perceive and respond *concurrently*.

The catch is that today's inference engines can't serve them — and not because some model adapter was written wrong. There is a **structural impedance mismatch**: the engine is request-oriented (`prompt → output → done`, KV freed when the request finishes), while the models are stream-oriented (continuous input *while* generating output, KV that must live for the whole conversation). Three independent model PRs — MiniCPM-o 4.5, `nemotron_duplex_h`, and SoulX-Duplug — each hit the **same wall**.

This post walks up the vLLM-Omni stack to make that concrete. First the **disaggregated stage runtime** it already has (the [arXiv:2602.02204](https://arxiv.org/abs/2602.02204) systems paper plus the official meetup framing). Then the **session primitive** that [RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745) proposes to close the gap. Finally three full-duplex models — MiniCPM-o 4.5 ([PR #3907](https://github.com/vllm-project/vllm-omni/pull/3907)), JoyVL ([PR #4623](https://github.com/vllm-project/vllm-omni/pull/4623)), and PersonaPlex ([PR #4771](https://github.com/vllm-project/vllm-omni/pull/4771)) — that land at three different points on this infrastructure.

## Inside the architecture: a disaggregated stage runtime

To understand the mismatch, you first have to understand how vLLM-Omni serves a multimodal request *today*. Its core idea is to decompose an any-to-any model into a **directed graph of stages**: each stage is an independently served engine with its own scheduler and batching, and stages are wired together through a unified connector.

<figure>
<img src="/images/vllm-omni/x3.png" alt="vLLM-Omni architecture">
<figcaption>Figure 1. vLLM-Omni architecture (paper Fig. 3). The Orchestrator holds the Stage Graph / Stage Manager / Request Queues / Data Store; each stage is an independent Exec Engine whose Model Runner loops <code>Schedule() → PreProcFn(req) → Forward(batch)</code> over its own <strong>Scheduler and KV Manager</strong>. Keep those two boxes in mind — full-duplex is exactly what touches them.</figcaption>
</figure>

For Qwen3-Omni this graph instantiates as three stages, `Thinker(LLM) → Talker(LLM) → Vocoder(DiT)`, with two transfer functions (`Thinker2Talker`, `Talker2Vocoder`) carrying the intermediate payloads (hidden states, codec tokens) between them:

<figure>
<img src="/images/vllm-omni/x4.png" alt="Qwen-Omni stage graph">
<figcaption>Figure 2. The Qwen2.5-Omni stage graph and each stage's workflow (paper Fig. 4). Every stage's forward is <code>[Batched]</code> — that is where vLLM-Omni's throughput comes from, and the batching assumption we'll return to repeatedly.</figcaption>
</figure>

Two substrate capabilities matter most for full-duplex:

**1) Streaming stage output (async chunk).** Rather than waiting for one stage to finish before handing off, vLLM-Omni **streams partial output to the next stage incrementally as it is produced**. In Qwen3-Omni, the moment the Talker emits a token, Code2Wav turns it into a waveform — text streaming and audio streaming happen at once:

<figure>
<img src="/images/vllm-omni/async-chunk-stream.png" alt="async chunk streaming">
<figcaption>Figure 3. Async-chunk streaming output (official meetup deck). The Thinker's prefill/decode chain feeds the Talker, whose decode chain feeds Code2Wav; <code>text_i</code> and <code>audio_i</code> stream out as they are computed. Reported gains: 12.4× first-token latency at concurrency 1, 8.2× throughput at concurrency 10.</figcaption>
</figure>

Under the hood this is `OmniConnector` (moves data) + `OmniChunkTransferAdapter` (owns the chunk lifecycle) + schedulers modified for chunk-level scheduling with async IO/compute overlap + background `save_loop`/`recv_loop` threads. This machinery is what we'll call `async_chunk`.

**2) A control-plane / data-plane-decoupled connector.** Metadata travels as lightweight control signals; heavy payloads are offloaded to a high-performance data plane — natively Shared Memory (SHM) or Mooncake:

<figure>
<img src="/images/vllm-omni/x5.png" alt="unified connector">
<figcaption>Figure 4. Disaggregated data transfer with the unified connector (paper Fig. 5). The Orchestrator sends only Request Meta; heavy Payloads go through the connector via D2H/D2D/H2D into the MoonCake Store / Shared Mem. This figure reappears later — full-duplex's "SHM direct-ingress ring" reuses exactly this.</figcaption>
</figure>

This substrate is strong: the paper reports up to a 91.4% reduction in job-completion time versus the baseline. **But it rests on one assumption — it is request-oriented.** A request goes `prefill → decode → finish`, and at finish its KV blocks are returned to the manager. Perfect for one-shot Q&A; for "listen and speak concurrently, conversation alive for minutes," it is precisely the wall.

## The impedance mismatch

RFC #3745 breaks that wall down into six concrete blockers you can point at in the code:

<div class="table-caption">Table 1. Six places the request-oriented runtime blocks full-duplex (per RFC #3745).</div>
<table>
<thead><tr><th>#</th><th>Where</th><th>Current behavior</th><th>Why it blocks duplex</th></tr></thead>
<tbody>
<tr><td>1</td><td><code>OmniARScheduler._free_request / _free_blocks</code></td><td>KV blocks returned to the manager on request finish</td><td>No conversation-lifetime KV — every turn re-prefills</td></tr>
<tr><td>2</td><td><code>_replace_session_with_streaming_update</code></td><td>On a streaming update, sets <code>num_computed_tokens = 0</code> and clears token buffers</td><td>The existing "streaming" path re-prefills each segment — KV is not preserved across turns</td></tr>
<tr><td>3</td><td><code>process_pending_full_payload_inputs</code></td><td>Requests without a ready next chunk are removed from the waiting queue, then re-inserted</td><td>A momentarily-late chunk drops out of the GPU batch → under-batching (~½ throughput, 2× ITL)</td></tr>
<tr><td>4</td><td><code>Orchestrator._route_output</code></td><td>Assumes requests have a begin/end; finalizes on finish</td><td>No notion of a persistent session owning long-lived per-stage requests</td></tr>
<tr><td>5</td><td>Per-token <code>core → client → core</code> over ZMQ</td><td>The next audio chunk round-trips through the orchestrator client</td><td>+3–5 ms/token, growing under concurrency</td></tr>
<tr><td>6</td><td><code>StageExecutionType</code> enum</td><td>Only <code>LLM_AR / LLM_GENERATION / DIFFUSION</code></td><td>No stage type owns turn-taking</td></tr>
</tbody>
</table>

The RFC's thesis in one line: **the unit of work should be the session, not the request.** The Thinker's KV cache should be **leased to the session** and survive every turn; a barge-in flushes downstream stages and bumps an epoch but **never frees the conversation KV**.

## The session primitive (RFC #3745)

The RFC collapses the gap into one new primitive, `DuplexSession`, plus a ring of changes around it. This is the heart of the design.

**`DuplexSession` — the new unit of work.** Owned by the Orchestrator, keyed by `session_id`, holding:
- the set of long-lived per-stage request IDs (one resumable `Request` per stage);
- the **KV lease** handle (blocks pinned for the conversation);
- an input **ring buffer** of incoming chunks (audio / observation — an opaque, modality-agnostic payload, so World Models [#1987] reuse it);
- the **turn state machine** (driven by the `duplex_vad` stage);
- a monotonically increasing **barge-in epoch**;
- TTL-based GC (idle sessions reclaimed, lease released).

**`StageDuplexClient` — a model-agnostic stage interface.** A model conforms by implementing a thin adapter (chunk-in hook, turn-state hook, KV-append hook):

```python
class StageDuplexClient(StagePoolClient, Protocol):
    def open_session(self, sid, params): ...
    def push_chunk(self, sid, chunk): ...                 # SHM ring, not ZMQ
    def signal_turn(self, sid, event): ...
    def barge_in(self, sid, epoch, scope): ...            # scope: "current" | "all"
    def close_session(self, sid): ...
```

**`duplex_vad` stage — owns turn-taking.** A new `StageExecutionType.DUPLEX_VAD` maps to a new `OmniDuplexScheduler`. It runs the streaming VAD / dialogue-state model (the SoulX-Duplug pattern: 160 ms block-causal chunks → `user_idle / user_nonidle / user_backchannel / user_complete / user_incomplete`) and is the single owner of turn boundaries and barge-in detection, gating the downstream thinker/talker.

**Conversation-lifetime KV cache (the deepest change).** Two parts:
- *Session KV lease.* `OmniDuplexScheduler` overrides `_free_request`/`_free_blocks`: a session-bound request's blocks are **not** returned on segment finish — they are released only on `close_session` / TTL-GC.
- *Duplex-append mode.* Replace the `num_computed_tokens = 0` reset in `_replace_session_with_streaming_update` with an **append** path: keep `num_computed_tokens`, extend the block table with the new chunk's tokens, and invalidate from a checkpoint only on rollback (ASR correction).

**Scheduler cadence fix.** Replace "drop a late chunk from the queue" with a **bounded adaptive coalescing window**: a session declares its chunk period (e.g. 160 ms / 80 ms), and a step may wait up to `duplex_batch_window_ms` for in-flight chunks to land so they batch together. Short/zero wait under high concurrency (throughput), small wait under low concurrency (latency).

**IPC round-trip elimination.** Reuse the connector from Figure 4 to build a **session-keyed SHM direct-ingress ring**: the client writes audio chunks straight into the stage-0 process's ring, and only control events (turn signals, barge-in) traverse ZMQ.

**Epoch barge-in semantics.** Every output chunk carries `(session_id, turn_id, epoch)`. A barge-in (VAD-detected user speech during model speech, or a client `input.cancel`) bumps the epoch; all stages drop in-flight work tagged with a stale epoch — **but never the session KV lease** — and resume listening.

The whole thing lands across six phases (Phase 0 brings each model up in request/response mode, Phase 1 adds the session primitive + KV lease, Phase 5 generalizes to World Models #1987). Crucially, **full-duplex does not gate the model PRs from merging first** — a deliberate engineering choice that lets the three models below each come up independently.

## Landing #1: native full-duplex MiniCPM-o 4.5 (PR #3907)

PR #3907 is the first **native** landing of the RFC direction, extending MiniCPM-o 4.5 from ordinary staged serving into a session-oriented audio stream:

```text
client websocket
  -> /v1/duplex  or  /v1/realtime?duplex=1
  -> duplex session actor / event adapter
  -> AsyncOmniEngine duplex data plane
  -> Stage0 MiniCPM-o listen/speak decode
  -> Stage0 -> Stage1 handoff
  -> Stage1 TTS / token2wav
  -> realtime audio delta / done events
```

What it **actually builds**: session-scoped duplex state (session id, response id, **epoch**, **playback cursor**, active-response, and close/cancel lifecycle all tracked explicitly); a real audio-in → Stage0 → Stage1 → audio-out loop; pcm16 → MiniCPM-native pcm_f32le conversion; an overlap policy (a short input doesn't wrongly cancel the current response); stale-epoch filtering on barge-in; and playback-ack → committing played content into conversation memory. `/v1/realtime?duplex=1` also maps OpenAI Realtime-style events. Key H20 E2E signals: `stale_audio_delta_count=0`, `playback_commit_ok=true`, `overlap_listen=true`.

What it **honestly defers** matters just as much: the **persistent core KV lease is not implemented** — `resumable`/session state is *not* the full scheduler-owned lease (with allocation, rollback, migration, release); nor is one-long-lived-request-per-stage, byte-perfect Realtime, or production-grade multi-session admission control. In other words, #3907 validates the **control semantics** (epoch / overlap / playback / handoff) and leaves the RFC's deepest piece — the KV lease — for follow-ups.

## Landings #2 and #3: JoyVL and PersonaPlex — same idea, three positions

The interesting part is that full-duplex isn't a single "go native in the engine" path. Put the three PRs side by side and they fall on a **spectrum** — from "orchestrate on top of a plain serve" to "drive natively inside the engine":

**JoyVL (PR #4623, out-of-process orchestration end).** JoyVL's interaction magic is an **orchestration layer**, not weights: per-second decisions to speak/stay-silent/delegate, a 3-tier memory, and background-agent delegation. It lands a model-agnostic `core/` (`DuplexRuntime` / `DuplexSession` / `DuplexAdapter`, epoch barge-in + streaming) plus a `joyvl/` implementation under `vllm_omni/experimental/fullduplex/`. The day-0 shape is a **standalone orchestrator process** talking to a plain `vllm serve` over HTTP, with pluggable external ASR/TTS bridges. It does **not** go in-engine — it leans on vLLM's automatic radix prefix cache for KV reuse. (Today `core/` is a tested demonstration; the live path drives decision + memory directly.)

**PersonaPlex (PR #4771, native-engine end).** A Moshi-based full-duplex S2S model (`nvidia/personaplex-7b-v1`), natively ported to vLLM: Helium temporal transformer + depformer (RQ code predictor) + Code2Wav (Mimi), each parity-checked (temporal cos ≈ 0.999). Serving is a `FrameStepper`/`PersonaPlexSession` **lockstep** driver — one 80 ms user frame in, one agent frame + text out — at RTF ≈ 0.33 (26 ms vs the 80 ms budget) after CUDA-graphing. It serves the official PersonaPlex web client directly. Currently one conversation per server instance (a single lockstep KV state), greedy by default.

All three in one table:

<div class="table-caption">Table 2. Where three full-duplex models land on vLLM-Omni.</div>
<table>
<thead><tr><th>Dimension</th><th>MiniCPM-o 4.5 (#3907)</th><th>JoyVL (#4623)</th><th>PersonaPlex (#4771)</th></tr></thead>
<tbody>
<tr><td>Modality</td><td>audio ↔ audio</td><td>video+text → text (speech bolted on)</td><td>audio ↔ audio (S2S)</td></tr>
<tr><td>Driver position</td><td>in-engine native duplex data plane</td><td>out-of-process orchestrator (HTTP)</td><td>in-engine native lockstep</td></tr>
<tr><td>Session / KV</td><td>explicit duplex state; KV lease <strong>deferred</strong></td><td>radix prefix cache (out-of-process can't reach session KV)</td><td>single lockstep KV, one conversation/instance</td></tr>
<tr><td>Barge-in</td><td>epoch + stale-epoch filter</td><td><code>core/</code> epoch / <code>is_stale</code></td><td>per-frame lockstep</td></tr>
<tr><td>Maturity</td><td>WIP, control semantics verified</td><td>merged (day-0 orchestration)</td><td>open (greedy, single session)</td></tr>
</tbody>
</table>

This is the design working as intended: the RFC defines session management once, and each model **implements a subset** that fits its reality — JoyVL grabs day-0 with out-of-process orchestration, MiniCPM/PersonaPlex go native in-engine. The shared landing zone is `experimental/fullduplex/`, which is why PersonaPlex #4771 is extending the very `core/` JoyVL first built — it is the second consumer of that abstraction.

## Three threads running through all of it

Stack the RFC and the three landings together and three concepts run **from the abstract design all the way through every implementation** — the fastest three keys to understanding full-duplex:

**1) Epoch barge-in.** A monotonically increasing session epoch: output carries the epoch, a barge-in bumps it, stale work is dropped, KV is untouched. RFC #3745 defines it → #3907 implements it (stale-epoch filtering, measured `stale_audio_delta_count=0`) → JoyVL's `core/session.py` has the `epoch` field + `is_stale()` → PersonaPlex's per-frame lockstep. **Everyone does this one** — it's the minimal mechanism for "you can interrupt."

**2) Playback cursor.** The client reports "I've actually played up to sample N" — distinguishing *sent* from *heard*. If the user interrupts at 3 seconds, the model's memory of "what I said" should record only those 3 seconds, not the 10 seconds already streamed out. The RFC puts it in session state → #3907 **actually uses it** (playback-ack → commit to history) → JoyVL's `core/session.py` has a `playback_cursor` field but it's **scaffolding** (the hook is a no-op; a text model doesn't need it). **Only native audio models fill this one in for real.**

**3) KV lease (conversation-lifetime KV).** Blocks leased to the session, alive across every turn, released only on close. This is the **heart** of the RFC and also the **hardest, least-finished** thread: #3907 explicitly defers it, JoyVL can only approximate it out-of-process via the radix prefix cache, and PersonaPlex sidesteps it with "one instance, one conversation" single-lockstep KV. **No implementation has built the full lease the RFC describes** (with allocation, rollback, migration, release). This is the bone full-duplex has to chew to go from "demoable" to "productionizable."

## What's left

On the official roadmap, full-duplex is **P1** — "Auto-regressive DiT models (interactive / world models)" and "Video streaming input / output," tracked in [issue #2136](https://github.com/vllm-project/vllm-omni/issues/2136). RFC #3745's Phase 5 is explicit that `DuplexChunk` becomes an opaque typed payload so a World-Model (#1987) observe→predict loop reuses the same `DuplexSession` — **one session-management system, not two**.

As it stands, three pieces remain between here and the RFC's end state, in increasing difficulty:

1. **API-server unification (light).** Fold the out-of-process orchestration / duplex serving into a single `vllm serve` process — one port, one process.
2. **ASR/TTS onto `async_chunk` stages (medium).** TTS first (it already runs as a staged async_chunk model); streaming ASR is the bigger lift.
3. **The full KV lease (heavy).** The RFC itself calls this "the deepest change, with its own sub-design + tests" — allocation, rollback (ASR correction), migration, release, plus the coalescing window and the SHM direct-ingress ring.

Drive those three through the RFC's phases and the request-oriented engine grows a real session skeleton — and the three full-duplex implementations that look independent today (MiniCPM-o 4.5, JoyVL, PersonaPlex) converge onto one piece of infrastructure. That's the whole point: full-duplex isn't "adapt one more model," it's swapping the unit of work from a request to a session.

---

*Sources: the vLLM-Omni systems paper ([arXiv:2602.02204](https://arxiv.org/abs/2602.02204); Figures 1/2/4 are its Fig. 3/4/5), the official Apr-2026 meetup deck (Figure 3, async-chunk), RFC [#3745](https://github.com/vllm-project/vllm-omni/issues/3745), and PRs [#3907](https://github.com/vllm-project/vllm-omni/pull/3907) / [#4623](https://github.com/vllm-project/vllm-omni/pull/4623) / [#4771](https://github.com/vllm-project/vllm-omni/pull/4771). Project: [github.com/vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni).*
