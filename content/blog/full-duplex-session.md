---
slug: "/blog/full-duplex-session"
date: "2026-06-30"
title: "From Requests to Sessions: Full-Duplex Inference in vLLM-Omni"
description: "Why a request-oriented engine can't serve realtime voice, what the full-duplex session architecture (RFC #3745) changes about vLLM-Omni's runtime, and how the MiniCPM-o 4.5 native duplex PR (#3907) realizes it."
---

Full-duplex voice is the "Doubao / Gemini Live / GPT-4o voice" experience: no push-to-talk, no turn button. You can interrupt the model mid-sentence, it listens and speaks at the same time, a single conversation context stays alive for minutes, and barge-in lands under 300–400 ms. A new class of speech models — MiniCPM-o 4.5, Nemotron VoiceChat, SoulX-Duplug, Moshi — are **full-duplex by design**: they perceive and respond *concurrently*.

Today's inference engines can't serve them, and not because some adapter was written wrong. There is a **structural impedance mismatch**: the engine is request-oriented (`prompt → output → done`, KV freed when the request finishes), while the models are stream-oriented (continuous input *while* generating output, KV that must live for the whole conversation). Three independent model PRs hit the **same wall**.

This post is about the design that fixes it. We walk up the vLLM-Omni stack: the **disaggregated stage runtime it already has** (from the [arXiv:2602.02204](https://arxiv.org/abs/2602.02204) systems paper and the official meetup deck), the **full-duplex session architecture** that [RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745) layers on top of it, and the **MiniCPM-o 4.5 native duplex PR** ([#3907](https://github.com/vllm-project/vllm-omni/pull/3907)) that turns the design into running code. The paper and the current runtime are here to make the design legible — not as the subject, but as the ground it stands on.

## Inside vLLM-Omni today: a disaggregated stage runtime

To see the mismatch you first need to see how vLLM-Omni serves a multimodal request *today*. Its core idea is to decompose an any-to-any model into a **directed graph of stages**: each stage is an independently served engine with its own scheduler and batching, wired to the next through a unified connector.

<figure>
<img src="/images/vllm-omni/x3.png" alt="vLLM-Omni architecture">
<figcaption>Figure 1. vLLM-Omni architecture (paper Fig. 3). The Orchestrator holds the Stage Graph / Stage Manager / Request Queues / Data Store; each stage is an independent Exec Engine whose Model Runner loops <code>Schedule() → PreProcFn(req) → Forward(batch)</code> over its own <strong>Scheduler and KV Manager</strong>. Keep those two boxes in mind — full-duplex is exactly what touches them.</figcaption>
</figure>

For Qwen3-Omni this graph instantiates as `Thinker(LLM) → Talker(LLM) → Vocoder(DiT)`, with two transfer functions carrying intermediate payloads (hidden states, codec tokens) between stages. Every stage's forward is `[Batched]` — that is where the throughput comes from, and the batching assumption the duplex design has to protect.

<figure>
<img src="/images/vllm-omni/x4.png" alt="Qwen-Omni stage graph">
<figcaption>Figure 2. The Qwen2.5-Omni stage graph and each stage's workflow (paper Fig. 4).</figcaption>
</figure>

Two substrate capabilities matter most for what follows.

**Streaming stage output (async chunk).** Rather than waiting for one stage to finish before handing off, vLLM-Omni **streams partial output to the next stage incrementally as it is produced** — the moment the Talker emits a token, Code2Wav turns it into a waveform. Text and audio stream at once (`OmniConnector` + `OmniChunkTransferAdapter` + chunk-aware schedulers + background `save_loop`/`recv_loop`). This machinery is what we'll call `async_chunk`.

<figure>
<img src="/images/vllm-omni/async-chunk-stream.png" alt="async chunk streaming">
<figcaption>Figure 3. Async-chunk streaming output (official meetup deck): the Thinker→Talker→Code2Wav chain emits <code>text_i</code> and <code>audio_i</code> as they are computed (reported: 12.4× first-token latency at concurrency 1, 8.2× throughput at concurrency 10).</figcaption>
</figure>

**A control/data-plane-decoupled connector.** Metadata travels as lightweight control signals; heavy payloads are offloaded to a high-performance data plane — natively SHM or Mooncake. The Orchestrator sends only request meta; payloads move D2H/D2D/H2D through the connector. (Full-duplex's "SHM direct-ingress ring," below, reuses exactly this.)

This substrate is strong — the paper reports up to a 91.4% reduction in job-completion time. **But it rests on one assumption: it is request-oriented.** A request goes `prefill → decode → finish`, and at finish its KV blocks return to the manager. Perfect for one-shot Q&A; for "listen and speak concurrently, context alive for minutes," it is the wall.

## The impedance mismatch

RFC #3745 breaks that wall into six concrete blockers you can point at in the code:

<div class="table-caption">Table 1. Six places the request-oriented runtime blocks full-duplex (per RFC #3745).</div>
<table>
<thead><tr><th>#</th><th>Where</th><th>Current behavior</th><th>Why it blocks duplex</th></tr></thead>
<tbody>
<tr><td>1</td><td><code>OmniARScheduler._free_request / _free_blocks</code></td><td>KV blocks returned to the manager on request finish</td><td>No conversation-lifetime KV — every turn re-prefills</td></tr>
<tr><td>2</td><td><code>_replace_session_with_streaming_update</code></td><td>On a streaming update, sets <code>num_computed_tokens = 0</code> and clears token buffers</td><td>The existing "streaming" path re-prefills each segment — KV not preserved across turns</td></tr>
<tr><td>3</td><td><code>process_pending_full_payload_inputs</code></td><td>Requests without a ready next chunk are removed from the waiting queue, then re-inserted</td><td>A momentarily-late chunk drops out of the GPU batch → under-batching (~½ throughput, 2× ITL)</td></tr>
<tr><td>4</td><td><code>Orchestrator._route_output</code></td><td>Assumes requests have a begin/end; finalizes on finish</td><td>No notion of a persistent session owning long-lived per-stage requests</td></tr>
<tr><td>5</td><td>Per-token <code>core → client → core</code> over ZMQ</td><td>The next audio chunk round-trips through the orchestrator client</td><td>+3–5 ms/token, growing under concurrency</td></tr>
<tr><td>6</td><td><code>StageExecutionType</code> enum</td><td>Only <code>LLM_AR / LLM_GENERATION / DIFFUSION</code></td><td>No stage type owns turn-taking</td></tr>
</tbody>
</table>

The thesis in one line: **the unit of work should be the session, not the request.** The Thinker's KV cache should be **leased to the session** and survive every turn; a barge-in flushes downstream stages and bumps an epoch but **never frees the conversation KV**.

## The full-duplex session architecture (RFC #3745)

The fix is one new primitive — `DuplexSession` — plus a ring of changes around it. The clearest way in is the target experience: what should happen, tick by tick, when a user talks over the model.

```text
 user mic ──PCM 80–160ms──▶ WS /v1/audio/conversation ──▶ DuplexSession (orchestrator)
                                                              │  KV lease acquired
   ┌──────────────────────── loop: never "done" ─────────────┤
   │  chunk in ─▶ duplex_vad stage ─▶ turn state                │
   │                       │                                    │
   │           user_complete │                  user speaks while model speaking
   │                       ▼                                    ▼
   │   gate OPEN ─▶ Thinker append (KV KEPT) ─▶        barge-in: epoch++  ───────────┐
   │   text delta ─▶ Talker ──async chunk──▶ Code2Wav        │  flush stale-epoch     │
   │   ▲                                      │ audio out      │  output, abort talker  │
   │   └──────────── playback ack ◀───────────┘                │  KEEP KV, resume listen│
   └───────────────────────────────────────────────────────────┴────────────────────┘
                                              session.close ─▶ KV lease released
```

The session is the unit, not the request. The Thinker's KV is **leased to the session** and survives every turn; barge-in bumps an epoch and flushes downstream stages but never frees that KV. Concretely the RFC adds:

**`DuplexSession` — the new unit of work.** Owned by the Orchestrator, keyed by `session_id`, holding: the long-lived per-stage request IDs (one resumable `Request` per stage); the **KV lease** handle (blocks pinned for the conversation); an input **ring buffer** of chunks (an opaque, modality-agnostic payload, so World Models [#1987] reuse it); the **turn state machine**; a monotonically increasing **barge-in epoch**; and TTL-based GC.

**`StageDuplexClient` — a model-agnostic stage interface.** A model conforms by implementing a thin adapter (chunk-in, turn-state, KV-append hooks):

```python
class StageDuplexClient(StagePoolClient, Protocol):
    def open_session(self, sid, params): ...
    def push_chunk(self, sid, chunk): ...                 # SHM ring, not ZMQ
    def signal_turn(self, sid, event): ...
    def barge_in(self, sid, epoch, scope): ...            # scope: "current" | "all"
    def close_session(self, sid): ...
```

**`duplex_vad` stage — owns turn-taking.** A new `StageExecutionType.DUPLEX_VAD` maps to a new `OmniDuplexScheduler`. It runs the streaming VAD / dialogue-state model (the SoulX-Duplug pattern: 160 ms block-causal chunks → `user_idle / user_nonidle / user_backchannel / user_complete / user_incomplete`) and is the single owner of turn boundaries and barge-in detection, gating the downstream thinker/talker.

**Conversation-lifetime KV cache — the deepest change.** Two parts:
- *Session KV lease.* `OmniDuplexScheduler` overrides `_free_request`/`_free_blocks`: a session-bound request's blocks are **not** returned on segment finish — only on `close_session` / TTL-GC.
- *Duplex-append mode.* Replace the `num_computed_tokens = 0` reset in `_replace_session_with_streaming_update` with an **append** path: keep `num_computed_tokens`, extend the block table with the new chunk's tokens, invalidate from a checkpoint only on rollback (ASR correction).

**Scheduler cadence fix.** Replace "drop a late chunk from the queue" with a **bounded adaptive coalescing window**: a session declares its chunk period (e.g. 160 ms / 80 ms), and a step may wait up to `duplex_batch_window_ms` for in-flight chunks to land so they batch — short/zero wait under high concurrency, small wait under low.

**IPC round-trip elimination.** A **session-keyed SHM direct-ingress ring** (reusing the connector): the client writes audio chunks straight into the stage-0 process's ring; only control events traverse ZMQ.

**Epoch barge-in.** Every output chunk carries `(session_id, turn_id, epoch)`. A barge-in bumps the epoch; all stages drop in-flight work tagged with a stale epoch — **but never the KV lease** — and resume listening.

The whole thing lands across six phases; importantly, **full-duplex does not gate the model PRs from merging first** — each model comes up in request/response mode, then opts into the session machinery.

## Realizing it: MiniCPM-o 4.5 native duplex (PR #3907)

RFC #3745 is the design; PR #3907 is the first **native** realization. It extends MiniCPM-o 4.5 from ordinary staged serving into a session-oriented audio stream that runs the real data plane, not a smoke test around the chat endpoint:

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

Mapped onto the RFC's pieces, here is what #3907 builds:

<div class="table-caption">Table 2. How MiniCPM-o 4.5 (#3907) realizes the session architecture.</div>
<table>
<thead><tr><th>RFC concept</th><th>In #3907</th></tr></thead>
<tbody>
<tr><td>Session as the unit</td><td>Explicit duplex state — session id, response id, <strong>epoch</strong>, <strong>playback cursor</strong>, active-response, close/cancel lifecycle — instead of one-off request state</td></tr>
<tr><td>Independent in/out flow</td><td>WS input append no longer a blocking request→response turn; cancel/barge-in observed <em>while</em> output is active</td></tr>
<tr><td>Duplex data plane</td><td>Audio append + Stage0→Stage1 handoff travel the engine/orchestrator/scheduler/worker path, not a fake chat-completion control surface</td></tr>
<tr><td>Turn-taking + gating</td><td>Stage0 runs MiniCPM's audio-streaming path and listen/speak policy; Stage1 consumes the handoff and emits TTS</td></tr>
<tr><td>Epoch barge-in</td><td>In-flight cancel filters stale-epoch output; an overlap policy means a short input doesn't wrongly cancel the current response</td></tr>
<tr><td>Playback-aware memory</td><td>Playback ack commits <em>played</em> assistant content to history, not every emitted byte</td></tr>
<tr><td>Protocol layer</td><td><code>/v1/realtime?duplex=1</code> maps OpenAI Realtime-style events (<code>session.update</code>, <code>input_audio_buffer.append/commit</code>, <code>response.create/cancel</code>, <code>response.audio.delta/done</code>)</td></tr>
</tbody>
</table>

What it **honestly defers** matters just as much: the **persistent core KV lease is not implemented** — `resumable`/session state is *not* the full scheduler-owned lease (allocation, rollback, migration, release); nor is one-long-lived-request-per-stage, byte-perfect Realtime, or production multi-session admission. In RFC terms, #3907 nails the **control semantics** (Phase 0–3 territory: session state, epoch, overlap, playback, handoff) and leaves the **conversation-lifetime KV lease** (the Phase 1 deep change) for a follow-up.

The H20 end-to-end run is the receipt: `model_listen_policy_observed=true`, `overlap_listen=true`, `overlap_barge_in=true`, `short_ack_cancelled=false`, `playback_commit_ok=true`, and crucially `stale_audio_delta_count=0` — barge-in actually drops the stale stream. That is the listen-while-speak loop working on real audio.

## Three threads, from design to running code

The fastest way to *understand* the full-duplex design is to follow three concepts from the RFC straight into #3907 — they are the load-bearing ideas:

**1) Epoch barge-in.** A monotonically increasing session epoch: output carries it, a barge-in bumps it, stale work is dropped, KV is untouched. RFC defines it → #3907 implements it, with `stale_audio_delta_count=0` proving the stale stream is actually discarded. This is the minimal mechanism for "you can interrupt."

**2) Playback cursor.** The client reports "I've actually played up to sample N" — distinguishing *sent* from *heard*. If the user cuts in at 3 seconds, the model's memory of "what I said" should record only those 3 seconds, not the 10 seconds already streamed. RFC puts it in session state → #3907 actually uses it (playback ack → commit to history). Without it, the model believes it said things the user never heard.

**3) KV lease (conversation-lifetime KV).** Blocks leased to the session, alive across every turn, released only on close. This is the **heart** of the RFC and the **hardest, least-finished** thread — #3907 explicitly defers it, falling back to resumable session state. It is the bone full-duplex has to chew to go from "demoable" (#3907 today) to "no re-prefill between turns, KV held for the whole conversation" (the RFC's measurable Phase-1 goal).

## The same primitive, other positions

#3907 sits at the native-in-engine end of a spectrum, and it isn't alone. **JoyVL** ([#4623](https://github.com/vllm-project/vllm-omni/pull/4623), merged) sits at the opposite end: an out-of-process orchestrator over a plain `vllm serve`, day-0, leaning on vLLM's automatic radix prefix cache for KV reuse — it ships the same `experimental/fullduplex/core/` (`DuplexRuntime`/`DuplexSession`/`DuplexAdapter`, epoch barge-in) that the native models can later build on. **PersonaPlex** ([#4771](https://github.com/vllm-project/vllm-omni/pull/4771), a Moshi-based S2S model) sits near #3907: a native lockstep `FrameStepper` (80 ms frame in → frame + text out, RTF ≈ 0.33), and the second consumer of that shared `core/`. The point of the RFC is that all three implement *subsets* of one session-management system rather than three ad-hoc streaming paths — which is exactly why they share `experimental/fullduplex/`.

## What's left

On the official roadmap, full-duplex is **P1** — "Auto-regressive DiT models (interactive / world models)" and "Video streaming input / output," tracked in [issue #2136](https://github.com/vllm-project/vllm-omni/issues/2136). RFC #3745's Phase 5 makes `DuplexChunk` an opaque typed payload so a World-Model (#1987) observe→predict loop reuses the same `DuplexSession` — one session-management system, not two.

Three pieces remain between #3907 and the RFC's end state, in increasing difficulty:

1. **API-server unification (light).** One `vllm serve` process hosting the duplex serving path.
2. **ASR/TTS onto `async_chunk` stages (medium).** TTS first (already a staged async_chunk model); streaming ASR is the bigger lift.
3. **The full KV lease (heavy).** The RFC itself calls this "the deepest change, with its own sub-design + tests" — allocation, rollback, migration, release, plus the coalescing window and the SHM direct-ingress ring.

Drive those through the RFC's phases and the request-oriented engine grows a real session skeleton. That's the whole point: full-duplex isn't "adapt one more model," it's swapping the unit of work from a request to a session — and MiniCPM-o 4.5 is the first model to run on it.

---

*Sources: the vLLM-Omni systems paper ([arXiv:2602.02204](https://arxiv.org/abs/2602.02204); Figures 1–2 are its Fig. 3/4), the official Apr-2026 meetup deck (Figure 3, async-chunk), RFC [#3745](https://github.com/vllm-project/vllm-omni/issues/3745), and the MiniCPM-o 4.5 duplex PR [#3907](https://github.com/vllm-project/vllm-omni/pull/3907) (with [#4623](https://github.com/vllm-project/vllm-omni/pull/4623) / [#4771](https://github.com/vllm-project/vllm-omni/pull/4771) for context). Project: [github.com/vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni).*
