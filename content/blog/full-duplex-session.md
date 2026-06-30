---
slug: "/blog/full-duplex-session"
date: "2026-06-30"
title: "From Requests to Sessions: The Full-Duplex Architecture in vLLM-Omni"
description: "A walkthrough of RFC #3745 — the full-duplex session primitive for vLLM-Omni — how the community design review reshaped it, and how the MiniCPM-o 4.5 PR (#3907) lands the refined design."
---

Full-duplex voice is the "Doubao / Gemini Live / GPT-4o voice" experience: no push-to-talk, no turn button. You can interrupt the model mid-sentence, it listens and speaks at the same time, a single conversation context stays alive for minutes, and barge-in lands under 300–400 ms. A new class of speech models — MiniCPM-o 4.5 (Omni-Flow / TDM), Nemotron VoiceChat (`nemotron_duplex_h`), SoulX-Duplug, Moshi — are **full-duplex by design**: they perceive and respond *concurrently*.

vLLM-Omni cannot serve this today, and not because an adapter was written wrong. There is a **structural impedance mismatch**: the runtime is request-oriented (`prompt → output → done`, KV freed at finish), while the models are stream-oriented (continuous audio in *while* generating out, conversation-lifetime KV). Three model PRs (MiniCPM-o 4.5, `nemotron_duplex_h`, SoulX-Duplug) independently hit the **same wall**, each at risk of growing its own ad-hoc streaming path.

[RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745) defines the duplex session primitive **once** and enumerates how each model conforms. This post walks through it: the substrate it builds on, the primitive itself, how the design review reshaped it across six model families, and how [PR #3907](https://github.com/vllm-project/vllm-omni/pull/3907) lands it for MiniCPM-o 4.5.

## Inside vLLM-Omni today: a disaggregated stage runtime

To see the mismatch you first have to see how vLLM-Omni serves a multimodal request *today*. It decomposes an any-to-any model into a **directed graph of stages**: each stage is an independently served engine with its own scheduler and batching, wired to the next through a unified connector.

<figure>
<img src="/images/vllm-omni/x3.png" alt="vLLM-Omni architecture">
<figcaption>Figure 1. vLLM-Omni architecture (<a href="https://arxiv.org/abs/2602.02204">arXiv:2602.02204</a>, Fig. 3). Each stage is an Exec Engine whose Model Runner loops <code>Schedule() → PreProcFn(req) → Forward(batch)</code> over its own <strong>Scheduler and KV Manager</strong> — the two boxes full-duplex has to change.</figcaption>
</figure>

For Qwen3-Omni this graph instantiates as `Thinker(LLM) → Talker(LLM) → Vocoder(DiT)`, every forward `[Batched]`. Two substrate capabilities matter for what follows: **streaming stage output (`async_chunk`)** — partial output streams to the next stage incrementally, so the moment the Talker emits a token, Code2Wav turns it into a waveform — and a **control/data-plane-decoupled connector** (SHM or Mooncake) that the duplex design later reuses for its direct-ingress ring.

<figure>
<img src="/images/vllm-omni/async-chunk-stream.png" alt="async chunk streaming">
<figcaption>Figure 2. Async-chunk streaming (official meetup deck): the Thinker→Talker→Code2Wav chain emits <code>text_i</code> and <code>audio_i</code> as they are computed.</figcaption>
</figure>

This substrate is strong — up to 91.4% lower job-completion time versus baseline. **But it is request-oriented.** A request goes `prefill → decode → finish`, and at finish its KV blocks return to the manager. Perfect for one-shot Q&A; for "listen and speak concurrently, context alive for minutes," it is the wall.

## The impedance mismatch

```mermaid
flowchart LR
    subgraph REQ["vLLM-Omni today — request-oriented"]
        A1[add_request] --> A2[prefill prompt] --> A3[decode loop] --> A4[finish]
        A4 --> A5["free_request → free_blocks<br/>KV returned to manager"]
    end
    subgraph DUP["Duplex models — stream-oriented"]
        B1[open session] --> B2[continuous chunks in]
        B2 --> B3[generate out concurrently] --> B2
        B2 --> B4[barge-in / turn change] --> B3
        B3 --> B5["close session<br/>minutes later"]
    end
```

RFC #3745 grounds the wall in six concrete blockers in the current code:

<div class="table-caption">Table 1. Six places the request-oriented runtime blocks full-duplex (RFC #3745).</div>
<table>
<thead><tr><th>#</th><th>Where</th><th>Current behavior</th><th>Why it blocks duplex</th></tr></thead>
<tbody>
<tr><td>1</td><td><code>OmniARScheduler._free_request / _free_blocks</code></td><td>KV blocks returned to the manager on request finish</td><td>No conversation-lifetime KV — every turn re-prefills</td></tr>
<tr><td>2</td><td><code>_replace_session_with_streaming_update</code></td><td>Sets <code>num_computed_tokens = 0</code>, clears token buffers</td><td>The existing "streaming" path re-prefills each segment</td></tr>
<tr><td>3</td><td><code>process_pending_full_payload_inputs</code></td><td>A request without a ready next chunk is pulled from the queue, re-inserted later</td><td>A late chunk drops out of the GPU batch → under-batching (~½ throughput, 2× ITL)</td></tr>
<tr><td>4</td><td><code>Orchestrator._route_output</code></td><td>Assumes requests have a begin/end; finalizes on finish</td><td>No persistent session owning long-lived per-stage requests</td></tr>
<tr><td>5</td><td>Per-token <code>core → client → core</code> over ZMQ</td><td>The next audio chunk round-trips through the orchestrator client</td><td>+3–5 ms/token, growing under concurrency</td></tr>
<tr><td>6</td><td><code>StageExecutionType</code> enum</td><td>Only <code>LLM_AR / LLM_GENERATION / DIFFUSION</code></td><td>No stage type owns turn-taking</td></tr>
</tbody>
</table>

The thesis in one line: **the unit of work is the session, not the request.** The Thinker's KV is leased to the session and survives every turn; barge-in flushes downstream stages and bumps an epoch but never frees the conversation KV.

## The session primitive (RFC #3745)

The target experience is a loop that never says "done":

```mermaid
sequenceDiagram
    participant U as User (mic)
    participant S as DuplexSession (orchestrator)
    participant DV as duplex_vad
    participant TH as Thinker (session-pinned KV)
    participant TK as Talker / Token2Wav
    U->>S: open_session — KV lease acquired
    loop continuous, never "done"
        U-->>S: PCM chunk (80–160 ms)
        S->>DV: chunk → turn state
        alt user_complete
            S->>TH: gate open — append chunk (KV kept)
            TH-->>U: text delta
            TH->>TK: hidden states (async chunk)
            TK-->>U: audio
        else user speaks while model speaking
            U-->>S: PCM (barge-in)
            S->>TK: epoch++ — flush stale, abort talker
            S->>TH: keep KV, resume listening
        end
    end
    U->>S: close — KV lease released
```

The RFC lays this out as four layers. The key pieces:

```mermaid
flowchart TB
    subgraph API["Protocol layer — OpenAI-Realtime-aligned"]
        WS["WS /v1/audio/conversation"]
    end
    subgraph ENG["Engine layer"]
        DS["DuplexSession registry<br/>TTL-GC, epoch, ring buffer"]
        SDC["StageDuplexClient<br/>open / push_chunk / signal_turn / barge_in / close"]
    end
    subgraph SCHED["Scheduler layer"]
        DSCH["OmniDuplexScheduler<br/>session KV lease + coalescing window"]
    end
    subgraph STAGE["Stage layer"]
        DV["DUPLEX_VAD — owns turn-taking"]
        AR["LLM_AR — duplex-append mode"]
    end
    WS --> DS --> SDC --> DSCH --> DV
    DSCH --> AR
    SDC -. "direct SHM chunk ring (bypass ZMQ)" .-> STAGE
```

- **`DuplexSession`** (owned by the Orchestrator, keyed by `session_id`): the long-lived per-stage request IDs (one resumable `Request` per stage), the **KV lease**, an input **ring buffer** of opaque chunks, the **turn state machine**, a monotonic **barge-in epoch**, and TTL-GC. A session-bound request is never finalized by `_route_output` until `close_session`.
- **`StageDuplexClient`**: a model conforms by implementing a thin adapter — `push_chunk` (over the SHM ring, not ZMQ), `signal_turn`, `barge_in(epoch, scope)`, `close_session`.
- **Conversation-lifetime KV** — the deepest change: a *session KV lease* (`_free_blocks` not called on segment finish, only on close / TTL-GC) plus *duplex-append mode* (keep `num_computed_tokens`, extend the block table with the new chunk, invalidate from a checkpoint only on ASR-correction rollback).
- **Adaptive coalescing window** (fixes blocker #3): wait up to `duplex_batch_window_ms` for in-flight chunks to batch — short under high concurrency, small under low.
- **Epoch barge-in**: every output chunk carries `(session_id, turn_id, epoch)`; a barge-in bumps the epoch and stages drop stale-epoch work — but never the KV lease.

## One primitive, many model shapes

The reason this is worth doing *once* is that full-duplex models share almost nothing structurally except needing persistent KV. The design review (esp. @linyueqian walking #3642 and #3512 against the primitive) made that concrete with two shapes that look nothing alike.

**MiniCPM-o 4.5 (Omni-Flow): 1-second chunk groups.** A structured token group is appended once per chunk period; output is variable-length, terminated by a learned `⟨chunk_eos⟩`; the model self-VADs with `⟨listen⟩` / `⟨speak⟩`, so barge-in is only possible at chunk boundaries.

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant LM as Omni LM (session KV)
    participant T2W as Flow-Matching Decoder
    Note over LM: t=0..1s, user mid-sentence
    U->>LM: append [chunk_bos v0 a0]
    LM-->>U: listen / chunk_eos
    Note over LM: t=1..2s, user done, model speaks
    U->>LM: append [chunk_bos v1 a1]
    LM-->>T2W: speak / tts_bos t1 s1 ... tts_eos / chunk_eos
    T2W-->>U: audio frames
    Note over LM: t=2..3s, user barges in
    U->>LM: append [chunk_bos v2 a2]
    LM-->>U: listen / chunk_eos
    Note over T2W,U: in-flight audio plays out <= 1 s
```

**Nemotron VoiceChat: per-decode-step embedding fusion.** No chunks, no boundary tokens. One acoustic embedding (a continuous tensor, not a token) is added at the input embedding every decode tick; one text token + one ASR token come out per tick; EarTTS runs lag-by-one as a separate stage. There is no learned `⟨listen⟩`/`⟨speak⟩` — "silent" is text EOS reused — so barge-in is *not* a model affordance and must be enforced engine-side.

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant ENC as Acoustic Encoder
    participant LM as NemotronDuplexH (session KV)
    participant TTS as EarTTS (separate stage)
    Note over LM: each decode tick (~25..40 ms)
    U->>ENC: PCM frame
    ENC->>LM: acoustic_embedding_t (continuous tensor)
    LM-->>LM: text_token_t (LM head)
    LM-->>LM: asr_token_t (parallel ASR head)
    LM->>TTS: text_token_t
    TTS-->>U: speech_code (lag-by-1)
```

These two architectures, plus the future Moshi-class joint predictor, set hard **latency floors** and force the adapter to support **three injection patterns** rather than one:

<div class="table-caption">Table 2. Barge-in latency floors (per the RFC discussion).</div>
<table>
<thead><tr><th>Barge-in latency</th><th>What it takes</th><th>Where we are</th></tr></thead>
<tbody>
<tr><td>~1 chunk (≈1 s, MiniCPM-o 4.5)</td><td>Session + KV lease + append; model self-VADs at chunk boundaries</td><td>reached natively</td></tr>
<tr><td>~150–300 ms</td><td>External streaming VAD + sub-chunk cancel in <code>OmniARScheduler</code> + <code>audio_end_ms</code> truncate on cancel</td><td>needs new engine work</td></tr>
<tr><td>~80 ms (Moshi-class joint)</td><td>A different model architecture entirely</td><td>lockstep S2S models now arriving</td></tr>
</tbody>
</table>

<div class="table-caption">Table 3. The three injection patterns a duplex adapter must express.</div>
<table>
<thead><tr><th>Pattern</th><th>Example</th><th>Engine unit per cadence tick</th><th>Terminator</th></tr></thead>
<tbody>
<tr><td>Chunk-group append</td><td>MiniCPM-o 4.5</td><td>structured token group</td><td>learned <code>⟨chunk_eos⟩</code></td></tr>
<tr><td>Per-step tensor inject</td><td>Nemotron VoiceChat</td><td>one tensor at the input embedding</td><td>none (continuous)</td></tr>
<tr><td>Parallel-frame joint</td><td>Moshi-class</td><td>joint <code>(audio_in, audio_out)</code> tuple, multi-codebook head</td><td>none (frame-clocked)</td></tr>
</tbody>
</table>

## What the design review converged on

The thread (8+ contributors, six model families) tightened the RFC in a few decisive ways. Synthesized:

- **Turn-taking is not owned by one stage.** `DUPLEX_VAD` should be optional, not mandatory: an end-to-end model that self-VADs fills the role itself, so turn boundaries come from *multiple signal sources* — VAD, model-native control tokens, client events, server policy (Sy0307, tc-mb, and the author's own open question).
- **Append is a per-model capability, not a default.** Different models need append / replace-latest-chunk / full re-encode / rollback. Gate the whole duplex path behind `session_mode: turn | duplex` so the six existing TTS pipelines stay on `turn` and never regress (Sy0307, linyueqian).
- **A resumable-request-per-stage is a real semantic change** — a request becomes a session-bound stage actor needing stage binding, abort/recovery, and fairness, not just "don't finalize on finish" (Sy0307). And conversation-lifetime KV is mostly a *stage-0* problem; downstream stages are epoch-flushable (yinpeiqi).
- **Sent ≠ heard.** Barge-in drops stale audio, but history commit needs a **playback cursor / committed offset** — otherwise the next turn includes assistant speech the user never heard (Sy0307).
- **Single-session first.** Land the primitive shape (registry, KV lease, adapter hooks) on one session before multi-session scheduling (tc-mb, broadly agreed).
- **Pin each phase to a latency it commits to**, and keep `core ↔ client` transports (ZMQ + SHM) interchangeable for closed-loop function calling (linyueqian, vklimkov-nvidia).

## Realizing it: MiniCPM-o 4.5 native duplex (PR #3907)

PR #3907 is the first native landing — and, notably, it implements the *refined* RFC, not the first draft. It extends MiniCPM-o 4.5 from staged serving into a session-oriented audio stream over two endpoints, `/v1/duplex` (native) and `/v1/realtime?duplex=1` (an OpenAI-Realtime adapter), driving a real `audio-in → Stage0 listen/speak → Stage0→Stage1 handoff → Stage1 TTS/token2wav → audio-out` loop.

The ~27k-line PR maps cleanly onto the RFC's layers:

<div class="table-caption">Table 4. PR #3907 files by RFC layer.</div>
<table>
<thead><tr><th>RFC layer</th><th>Files in #3907</th></tr></thead>
<tbody>
<tr><td>Protocol</td><td><code>serving_duplex.py</code> (4.5k), <code>native_realtime_protocol.py</code>, <code>protocol/duplex.py</code>, <code>duplex_adapters/minicpmo45.py</code></td></tr>
<tr><td>Engine / session</td><td><code>engine/duplex.py</code>, <code>inputs/duplex_intermediate.py</code></td></tr>
<tr><td>Model / worker</td><td><code>minicpmo_4_5/duplex_runtime.py</code> (1.5k), <code>duplex_policy.py</code>, <code>duplex_worker_adapter.py</code>, <code>worker/native_duplex.py</code>, <code>stage_input_processors/minicpmo_4_5_omni.py</code></td></tr>
</tbody>
</table>

The most striking part is `engine/duplex.py`: its type system reads like a checklist of the review feedback above. Each abstraction the thread asked for is a first-class type:

<div class="table-caption">Table 5. How #3907's <code>engine/duplex.py</code> encodes the review.</div>
<table>
<thead><tr><th>Review ask</th><th>Type in #3907</th></tr></thead>
<tbody>
<tr><td>Gate duplex behind <code>session_mode</code></td><td><code>SessionMode</code></td></tr>
<tr><td>Three injection patterns, not one</td><td><code>DuplexAdapterPattern</code></td></tr>
<tr><td>Append as a per-model capability</td><td><code>DuplexInputMode</code></td></tr>
<tr><td>Turn-taking from multiple signal sources</td><td><code>DuplexSignalSource</code></td></tr>
<tr><td>"Full suite, models use a subset via config"</td><td><code>DuplexRuntimeCapabilities</code></td></tr>
<tr><td>Sent ≠ heard → committed offset</td><td><code>DuplexPlaybackCommitCursor</code> (<code>mark_generated</code> / <code>mark_sent</code> / <code>acknowledge</code>)</td></tr>
<tr><td>Request-as-stage-actor → stage binding</td><td><code>DuplexStageBinding</code></td></tr>
<tr><td>The session itself + registry</td><td><code>DuplexSessionRuntimeState</code>, <code>DuplexSessionRuntimeManager</code></td></tr>
</tbody>
</table>

MiniCPM is the **chunk-group append** pattern from Table 3, and the commit log is where the Omni-Flow specifics live: pacing bridge results to "the official 1-s-per-result rhythm," padding "the turn-end vocoder flush to one fixed window shape," slicing "past the leading listen run" so the **playback cursor survives turn close**, and keeping an "audio+transcript delta pair for text-less units." These are exactly the 1-second-chunk, `⟨chunk_eos⟩`, and playback-cursor details the design predicted.

What #3907 **honestly defers** is the one piece the RFC calls its deepest: the **persistent core KV lease** — `resumable`/session state here is *not* the full scheduler-owned lease (allocation, rollback, migration, release), nor one-long-lived-request-per-stage or byte-perfect Realtime. In RFC terms it nails the **control semantics** (Phase 0–3 territory) and leaves the Phase-1 KV lease for a follow-up. The H20 end-to-end run is the receipt: `overlap_listen=true`, `overlap_barge_in=true`, `playback_commit_ok=true`, and crucially `stale_audio_delta_count=0` — barge-in actually drops the stale stream.

## Phased rollout, and what's left

```mermaid
gantt
    title Full-Duplex Session Architecture
    dateFormat YYYY-MM-DD
    axisFormat %b
    section Phase 0
    Req/resp bring-up (MiniCPM-o 4.5, #3907) :done, p0, 2026-05-12, 20d
    section Phase 1
    DuplexSession + StageDuplexClient + KV lease :p1, after p0, 14d
    section Phase 2
    DUPLEX_VAD + OmniDuplexScheduler + cadence :p2, after p1, 14d
    section Phase 3
    Barge-in epoch + WS protocol :p3, after p2, 10d
    section Phase 4
    SHM direct-ingress ring + HC batching :p4, after p3, 10d
    section Phase 5
    Generalize to World Models (#1987) :p5, after p4, 14d
```

The phasing is latency-pinned: Phase 1 delivers the one-chunk floor (validated on MiniCPM-o 4.5 multi-turn — flat TTFT on the Nth turn, KV growing linearly with chunks instead of quadratically with re-prefill); the sub-300 ms floor needs sub-chunk cancel + `audio_end_ms` truncate + the WS endpoint in Phase 3. Phase 5 makes `DuplexChunk` an opaque typed payload so a World-Model (#1987) observe→predict loop reuses the same `DuplexSession` — **one session-management system, not two** (full-duplex is P1 on the roadmap, [issue #2136](https://github.com/vllm-project/vllm-omni/issues/2136)).

Three pieces remain between #3907 and the RFC's end state, in increasing difficulty: **API-server unification** (one `vllm serve` hosting the duplex path), **ASR/TTS onto `async_chunk` stages** (TTS first, streaming ASR the bigger lift), and **the full KV lease** — the RFC's own "deepest change, with its own sub-design + tests." Drive those through the phases and the request-oriented engine grows a real session skeleton. That's the whole point: full-duplex isn't "adapt one more model," it's swapping the unit of work from a request to a session — and MiniCPM-o 4.5 is the first model running on it.

---

*Sources: RFC [#3745](https://github.com/vllm-project/vllm-omni/issues/3745) and its discussion; the MiniCPM-o 4.5 duplex PR [#3907](https://github.com/vllm-project/vllm-omni/pull/3907); the vLLM-Omni systems paper ([arXiv:2602.02204](https://arxiv.org/abs/2602.02204), Figure 1) and the official Apr-2026 meetup deck (Figure 2). Project: [github.com/vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni).*
