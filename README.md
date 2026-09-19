<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-lockup-dark.png" />
    <img src="assets/logo-lockup.png" alt="Laya" width="330" />
  </picture>
</p>

**Multilingual, non-autoregressive System 1 decision engine — for .NET.**
Typed decisions over 100+ languages in a single forward pass, with calibrated probabilities, on
managed CPU SIMD. No Python, no PyTorch, no native dependency.

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![.NET](https://img.shields.io/badge/.NET-10%20%7C%2011-512BD4)](https://dotnet.microsoft.com/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-convaiinnovations%2Flaya-blue)](https://huggingface.co/convaiinnovations/laya)

Laya answers typed questions (`choice`, `score`, `noul`) about any state — text, an email, a ticket,
a JSON document — in **one forward pass per question set**. Nothing is generated, so there is
nothing to parse and nothing to hallucinate; every answer comes back as a distribution with a
calibrated confidence.

This repository is the C# port. The original Python implementation is preserved verbatim under
[`.reference/`](.reference/) and is the behavioural specification the port is tested against —
layer by layer, see [Parity](#parity).

---

## Status

| | |
|---|---|
| Checkpoints | `english`, `multilingual`, `typed-decisions` — all three run |
| Parity with PyTorch | every encoder layer, head layer, logit and answer, on all three checkpoints |
| Tokenizers | byte-level BPE and SentencePiece-style BPE, exact against `transformers` |
| Execution | CPU, managed SIMD (AVX2 / AVX-512 / NEON), multi-threaded |
| Not ported | training, GPU execution |

---

## Install

There is no package feed yet; build from source. The repository targets **.NET 10** and **.NET 11**
— the `net11.0` target is added automatically when an 11.x SDK is installed, so a .NET 10 SDK
builds it unchanged.

```bash
git clone https://github.com/theolivenbaum/laya.git
cd laya
dotnet build Laya.slnx -c Release
```

---

## Quickstart

```csharp
using Laya;
using Laya.Runtime;

// Downloads the English checkpoint on first use (~840 MB) into ~/.cache/laya.
using var agent = Agent.Load();

var state = new List<KeyValuePair<string, object?>>
{
    new("from",    "user@acme.com"),
    new("subject", "Duplicate charge on invoice #4411"),
    new("body",    "Hi, we were billed twice for March. Please refund the duplicate today "
                 + "or we will cancel our plan."),
};

var result = agent.SystemOne(state, Presets.Triage());

Console.WriteLine(result["intent"].Choice);              // refund
Console.WriteLine(result["frustration"].Score);          // 1.91
Console.WriteLine(result["churn_risk"].Noul);            // 0.95
Console.WriteLine(result["intent"].Confidence);          // 0.62
```

Questions are typed, and the type decides what comes back:

```csharp
var questions = new QuestionSet()
    .Add("intent", Question.Choice("What does the customer want in `body`?",
        ("refund",         "money returned or a duplicate charge reversed"),
        ("technical_help", "a bug, outage or integration problem"),
        ("other",          "none of the other options fits")))
    .Add("urgency", Question.Score("How urgent is this?",
        "no time pressure", "needs attention soon", "blocking issue or hard deadline"))
    .Add("needs_reply", Question.Noul("Does the sender expect a reply?"));
```

| type | answer | also returns |
|---|---|---|
| `choice` | the winning option key | a probability per option |
| `score` | the expected level, `Σ i · p(i)` | a probability per level, and the level legend |
| `noul` | the probability the statement holds | — |

Every answer carries a `Confidence` (normalised entropy) and an `Action.ActProbability` from the
model's escalation head.

### Routing between checkpoints

The English checkpoint does not degrade gently off English — it collapses, confidently (0.100 on
20-option Hindi intent, against 0.050 for random). `Router` detects the script first and the
language second, and sends each request to a checkpoint that can read it:

```csharp
using var router = new Router(maxLoaded: 2);

router.Predict("I was charged twice", Presets.Triage());                  // -> english
router.Predict("請求書4411で二重に請求されました", Presets.Triage());           // -> multilingual
router.Predict("Mir wurde meine Rechnung letzten Monat zweimal abgebucht " +
               "und der Support hat nicht geantwortet", Presets.Triage());  // -> multilingual

// Routing on its own loads nothing and costs microseconds.
var decision = router.Route("मुझसे दो बार शुल्क लिया गया", Presets.Triage());
Console.WriteLine(decision.Reason);
// non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it
```

A cold load costs seconds while detection costs microseconds, so a server that alternates languages
should `Preload()` rather than let the LRU evict on every request.

Note the asymmetry: script detection is exact, so *any* amount of non-Latin text routes correctly,
but the Latin-script language guess is a stopword heuristic that deliberately abstains on short
inputs. `"Mein Konto wurde zweimal belastet"` routes to **english** — five words is not enough
evidence to overrule the default, and guessing wrong on ordinary English would be worse. Pass
`lang:` or `model:` when you already know.

### Presets

`Presets.Triage()`, `Presets.Email()`, `Presets.Guard()`, `Presets.Moderation()` and
`Presets.ModelRouter()` are the shipped question sets — support triage, email and threat filtering,
LLM input guardrails, content moderation, and model routing. `laya presets` prints them.

---

## Checkpoints

| name | encoder | params | context | use it for |
|---|---|---|---|---|
| `english` | ModernBERT-large | 421M | 512 | English |
| `multilingual` | mmBERT-base | 322M | 1024 | 100+ languages |
| `typed-decisions` | ModernBERT-large | 421M | 1024 | the four typed-decisions workflows |

They are published twice. [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
bundles all three — English at the root, the other two in subfolders — and each also has its own
repository: [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) and
[`laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions), where the
same five files sit at the root. Either layout works, and only the checkpoint you ask for is
fetched: the bundle is 2.3 GB, one checkpoint is 640–840 MB.

```bash
# From the bundle — downloads the multilingual subfolder and nothing else
dotnet run --project src/Laya.Cli -- download --model multilingual --cache artifacts/models-cache

# From its own repository
dotnet run --project src/Laya.Cli -- download --model typed-decisions --standalone
dotnet run --project src/Laya.Cli -- download --repo convaiinnovations/laya-typed-decisions
```

In code, `Agent.Load` takes the same two shapes, and `Router` has a catalogue for each:

```csharp
using var fromBundle     = Agent.Load("convaiinnovations/laya", subfolder: "typed-decisions");
using var fromOwnRepo    = Agent.Load("convaiinnovations/laya-typed-decisions");
using var standaloneOnly = new Router(standaloneRepos: true);
```

Set `HF_TOKEN` for a gated or private repository. Downloads resume, and re-running is a no-op.

---

## Command line

```
laya download      Fetch a checkpoint from Hugging Face
laya predict       Answer a question set about some state
laya route         Show which checkpoint a state would route to (loads nothing)
laya presets       List the built-in question sets
laya tokenize      Tokenize text with a checkpoint's tokenizer
laya dump-states   Write per-layer activations for parity checking
laya bench         Time the forward pass
```

```bash
dotnet run --project src/Laya.Cli -c Release -- \
    predict --model-dir artifacts/models/english --preset triage \
            --text "I was charged twice and nobody answers"

dotnet run --project src/Laya.Cli -c Release -- \
    predict --model-dir artifacts/models/english \
            --question 'urgent=noul:Is this time critical?' \
            --question 'team=choice:Who handles this?|billing,support,security' \
            --state-file ticket.json
```

---

## How it works

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
```

The state and every option are encoded together by a bidirectional encoder. A small typed head
reads the hidden state at each option's `[MASK]` marker and scores it, so *all* the options are
compared in one pass rather than generated one token at a time.

1. **ModernBERT encoder** — pre-norm, RoPE, GeGLU, and alternating attention: every third layer is
   full attention, the rest see a 128-token sliding window.
2. **Typed decision head** — two pre-norm transformer layers, plus a 3-entry question-type
   embedding added to every position.
3. **Scorer** — reads each `[MASK]` marker and emits one logit per option.
4. **Calibration** — a temperature fitted per question type *and* option count is applied before
   the softmax, which is what makes the reported probabilities mean something.
5. **Action head** — predicts whether to escalate, from the pooled `[CLS]` state plus four
   features of the option distribution.

### Inside the port

- **Sequences are concatenated, not padded.** All the questions in a set go through the encoder as
  one tall matrix, so the 421M parameters are read from memory once per call instead of once per
  question — and unlike a padded batch, a short question does not pay for the longest one in the
  set. Nothing crosses between questions: attention, the type embedding and the marker read-out are
  all per sequence.
- **Weights are repacked at load.** PyTorch stores `nn.Linear` weights as `[out, in]`. They are
  transposed once into panels of two SIMD vectors, so the inner loop broadcasts one activation and
  multiply-accumulates a contiguous run of outputs — eight FMAs against two vector loads, unit
  stride stores, no horizontal reduction. See
  [`PackedMatrix`](src/Laya/Numerics/PackedMatrix.cs) for why the *panel* layout, and not a plain
  transpose, is what makes this pay.
- **AVX-512 is used explicitly where it exists.** `Vector<T>` stays 256-bit on AVX-512 hardware
  unless the whole process opts in; a sustained GEMM is exactly the case that wants the wider
  vectors, so the kernel reaches for them directly. `LAYA_VECTOR_BITS=256` opts out.
- **Everything computes in fp32.** The checkpoints are fp16 on disk and widened once at load,
  which is what PyTorch does on CPU too — and what the parity tolerances are measured against.
  Keeping them fp16 in memory would halve the 1.57 GiB the weights occupy, but there is no
  vectorized fp16-to-fp32 widening to lean on, and bf16 would cost more accuracy than the parity
  budget allows.
- **Scratch is rented, not allocated.** Every buffer a pass needs is the same size every time and
  dies immediately, so they come from the array pool. This is the difference between 144 MiB of
  garbage per call and 1.8 MiB.
- **Attention keeps the thing being summed in the lanes.** A dot product per query-key pair ends
  in a horizontal reduction, and a forward pass has ~14 million of those pairs. The keys are
  gathered transposed, so adjacent keys sit in adjacent lanes and a score vector finishes with a
  plain store; the value accumulation likewise holds its running total in registers across the
  whole key loop. See [`AttentionKernels`](src/Laya/Numerics/AttentionKernels.cs).

### Performance

Measured on a 4-core Xeon @ 2.8 GHz with AVX-512, on the 5-question triage preset over a paragraph
of state (404 tokens in total). PyTorch is the reference implementation in `.reference/` on the
same machine and the same weights.

| | 1 thread | 4 threads |
|---|---|---|
| this port | **3.63 s** | 1.29 s |
| PyTorch 2.14 CPU (oneDNN) | 4.10 s | 1.23 s |
| this port, before any optimization | 16.5 s | 4.9 s |

Per call, in steady state: **1.8 MiB** allocated, **zero** GC collections, **2.0 GiB** working set
(1.57 GiB of that is the fp32 weights).

#### Where the time goes, and what is left

The projection kernel is ~76% of a forward pass and runs at **103 GFLOP/s** single-threaded. That
number only means something against this machine's actual roof, so it was measured: a loop of pure
register-to-register 512-bit FMAs sustains **151.7 GFLOP/s**, because the core drops to 2.37 GHz
under AVX-512 (the 256-bit roof is 89.5 GFLOP/s at the full 2.8 GHz). So the GEMM is at 68% of the
roof, and the same instruction mix in isolation — L1-resident, no panel switching — reaches ~130.

That bounds what is still available. Even a *perfect* GEMM would put the whole pass at ~2.7 s, and
2x against PyTorch would require the entire forward pass, attention and normalization included, to
sustain ~96% of the pure-FMA roof. That is not reachable in fp32; it would take lower-precision
arithmetic, which this port deliberately does not do because parity is measured against fp32.

| projection | shape | 1 thread | 4 threads |
|---|---|---|---|
| `mlp_in` | 404 × 1024 × 5248 | 42.1 ms — 103 GFLOP/s | 20.3 ms — 214 GFLOP/s |
| `qkv` | 404 × 1024 × 3072 | 24.9 ms — 102 GFLOP/s | 13.5 ms — 189 GFLOP/s |
| `mlp_out` | 404 × 2624 × 1024 | 25.5 ms — 85 GFLOP/s | 11.7 ms — 185 GFLOP/s |
| `attn_out` | 404 × 1024 × 1024 | 10.4 ms — 82 GFLOP/s | 4.9 ms — 174 GFLOP/s |

Reproduce any of it:

```bash
# End to end, with per-stage timings, allocation totals and a sampling profile
dotnet run --project src/Laya.Cli -c Release -- profile \
    --model-dir artifacts/models/english --preset triage --text "…" --threads 1

# BenchmarkDotNet: the GEMM shapes, the elementwise kernels, the whole forward pass
dotnet run --project benchmarks/Laya.Benchmarks -c Release -- --filter '*Gemm*'
dotnet run --project benchmarks/Laya.Benchmarks -c Release -- --filter '*Forward*'
```

`laya profile` needs no external tooling: stage timings come from the model itself, and the CPU
sampling profile and allocation report are captured in-process through
[`Memory.Introspect`](https://www.nuget.org/packages/Memory.Introspect/).

GPU execution is out of scope for this port.

---

## Parity

The port is checked against the Python implementation tensor by tensor, not just on its answers.

```bash
# 1. Dump reference tensors from PyTorch (needs torch + transformers)
python tools/dump_reference.py --model-dir artifacts/models/english \
    --text "I was charged twice" --preset triage --out artifacts/dumps/torch.json

# 2. Dump the same tensors from the .NET implementation
dotnet run --project src/Laya.Cli -c Release -- dump-states \
    --model-dir artifacts/models/english \
    --text "I was charged twice" --preset triage --out artifacts/dumps/dotnet.json

# 3. Diff them
python tools/compare_dumps.py artifacts/dumps/torch.json artifacts/dumps/dotnet.json
```

Golden dumps for all three checkpoints are committed under
[`tests/Laya.Tests/Fixtures/`](tests/Laya.Tests/Fixtures), so `dotnet test` verifies parity without
Python — it only needs the weights. Every encoder layer, head layer, logit, action logit and final
answer is compared; on the English checkpoint the worst per-layer deviation is ~3e-5 absolute
against activations in the tens of thousands, and the answers agree to every reported digit.

```bash
dotnet test tests/Laya.Tests -c Release
```

Tests that need weights skip themselves when `artifacts/models/` is empty; point
`LAYA_TEST_MODELS` somewhere else to override.

---

## Repository layout

```
.reference/            the original Python package — the behavioural specification
src/Laya/              the port: numerics, tokenizers, ModernBERT, decision head, runtime
src/Laya.Cli/          the command line tool
benchmarks/            BenchmarkDotNet suites for the kernels and the forward pass
tests/Laya.Tests/      xunit tests, including the PyTorch parity fixtures
tools/                 Python scripts that produce reference dumps
```

[`CLAUDE.md`](CLAUDE.md) documents the architecture that has to be reproduced exactly, and the
traps that cost the most time. [`TODO.md`](TODO.md) tracks what is done and what is not.

---

## License

Apache-2.0, as the original. The model weights are published by Convai Innovations under the same
license.
