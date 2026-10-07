<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-lockup-dark.png" />
    <img src="assets/logo-lockup.png" alt="Laya" width="330" />
  </picture>
</p>

**Typed decisions for .NET, in one forward pass — no Python, no PyTorch, no native dependency.**

[![NuGet](https://img.shields.io/nuget/v/Laya?color=004880&label=NuGet)](https://www.nuget.org/packages/Laya)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![.NET](https://img.shields.io/badge/.NET-10%20%7C%2011-512BD4)](https://dotnet.microsoft.com/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-convaiinnovations%2Flaya-blue)](https://huggingface.co/convaiinnovations/laya)

`Laya` is a C# library that answers **typed questions about a piece of state** — a message, an
email, a ticket, a JSON document — and returns a calibrated probability distribution for each
answer. Ask five questions about one ticket and you get five answers back in a single call, in
roughly the time one small classification takes.

Nothing is generated. There is no prompt to tune, no JSON to parse, no schema to validate, and
nothing to hallucinate: the model scores the options *you* define and hands back numbers.

```csharp
using Laya;
using Laya.Io;
using Laya.Runtime;

using var agent = RemoteCheckpoint.English.Load();

var result = agent.SystemOne("I was charged twice and nobody answers", Presets.Triage());

result["intent"].Choice;          // "refund"
result["frustration"].Score;      // 1.99  (expected level on the 0..3 legend)
result["refund_requested"].Noul;  // 0.90  (probability the statement holds)
result["intent"].Confidence;      // 0.48
```

It runs on the CPU, on managed SIMD (AVX2 / AVX-512 / NEON), and holds the model in your own
process — no service to call, no per-token bill, no data leaving the machine.

This is a C# implementation of [Laya](https://github.com/NandhaKishorM/laya) by Convai
Innovations, verified tensor-by-tensor against the original PyTorch implementation.

---

## Install

```bash
dotnet add package Laya
```

Targets **.NET 10** and **.NET 11**. The package contains no model data: a checkpoint is
downloaded on first use and cached, see [Checkpoints](#checkpoints).

---

## What you can build with it

Each of these is one call, five answers, one model load:

| | |
|---|---|
| **Support triage** | intent, urgency, frustration, refund requested, churn risk |
| **Email routing** | destination team, spam, phishing, urgency, needs a reply |
| **LLM guardrails** | jailbreak, prompt injection, sensitive data, harm severity, topic |
| **Content moderation** | toxicity, harassment, threats, spam, severity |
| **Model routing** | how hard is this request, does it need tools, does it need a big model |

Those five ship as `Presets`. Everything else is your own `QuestionSet`.

---

## Questions and answers

A question has a type, an instruction, and — for `choice` and `score` — the options or levels it
must decide between. The type decides what comes back.

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

| type | `Answer` member | what it means | also returns |
|---|---|---|---|
| `choice` | `Choice` | the winning option key | `Probabilities`, one per option |
| `score` | `Score` | the expected level, `Σ i · p(i)` | `Probabilities` and the level `Legend` |
| `noul` | `Noul` | the probability the statement holds | — |

Every answer also carries:

* `Confidence` — normalised entropy of the distribution, `1.0` when the model is certain.
* `Action.ActProbability` — the model's own escalate-to-a-human signal, from a separate head.
* `ProbabilityOf("refund")` — the calibrated probability of any single option.

```csharp
foreach (var (id, answer) in result.Answers)
{
    Console.WriteLine($"{id}: {answer.Choice ?? answer.Score?.ToString() ?? answer.Noul?.ToString()} " +
                      $"(confidence {answer.Confidence:P0})");

    if (answer.Probabilities is { } distribution)
        foreach (var (option, p) in distribution) Console.WriteLine($"    {option,-16} {p:P1}");
}

Console.WriteLine($"{result.Usage.InputTokens} input tokens, {result.Usage.OutputTokens} generated");
```

Write your own thresholds against the numbers rather than trusting a label:

```csharp
var answer = result["refund_requested"];
if (answer.Noul > 0.9)                    AutoRefund(ticket);
else if (answer.Action.ActProbability > 0.5) Escalate(ticket);
else                                      Queue(ticket);
```

### State can be anything

A string, key-value pairs, or JSON — it is serialised into the sequence the same way the Python
implementation does it, so field names are part of what the model reads. Name them well; the
instructions can refer to them with backticks.

```csharp
// A string
agent.SystemOne("I was charged twice", Presets.Triage());

// Fields — order is preserved
agent.SystemOne(new List<KeyValuePair<string, object?>>
{
    new("from",    "user@acme.com"),
    new("subject", "Duplicate charge on invoice #4411"),
    new("body",    "We were billed twice for March. Refund it or we cancel."),
}, Presets.Triage());

// JSON straight from a request body
agent.SystemOne(JsonDocument.Parse(json).RootElement, Presets.Triage());

// An email, with quoted replies, signatures and footers stripped first
agent.SystemOne(EmailUtils.EmailState(subject, body, sender), Presets.Email());
```

---

## Checkpoints

Three checkpoints, all Apache-2.0, all downloaded on demand:

| `RemoteCheckpoint` | encoder | params | context | download | use it for |
|---|---|---|---|---|---|
| `English` | ModernBERT-large | 421M | 512 tokens | 843 MB | English text, general questions |
| `Multilingual` | mmBERT-base | 322M | 1024 tokens | 644 MB | 100+ languages, ~2× faster |
| `TypedDecisions` | ModernBERT-large | 421M | 1024 tokens | 843 MB | the four typed-decisions workflows |

`English` and `TypedDecisions` share an encoder and a head; they differ in what they were trained
on and how much they can read. `TypedDecisions` is fine-tuned on four synthetic operational
workflows — customer service, invoice processing, security incidents and agent-trace
observability — and doubles the context to 1024 tokens. On that workflow benchmark it scores
0.766 where `English` scores 0.362, below the 0.461 majority-class baseline; `English` is the one
trained on the general English mix (AG News 0.947, BoolQ 0.830) and stays the default. So pick
`TypedDecisions` deliberately, for those workflows — it is never selected automatically.

`English` does not degrade gently outside English, it collapses: 0.100 on 20-option Hindi intent
against 0.050 for random guessing, while reporting high confidence. Use `Multilingual` or the
`Router` for anything that is not reliably English.

### Downloading and caching

```csharp
using Laya.Io;

// Downloads on first use, then loads from the cache every time after.
using var agent = RemoteCheckpoint.English.Load();
```

Files come from `https://models.curiosity.ai/laya/` and land in `~/.cache/laya` (override with
`$LAYA_HOME`, or `$HF_HOME/laya`). Interrupted downloads resume, a file already on disk is never
re-fetched, and two processes starting at once share one download instead of racing.

```csharp
var checkpoint = RemoteCheckpoint.Multilingual;

checkpoint.IsDownloaded();                       // is the cache warm?
await checkpoint.PrepareAsync(                   // warm it at deploy time, with progress
    progress: new Progress<DownloadProgress>(p =>
        Console.WriteLine($"{p.File} {p.Fraction:P0}")),
    cancellationToken: stoppingToken);

checkpoint.BaseUri;                              // https://models.curiosity.ai/laya/multilingual/main/
RemoteCheckpoint.FromName("typed-decisions");    // by name, for configuration
```

`LAYA_MODEL_BASE_URL` points the download at a mirror with the same
`<checkpoint>/<revision>/<file>` layout. Alternatively, `Agent.Load()` fetches a checkpoint from
the Hugging Face hub (set `HF_TOKEN` for a gated repository), and `Agent.FromDirectory(path)`
loads one you shipped yourself.

### In a long-lived service

Loading a checkpoint reads ~840 MB and builds 1.57 GiB of fp32 weights, so do it once. Everything
an `Agent` holds after that is read-only, and every call rents its own scratch from the array
pool, so one instance serves the whole process.

```csharp
builder.Services.AddSingleton(_ => RemoteCheckpoint.English.Load());

// Warm the cache before the first request instead of during it.
await RemoteCheckpoint.English.PrepareAsync();
```

The kernels already use every core for a single call (`LAYA_THREADS`, or `--threads`, caps it), so
throughput comes from batching questions into one `SystemOne` call rather than from calling it
concurrently. A host that does want passes side by side hands each one its own budget instead of
the process-wide default - the thread count never changes a result, only how long it takes:

```csharp
var quarter = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount / 4 };
var result = agent.SystemOne(state, questions, quarter);   // also on Predict, Router.Predict, DecisionModel.Forward
```

---

## d1: LiquidAI's decision model

[`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) is a 3.1B-parameter multimodal decision
model built on LFM2.5-VL-3B: the same typed questions (`choice`, `score`, `noul`), answered in one
causal forward pass with zero output tokens, over text, JSON and pictures. `D1Agent` runs it
in-process, implements `IDecisionEngine`, and takes the same `QuestionSet`s as `Agent`.

```csharp
using Laya.D1;

using var d1 = D1Agent.Load();          // LiquidAI/d1-3B from Hugging Face (~6.3 GB), cached
var result = d1.SystemOne("I was charged twice this month, please refund one of them.", Presets.Triage());

// Pictures are part of the state; with no text, the pictures are the whole state.
var photo = RgbImage.Load("cats.png");  // PPM or PNG; decode anything else to RGB bytes yourself
var cats = d1.SystemOne(null, new QuestionSet().Add("count",
    Question.Choice("How many cats are there?", ("one", "One"), ("two", "Two"), ("more", "Three or more"))), [photo]);
```

```bash
laya download --model d1
laya predict --model d1 --preset triage --text "I was charged twice"
laya predict --model d1 --image photo.png --question 'cat=noul:Is there a cat?'
```

How it differs from a laya checkpoint:

* **There is no decision head.** The state and the question are rendered as a chat prompt (the
  checkpoint's own `prompt.py`, ported verbatim in `D1Prompt`: options become single-token codes
  `A`, `B`, …), and the answer is a softmax over those codes' LM-head logits at the last position.
  Only the option tokens' logits are computed — the log-normalizer cancels — so the 128k-entry
  LM head costs a dozen dot products.
* **Several questions read the state once.** The state is the trunk of a tree and every question a
  branch (`Lfm2Tree`, the reference's `hybrid.Tree`): token-wise work runs over all branches in one
  product, a branch's convolution continues from the trunk's last tokens, and its attention sees
  the trunk and itself, never a sibling.
* **The answers are d1's:** `noul` is P(yes), `confidence` is the top probability (for a `noul`,
  max(p, 1 − p)); values are not rounded and there is no `action` estimate.
* **License.** The weights are LiquidAI's, under the LFM Open License v1.0, and are downloaded
  from Hugging Face rather than mirrored.

The backbone is a 30-layer LFM2 hybrid (22 gated short-convolution layers, 8 GQA attention layers,
SwiGLU MLPs, tied LM head) kept in bf16 — 5.0 GiB resident, half of fp32, and bit-identical inputs
to the fp32 arithmetic PyTorch does on CPU. Pictures go through the SigLIP2 NaFlex tower (27
layers), a 2 × 2 pixel unshuffle and a two-layer projector; tiling, thumbnails, and both the
torchvision and Pillow bicubic resizes are reproduced bit for bit.

Single-threaded on the same 4-core Xeon, warm, against the checkpoint's own PyTorch code in fp32:

| | 61 tokens, 1 question | 116 tokens, 3 questions | 608 tokens, 3 questions |
|---|---|---|---|
| this library | **3.2–3.5 s** | **6.4 s** | 30.4 s |
| PyTorch 2.14 (MKL), as shipped | 20.9 s | 39.4 s | 193 s |
| PyTorch 2.14 with `set_flush_denormal(True)` | 4.1 s | 6.6 s | **26.4 s** |

About a third of d1's short-convolution input weights are bf16 *subnormals*. Every x86 FMA that
touches one takes a microcode assist, and neither PyTorch nor .NET sets flush-to-zero, so the
reference spends most of its time there. `BFloat16Matrix` flushes subnormal weights and
activations to zero — numerically invisible (a 1e-39 term in an fp32 sum of order one), and the
parity tests are unchanged by it. Against PyTorch with flush-to-zero turned on, short decisions
are faster here; on long states MKL's GEMM (~130 GFLOP/s at 600 rows vs ~100 here) is ahead.

---

## Routing between languages

`Router` detects the script first and the language second, then sends each request to a checkpoint
that can read it — loading checkpoints lazily and evicting the least recently used one.

```csharp
using var router = new Router(maxLoaded: 2);

router.Predict("I was charged twice",               Presets.Triage());  // -> english
router.Predict("Mein Konto wurde zweimal belastet", Presets.Triage());  // -> multilingual
router.Predict("請求書4411で二重に請求されました",       Presets.Triage());  // -> multilingual

// Routing on its own loads nothing and costs microseconds.
RouteDecision decision = router.Route("मुझसे दो बार शुल्क लिया गया");
Console.WriteLine(decision.Model);   // multilingual
Console.WriteLine(decision.Reason);  // non-Latin script (devanagari, 100% of letters); …
```

A cold load costs seconds while detection costs microseconds, so a server that alternates
languages should `Preload()` rather than let the LRU evict on every request:

```csharp
using var router = new Router().Preload(["english", "multilingual"]);
```

`LanguageDetector.Analyse(state)` exposes the same detection on its own — script profile, guessed
Latin language, whether the text is English, and whether the language was identified at all — with
no model loaded.

The Latin-script guess is a stopword and diacritic heuristic, and it is honest about what it cannot
see: a language is named only on a word no other list claims, and text it cannot identify that
carries non-English letters (Romanian, Polish, Czech, Turkish, Vietnamese, …) goes to the
multilingual checkpoint rather than being assumed English. What it still cannot see is a plain-ASCII
language it holds no list for — Indonesian, Swahili, Tagalog, Turkish with its letters stripped. For
those, plug in a statistical language identifier; `Laya.Catalyst` wraps
[Catalyst](https://github.com/curiosity-ai/catalyst)'s (53 languages, model embedded, no download):

```csharp
// dotnet add package Laya.Catalyst
var router = new Router { LanguageClassifier = await CatalystLanguageClassifier.CreateAsync() };
router.Route("Saya ditagih dua kali untuk langganan saya bulan ini").Model;   // multilingual
```

The classifier is consulted only for text the heuristic left undecided and would have sent to the
English checkpoint, so it can move a state to the multilingual checkpoint but never an identified
one away from it.

---

## Many labels: shortlist first

Every option of a `choice` shares one `head_max_len` budget, so a 70-way intent leaves each label a
handful of tokens. `Shortlist.PredictShortlist` embeds the state and the labels, keeps the top `k` by
cosine similarity and runs one decision pass over those:

```csharp
EmbedFunction embed = Shortlist.EmbedFunctionFromAgent(agent);   // or any bi-encoder of your own
var result = Shortlist.PredictShortlist(agent, state, questions, embed, k: 20);
result.Shortlist;   // per question: the kept labels, their similarities, k and n
```

Probabilities on a shortlisted question are over the kept labels only.

---

## Command line

The `Laya.Cli` project is a thin face over the library; everything it does is public API.

```
laya download      Fetch a checkpoint (from the model host, or --source hub)
laya predict       Answer a question set about some state
laya route         Show which checkpoint a state would route to (loads nothing)
laya presets       List the built-in question sets
laya tokenize      Tokenize text with a checkpoint's tokenizer
laya dump-states   Write per-layer activations for parity checking
laya bench         Time the forward pass
laya profile       Stage timings, allocations and a sampling profile
laya dataset       Download a typed-decisions split to JSON lines
laya train         Fine-tune a checkpoint (see Fine-tuning)
laya calibrate     Refit a checkpoint's temperatures on labelled cases, weights untouched
laya evaluate      Accuracy, soft accuracy, Brier, ECE, KL, score MAE on typed-decisions cases
```

```bash
dotnet run --project src/Laya.Cli -c Release -- download --model english

dotnet run --project src/Laya.Cli -c Release -- \
    predict --model english --preset triage --text "I was charged twice and nobody answers"

dotnet run --project src/Laya.Cli -c Release -- \
    predict --model english \
            --question 'urgent=noul:Is this time critical?' \
            --question 'team=choice:Who handles this?|billing,support,security' \
            --state-file ticket.json
```

---

## Fine-tuning

`Laya.Training` fine-tunes a checkpoint on your own typed decisions, on the CPU. It is the C# port
of the reference [Kaggle notebook](.reference/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb):
the same RLCD objective (a group-relative policy gradient over noisy logits, scored by a strictly
proper reward, plus soft cross-entropy), AdamW with cosine annealing, gradient clipping, and
post-training temperature calibration. The result is an ordinary checkpoint directory that
`Agent.FromDirectory` loads.

```bash
laya dataset --split train                                   # LocalLLaMA/typed-decisions -> JSON lines
laya train --model english --data artifacts/data/LocalLLaMA_typed-decisions.all.train.jsonl \
           --out artifacts/models/my-decisions --train-layers 4
laya evaluate --model-dir artifacts/models/my-decisions --split test
```

```csharp
using Laya.Training;

var cases = DecisionDataset.ReadJsonLines("train.jsonl");       // {state, questions, gold} per line
var items = DecisionDataset.BuildItems(tokenizer, cases, maxLength: 512, headMaxLength: 192);
var report = new Trainer(new TrainerOptions { TrainableEncoderLayers = 4 })
    .Train(sourceDirectory, [.. items.Select(i => i.Item)], "my-decisions");
```

Training the whole 421M-parameter encoder needs about 10 GB of RAM and, on four cores, 15–25 s per
micro-batch of four sequences; `--train-layers n` trains only the top `n` encoder layers (0 = the
head only) and costs proportionally less. Every layer is checkpointed, so activation memory stays
at one layer's worth whatever the batch. The defaults are the notebook's; the notebook ran two GPUs,
so its effective batch was twice one process's.

Three things differ from the notebook on purpose. Its calibration set, `all_items[::15]`, strides
over items grouped five questions to a case, so every sample is the same question — a `choice` —
and `score` and `noul` are never fitted; the same number of items is drawn at random instead.
Calibration also fits the `type:option-count`
buckets that inference reads first, and replaces the base checkpoint's buckets — the notebook left
them in place, where they shadowed its new per-type fit. And fitted temperatures are clamped to the
range inference applies, so what is saved is what is used.

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

### Inside the implementation

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
- **Scratch is rented, not allocated.** Every buffer a pass needs is the same size every time and
  dies immediately, so they come from the array pool. This is the difference between 144 MiB of
  garbage per call and 1.8 MiB.
- **Attention keeps the thing being summed in the lanes.** A dot product per query-key pair ends
  in a horizontal reduction, and a forward pass has ~14 million of those pairs. The keys are
  gathered transposed, so adjacent keys sit in adjacent lanes and a score vector finishes with a
  plain store. See [`AttentionKernels`](src/Laya/Numerics/AttentionKernels.cs).

---

## Performance

Measured on a 4-core Xeon @ 2.8 GHz with AVX-512, on the 5-question triage preset over a paragraph
of state (404 tokens in total). PyTorch is the original implementation on the same machine and the
same weights.

| | 1 thread | 4 threads |
|---|---|---|
| this library | **3.63 s** | 1.29 s |
| PyTorch 2.14 CPU (oneDNN) | 4.10 s | 1.23 s |
| this library, before any optimization | 16.5 s | 4.9 s |

Per call, in steady state: **1.8 MiB** allocated, **zero** GC collections, **2.0 GiB** working set
(1.57 GiB of that is the fp32 weights).

The projection kernel is ~76% of a forward pass and runs at **103 GFLOP/s** single-threaded —
68% of this machine's measured ceiling of 151.7 GFLOP/s for pure register-to-register 512-bit FMAs
(the core drops to 2.37 GHz under AVX-512; the 256-bit roof is 89.5 GFLOP/s at the full 2.8 GHz).
Even a *perfect* GEMM would put the whole pass at ~2.7 s, so what is left is bounded and small.

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
    --model english --preset triage --text "…" --threads 1

# BenchmarkDotNet: the GEMM shapes, the elementwise kernels, the whole forward pass
dotnet run --project benchmarks/Laya.Benchmarks -c Release -- --filter '*Gemm*'
```

`laya profile` needs no external tooling: stage timings come from the model itself, and the CPU
sampling profile and allocation report are captured in-process through
[`Memory.Introspect`](https://www.nuget.org/packages/Memory.Introspect/).

GPU execution and training are out of scope; this library runs inference on the CPU.

---

## Correctness

The answers are not "close enough" — the implementation is checked against the original PyTorch
one tensor by tensor: every encoder layer, head layer, logit, action logit and final answer, on
all three checkpoints. On the English checkpoint the worst per-layer deviation is ~3e-5 absolute
against activations in the tens of thousands, and the answers agree to every reported digit. The
tokenizers (byte-level BPE and SentencePiece-style BPE) match `transformers` exactly.

Training is held to the same standard: every parameter gradient of a small random
`DecisionModel` matches PyTorch autograd to ~1e-6 relative, a subset of the real English
checkpoint's gradients to ~1e-5, the RLCD loss and its logit gradient match the notebook's code on
the same random draws, AdamW and the cosine schedule match `torch.optim`, and training sequences
match the notebook's `build_training_item` token for token.

d1 is checked the same way against the checkpoint's own PyTorch code: the rendered prompts and
token ids exactly, every one of the 30 LFM2 layers (and the conv and attention internals of the
first of each), the full-vocabulary log-softmax at every answer slot (within 2.5e-5) and the
answers, on a lone question and on question trees. For pictures: the resized, normalized patches
(to 1.2e-7 — both bicubic resizers are bit-exact), all 27 SigLIP2 layers, the projected features
and the language model over them. The tokenizer matches `transformers` on 347 strings, and the
bf16 GEMM is bit-identical to a plain fp32 FMA chain.

Golden dumps live in [`tests/Laya.Tests/Fixtures/`](tests/Laya.Tests/Fixtures), so the parity
suite runs without Python:

```bash
dotnet test tests/Laya.Tests -c Release     # tests needing weights skip themselves without them
```

To regenerate them, [`tools/`](tools) has the PyTorch dump-and-compare scripts, and
`laya dump-states` writes the same tensors from this side.

---

## Building from source

```bash
git clone https://github.com/theolivenbaum/laya.git
cd laya
dotnet build Laya.slnx -c Release
dotnet test  tests/Laya.Tests -c Release
```

The `net11.0` target is added automatically when an 11.x SDK is installed, so a .NET 10 SDK builds
the repository unchanged; force it either way with `-p:LayaEnableNet11=true|false`.

```
src/Laya/              the library: numerics, tokenizers, ModernBERT, decision head, runtime
src/Laya.Cli/          the command line tool
benchmarks/            BenchmarkDotNet suites for the kernels and the forward pass
tests/Laya.Tests/      xunit tests, including the PyTorch parity fixtures
tools/                 Python scripts that produce reference dumps
.reference/            the original Python package, kept verbatim as the specification
```

[`CLAUDE.md`](CLAUDE.md) documents the architecture that has to be reproduced exactly, and the
traps that cost the most time. [`TODO.md`](TODO.md) tracks what is done and what is not.

[`.devops/azure-pipelines.yml`](.devops/azure-pipelines.yml) builds the package on every push to
`main` and pushes it to nuget.org with a CalVer version (`yy.M.<build id>`).

---

## Credits

Laya — the model, the method and the original Python implementation — is by
**[Convai Innovations](https://github.com/NandhaKishorM/laya)**. The checkpoints are published on
the [Hugging Face hub](https://huggingface.co/convaiinnovations/laya) under Apache-2.0, and this
repository keeps the Python source verbatim under [`.reference/`](.reference/) as the behavioural
specification it is tested against.

This C# implementation is maintained by [Curiosity](https://curiosity.ai).

## License

Apache-2.0, as the original. The model weights are published by Convai Innovations under the same
license; see [`NOTICE`](NOTICE) for the attributions that travel inside the package.
