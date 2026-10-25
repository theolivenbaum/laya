# CLAUDE.md — laya .NET port

## What this repository is

`laya` is a **non-autoregressive System 1 decision engine**: a bidirectional encoder
(ModernBERT) plus a small typed decision head that answers several typed questions about a
piece of state in **one forward pass**, with calibrated probabilities.

This repository is being ported from the original Python/PyTorch implementation to modern
C# (.NET 10 / .NET 11). The Python source is preserved verbatim under **`.reference/`** and is
the authoritative behavioural specification. When the C# and the Python disagree, the Python
is right unless there is a written note here saying otherwise.

## Layout

```
.reference/                             original Python package, tests, notebook, packaging (read-only spec)
src/StructuredDecisions.Base/           the shared base: IDecisionModel, Question/QuestionSet/Answer/DecisionResult,
                                        numerics (fp32 + bf16 GEMM), tokenizers, safetensors, downloads, diagnostics
src/StructuredDecisions.Laya/           Laya: ModernBERT, decision head, Agent, Router, presets, RemoteCheckpoint
src/StructuredDecisions.Laya.Catalyst/  optional statistical language identifier for the router (Catalyst)
src/StructuredDecisions.Laya.Training/  fine-tuning: backward pass, RLCD objective, AdamW, calibration, dataset
src/StructuredDecisions.Mica/           Mica v0.5 4B: Qwen3.5 decoder, request contract, prompt, codebook, MicaJudge
src/StructuredDecisions.Cli/            `laya` command line tool (predict, route, download, dump, bench, train, mica, …)
tests/StructuredDecisions.Tests/        xunit tests, including parity tests against dumped PyTorch tensors
benchmarks/StructuredDecisions.Benchmarks/  BenchmarkDotNet suites
tools/                                  Python helper scripts used only to produce reference dumps
artifacts/                              (gitignored) downloaded models and reference dumps
```

Namespaces follow the projects: `StructuredDecisions` (+ `.Numerics`, `.Tokenizers`, `.Io`, `.Diagnostics`) in
the base, `StructuredDecisions.Laya[.Models|.Runtime|.Io|.Catalyst|.Training]`, `StructuredDecisions.Mica[.Models]`.
**The base references no model family**, and the model families never reference each other: anything two of
them share moves down into the base. `IDecisionModel` is the contract every family implements; a question
option a family cannot honour (`Multi`, `AllowCannotGrade`, `AbstainThreshold` on Laya) throws
`NotSupportedException` instead of being ignored. Environment variables keep their `LAYA_` names
(`LAYA_THREADS`, `LAYA_VECTOR_BITS`, `LAYA_HOME`, `LAYA_MODEL_BASE_URL`) and apply to every family.

## Packaging

Five packages — `StructuredDecisions.Base`, `StructuredDecisions.Laya`, `StructuredDecisions.Laya.Catalyst`,
`StructuredDecisions.Laya.Training` and `StructuredDecisions.Mica` (formerly `Laya`, `Laya.Catalyst`, `Laya.Training`) —
published on every push to `main` with a CalVer version by the Azure
DevOps pipeline in `.devops/azure-pipelines.yml` (`GeneratePackageOnBuild`, then
`NuGetCommand@2` push through the `nuget-curiosity-org` service connection). **No model data is packaged.** `RemoteCheckpoint` downloads a
checkpoint's five files on demand from `https://models.curiosity.ai/laya/`, laid out as
`<checkpoint>/<revision>/<file>` — English at the root, the other two under `multilingual/` and
`typed-decisions/`, revision `main` — into `<cache>/models.curiosity.ai/<name>/<revision>/`, which
is the layout `Agent.FromDirectory` expects. `LAYA_MODEL_BASE_URL` overrides the host; the
Hugging Face path (`Agent.Load`, `laya download --source hub`) stays as it was.

## Ground truth: the model

Three checkpoints, published twice: bundled in `convaiinnovations/laya`, and one repo each
(`convaiinnovations/laya-multilingual`, `convaiinnovations/laya-typed-decisions`). In the bundle
English is at the root and the other two are subfolders; in a standalone repo the same five files
are at the root, so `HuggingFaceDownloader.CheckpointFilter(null)` serves both. The bundle layout:

| name            | subfolder          | encoder                    | params | max_len |
|-----------------|--------------------|----------------------------|--------|---------|
| english         | *(repo root)*      | `answerdotai/ModernBERT-large` | 421M | 512  |
| multilingual    | `multilingual`     | `jhu-clsp/mmBERT-base`     | 322M   | 1024    |
| typed-decisions | `typed-decisions`  | `answerdotai/ModernBERT-large` | 421M | 1024 |

Each checkpoint directory holds exactly five files: `rl_agent_config.json`, `encoder/config.json`,
`model.safetensors` (fp16 weights, three fp32 scalars for `temperature`), and
`tokenizer/tokenizer.json` + `tokenizer/tokenizer_config.json`. The root-level names are prefixes
of the subfolder copies, so the download filter must match them **exactly** — a `StartsWith` there
pulls the whole 2.3 GB bundle.

### Architecture that must be reproduced exactly

1. **ModernBERT encoder** — pre-norm, no biases anywhere except the attention/MLP
   projections the config enables (`attention_bias: false`, `mlp_bias: false`,
   `norm_bias: false`), RoPE, GeGLU MLP (`Wi` produces `2 * intermediate_size` and is split
   into gate/input halves), and **alternating attention**: every third layer
   (`global_attn_every_n_layers: 3`, i.e. layers 0, 3, 6, …) is full attention with
   `rope_theta = 160000`; all others are sliding-window attention with a window of
   `local_attention / 2 = 64` tokens on each side and `rope_theta = 10000`.
   Layer 0 has **no** `attn_norm` (it is `nn.Identity` in HF); every other layer does.
2. **Typed decision head** — two `nn.TransformerEncoderLayer`s with `norm_first=True`,
   `batch_first=True`, `nhead = hidden // 64`, `dim_feedforward = 4 * hidden`, and PyTorch's
   **default `relu` activation** (not gelu). Note that `laya` runs the layers directly
   (`for layer in self.head.layers`) so the `nn.TransformerEncoder`'s final norm is *not*
   applied — and there is none, since `norm=None`.
3. **`type_emb`** — a 3-entry embedding added to every position before the head.
4. **`scorer`** — `LayerNorm → Linear(d,d) → GELU → Linear(d,1)`, read at the `[MASK]`
   marker position of each option; masked options get `-1e4`.
5. **`act_head`** — `Linear(d+4, 256) → GELU → Linear(256, n_act)` over `h[:, 0]`
   concatenated with four calibration features (top-1 prob, top1−top2 margin, normalised
   entropy, `k / 255`). The features are computed from the **detached, untempered** softmax.

### Sequence construction (`build_sequence`)

`[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`,
with the option-text budget logic in `.reference/laya/common.py`. Option token ids are
truncated to 48 each; the head (instructions) is truncated to fit `head_max_len`.

### Tokenizer

Byte-level BPE (`tokenizer.json`): NFC normalizer, GPT-2 regex pre-tokenizer
(`ByteLevel`, `add_prefix_space: false`, `use_regex: true`), no unk token, and an
added-token vocabulary that must be matched **before** BPE is applied. `laya` always
calls the tokenizer with `add_special_tokens=False`, so the `TemplateProcessing`
post-processor never runs; special ids come from `tokenizer_config.json`.

## Porting rules

- **Parity first, speed second.** Every layer has a dump-and-compare path
  (`laya dump-states`, `tools/dump_reference.py`). A change that improves throughput but
  moves any layer's output beyond the tolerances in `tests/StructuredDecisions.Tests/ParityTests.cs` is a
  regression.
- **Compute in fp32.** The checkpoints are stored fp16; weights are converted once at load.
  PyTorch on CPU also runs this model in fp32, so fp32 is what parity is measured against.
- **SIMD via `System.Numerics.Vector<T>` and explicit `Vector512`**, following the patterns in the
  TensorSharp repository (`TensorSharp.Models/Embeddings/*`,
  `TensorSharp.Runtime/SafetensorsReader.cs`): span-based kernels, FMA where available, panel-packed
  weights, no allocation in the inner loop. Note that `Vector<T>` stays 256-bit on AVX-512 hardware
  unless the process opts in, so `PackedMatrix` reaches for `Vector512` directly and
  `SimdOps.UseVector512` decides (override with `LAYA_VECTOR_BITS`).
- **No unsafe pointer code unless a kernel measurably needs it**, and keep it inside
  `Laya.Numerics`.
- Public API names mirror the Python: `Agent`, `Router`, `RouteDecision`, `Presets`,
  `LanguageDetector`, `EmailUtils`. Method names are PascalCase (`SystemOne`, `Predict`).

## Routing: the heuristic, and the classifier behind it

`LanguageDetector` follows upstream `lang.py` exactly; `Router` with no `LanguageClassifier` routes
exactly as the Python does, and `LatinRoutingTests` pins it. The one C#-only addition is
`ILanguageClassifier` (`StructuredDecisions.Laya.Catalyst` implements it), and its contract is narrow on purpose: it is
consulted only when the heuristic left a Latin-script state undecided *and* would route it English,
and only for at least `MinimumWords` words. It can move a state to multilingual, never an identified
one away from it. The decision reads `1 − P(en)`, not the top language's probability: Catalyst
splits Indonesian between Indonesian, Malay and Tagalog, so the top-1 probability dips under any
threshold while "not English" stays certain — a top-1 rule made the tests flake one run in three.

## Training

`StructuredDecisions.Laya.Training` ports the fine-tuning notebook. Parity rules apply to it as to inference:

- **Gradients are checked against autograd, not reasoned about.** `tools/dump_training_reference.py`
  builds the reference `DecisionModel` on a tiny random ModernBERT (window shorter than the
  sequences, both head layers) and dumps every gradient; with `--checkpoint` it does a tensor subset
  of a real checkpoint. `TrainingGradientTests` compares them. Any change to a backward pass has to
  keep both at their current ~1e-6 / ~1e-5 relative error.
- **Dropout masks are a function of (seed, index)** (`Ops.Uniform`), because layers are
  checkpointed: the backward pass recomputes the forward, and a stateful RNG would draw different
  masks the second time. PyTorch draws its own, so the dropout path is checked by finite
  differences instead.
- **`LinearBackward` overwrites `dx`; every other backward adds into it.** The residual streams
  depend on that convention.
- **The notebook's quirks that are kept:** in-sample calibration, the action head
  receiving zero gradient but weight decay, and the item/target construction of `build_training_item`
  (pinned token for token by `TrainingItemParityTests`). **The ones that are not:** the calibration set (`all_items[::15]` over five questions per case
samples one question position, so only `choice` was ever fitted — a seeded random sample of the
same size is used), stale
  `temperature_by_options` buckets surviving the fit (they are refitted and replaced), and fitted
  temperatures outside the clamp range (they are clamped to what inference applies).

## Mica (a second model family: Qwen3.5-4B)

`StructuredDecisions.Mica` runs [Mica v0.5 4B](https://huggingface.co/sky7350/Mica-v0.5-4B), a Qwen3.5-4B fine-tune that
answers Mica's `/v1/systemone` requests. It is not an encoder + head: it is a causal decoder whose answer is
the softmax over the **label tokens' logits at the last prompt position** (`No`/`Yes`, or `A`…`Z`, `a`…`z`,
then two-letter labels; `codebook_255.json`), temperature 1. The behavioural spec is Mica's runtime in
github.com/akivet/Mica — `mica/contract.py`, `mica/native.py`, `mica/codebook.py`, `mica/calibration.py`
(the `v0.5/serving/lib` HF runtime is obfuscated but renders the same prompt; it escapes every added token
of the tokenizer, which is what `MicaPrompt` does). Long-input mode (`llama_set_upper_stale`) is an
approximation in their server and is deliberately not ported; this is their exact path.

The decoder (`Qwen35Model`, from transformers' `modeling_qwen3_5.py`):

1. 32 pre-norm layers, `layer_types` from the config: three Gated DeltaNet layers, then one full-attention
   layer, repeated. Every layer ends in a SwiGLU MLP. Vision tower and `mtp.*` are never loaded.
2. **RMSNorm is zero-centred**: `x * rsqrt(mean(x²) + eps) * (1 + w)` — for the input/post-attention norms,
   the final norm and q/k norms. The DeltaNet's output norm (`RMSNormGated`) is the plain `w * norm(x) * silu(z)`.
3. Full attention: `q_proj` emits `[q | gate]` **per head** (heads × 2 × 256), q/k RMSNorm per head, RoPE
   (`rotate_half`) on only the first 64 of 256 dims (`partial_rotary_factor 0.25`, θ = 1e7; mRoPE collapses to
   plain RoPE for text since all three position streams are equal), GQA 16/4, causal, then `· sigmoid(gate)`.
4. Gated DeltaNet: `in_proj_qkv` → causal depthwise conv (kernel 4, no bias) → SiLU; q, k L2-normalised
   (eps 1e-6), q scaled by `dk^-0.5`; value head `h` reads key head `h / 2` (`repeat_interleave`);
   `g = −exp(A_log)·softplus(a + dt_bias)`, `β = sigmoid(b)`; state `S ← S·e^g; S += k ⊗ β(v − Sᵀk); o = Sᵀq`.
   transformers runs the chunked form for a prefill; the recurrence here is algebraically identical.
5. Tied LM head; only the label rows are ever multiplied.

Rules that apply on top of the porting rules above:

- **Weights stay bf16** (`PackedMatrixBf16`, ~7.8 GiB) and are widened inside the GEMM — exact, so compute
  is still fp32. 4.2B parameters as fp32 do not fit in the 15 GB this was developed on.
- **Parity is judged against fp64, not just torch fp32.** `tools/dump_mica_reference.py tiny` writes a tiny
  random Qwen3.5 (committed, `Fixtures/qwen35-tiny`) with an fp32 and an fp64 run; `Qwen35ParityTests`
  requires every layer to be within 2× of torch fp32's own distance from fp64. For the real checkpoint,
  `dump_mica_reference.py checkpoint` runs transformers' `Qwen3_5DecoderLayer` one layer at a time in fp32
  into `artifacts/dumps/<name>/`, and `MicaCheckpointParityTests` compares every layer: currently ≤ 1.4e-5
  relative per layer and ≤ 5e-6 absolute on the probabilities.
- **A state read in pieces is bit-identical to reading it at once** (`ReadingInPiecesIsExactlyReadingAtOnce`).
  `MicaJudge` relies on it: rows share their longest common token prefix, read once, and each row rewinds
  to it (`Qwen35State.Save/Rewind` copies only the small DeltaNet state; the KV cache is overwritten).
- The contract fixtures (`Fixtures/mica-contract-cases.json`) are Mica's own; `MicaContractTests` checks rows,
  every rendered prompt byte for byte, typed results and the invalid-request list against them.

### Mica performance

Measured with `laya mica-bench` against `tools/bench_mica_python.py` (transformers on the same CPU, same token
ids, 4 threads; shut down build servers first — a lingering `MSBuild`/`VBCSCompiler` swung both sides by 15%).
On the 4-core Xeon: a 219-token prompt reads in 4.3 s (PyTorch fp32, one layer resident at a time, 5.4–6.3 s;
bf16, what Mica's own HF runtime uses, 16.4 s); a three-question request in 5.2 s against 15.5 s for one PyTorch
forward per question. The GEMMs are ~92% of a pass and run at ~120 GFLOP/s per core, ~82% of the FMA roof.

What made the difference, all exact:

- **Fresh state memory costs page faults**: 48 MB of DeltaNet state per `Qwen35State` took 380 ms to first-touch,
  as long as the whole recurrence. States, snapshots and KV caches come from `StateBufferPool`; dispose them.
- **Subnormal decays**: heads with `exp(A_log)` up to 75 give `exp(g) < 1.2e-38`; multiplying the state by it took
  a microcode assist per element and those heads serialised the recurrence. They are flushed to 0 (the product is
  absorbed by `k ⊗ δ` anyway). One thread and four were equally fast before — that is how it showed.
- **The fused delta-rule kernel** (`DecoderKernels.GatedDeltaRule128`): one sweep of the state per token with δ, o
  and m in zmm registers; bit-identical to the two-sweep form (`DeltaRuleTests`).
- **Weights stream once per pass**, ~0.33 s for 8 GB whatever the row count. So questions are batched into one pass
  (`Forward(IReadOnlyList<Qwen35Sequence>)`), cache snapshots are taken as in-pass checkpoints rather than by
  splitting the read, and the header is read at load (`MicaJudge.Warm`, Mica's `start_warm`).
- K-blocking the bf16 GEMM at 4096 (the 9216-wide down projection's panel no longer fits L2), padding row counts to
  the 6-row tile, splitting rows when a projection has few panels, last-layer pruning.

Measured and **rejected**:

| idea | result |
|---|---|
| Widen each 64×64 bf16 weight block once into an L1 fp32 buffer, then a pure-FMA loop | 102–119 GFLOP/s vs 120–126 streaming: the inline widening was not the limit |
| fp32 weights instead of bf16 (where memory allows) | 84 vs 122 GFLOP/s per core: the bf16 kernel is faster, it halves L2 traffic |
| Flushing decays below 1e-10 (not just subnormals) | no change: tiny normal decays were never the problem, and it would not be exact |
| Padding the 9216-float activation stride off a 4 KB multiple | no change |
| int8 weights (Mica's own SmoothQuant W8A8 `int8/model-int8.safetensors`), `pmaddubsw`+`pmaddwd` zmm (sentence-transformers-sharp's `QGemm`) | 52 G MAC/s per core vs the bf16 kernel's 60, and `pmaddubsw` saturates for weights near ±127 (their kernel clamps to ±64 + CSR outliers) |
| int8 via `AvxVnni` `vpdpbusd` ymm (all .NET 10 exposes) | 56–60 G MAC/s: no faster than bf16 |
| int8 via `AvxVnni.V512` `vpdpbusd` zmm (.NET 11 only) | register roof 72 G MAC/s, 6×4 tile 73: one instruction per ~2 cycles (single port) against two FMA ports. At most ~1.2× on the GEMMs before activation quantization, inexact, preview SDK — not worth it. This CPU *does* have AMX (`amx_int8`, `amx_bf16`) and AVX512-BF16, but .NET 10/11 expose neither (`Avx10v1/v2` report unsupported); see below |

### What upstream's quantizations cost

TensorSharp runs this family from GGUF (Q8_0, Unsloth's Q4_K_XL/Q4_K_M, IQ2_XXS, NVFP4); Mica publishes
`mica-v0.5-4b-Q8_0.gguf` and `mica-v0.5-4b-IQ4_XS.gguf` (IQ4_XS matrices, Q5_K `in_proj_qkv`/`v_proj`, Q6_K
embedding). `tools/measure_mica_quant.py` runs the 105 fixture prompts (52k tokens) layer by layer in fp32 with
each file's matrices dequantized (`-w`), and also with ggml's CPU activation quantization (Q8_0 / Q8_K) on every
matmul input (`-wa`, what llama.cpp computes; its f16 KV cache is not emulated). Against the exact path:

| variant | mean \|Δp\| | p90 | worst | mean KL | same top-1 |
|---|---|---|---|---|---|
| Q8_0 weights | 0.0035 | 0.010 | 0.029 | 1.6e-4 | 100% |
| Q8_0 as ggml runs it | 0.0060 | 0.018 | 0.060 | 4.2e-4 | 100% |
| IQ4_XS weights | 0.037 | 0.113 | 0.413 | 1.7e-2 | 96.2% |
| IQ4_XS as ggml runs it | 0.040 | 0.152 | 0.399 | 1.8e-2 | 96.2% |

Q8_0's error is ~1000× the C# port's own (5e-6) but never changes an answer. IQ4_XS flips 4 of 105 answers (all
near-ties, ≤ 0.58 confident) and flattens confident ones (0.81 → 0.58 on `multi_26`), which matters because the
probabilities are the calibrated output. The weight error dominates: activation quantization adds little.
llama.cpp's converter reorders the DeltaNet value heads (tiled vs `repeat_interleave`) and bakes `1 + w` into the
norms; the tool matches heads back block by block and checks every matrix's error (Q8_0 0.54%, IQ4_XS ~7.8%).

Speed and memory (`llama-bench -p 219 -n 0 -lm none`, llama.cpp `2280f0f` built `GGML_NATIVE`, against `laya mica-bench`
on the same 219-token prompt, same day; peak RSS sampled from `/proc`):

| runtime / format | 4 threads | 1 thread | vs C# bf16 | peak RSS |
|---|---|---|---|---|
| **C# bf16 (this port, exact)** | 44 tok/s (4.96 s) | 14 tok/s | 1.00× | 7.83 GiB weights (10.6 GiB peak while loading) |
| llama.cpp BF16 | 31 | 8.4 | 0.71× | 8.2 GiB |
| llama.cpp Q8_0, no repack (AVX-512 path) | 26 | 8.6 | 0.60× | 4.4 GiB |
| llama.cpp Q8_0, AMX repack (default) | **76** | 22.5 | **1.73×** | 5.6 GiB |
| llama.cpp IQ4_XS, no repack | 65 | 21.6 | 1.47× | 2.6 GiB |
| llama.cpp IQ4_XS, AMX repack (default) | 62 | 19.2 | 1.41× | 3.1 GiB |

The quantized speed-up over this port comes from AMX tile instructions (Q8_0) and llama.cpp's IQ4_XS × Q8_K
VNNI kernel. .NET has no AMX intrinsics, so the Q8_0 number is out of reach without native code; IQ4_XS's
non-AMX 1-thread rate (~77 G MAC/s) is at the measured `vpdpbusd` zmm roof, so a .NET 11 VNNI kernel could
approach it, at the precision cost in the table above.

Checkpoint files: hub `sky7350/Mica-v0.5-4B` pinned to `MicaCheckpoint.HubRevision` (config, two bf16 shards +
index, tokenizer; the repo's GGUF/INT8 files are excluded by an exact-name filter). `laya mica-convert` writes
the host layout — `config.json`, `calibration.json`, `tokenizer.json`, `tokenizer_config.json` and a single
text-only `model.safetensors` (bf16 bytes copied unchanged) — for `https://models.curiosity.ai/laya/mica/main/`.

## Target frameworks

`net10.0;net11.0`. The `net11.0` target is only added when an 11.x SDK is actually installed (see
`Directory.Build.props`), so the repository still builds on a .NET 10 SDK; force it either way with
`-p:LayaEnableNet11=true|false`. The two targets currently share their kernels — this port was
developed against a .NET 10 SDK, and unverifiable intrinsics behind `#if NET11_0_OR_GREATER` would
be a parity risk. `SimdOps.Capabilities` reports the running target framework, so a net11-only path
has somewhere obvious to land.

## Measuring before changing

Performance work on this port has one rule: measure first, and measure the thing you are about to
change. Three tools, in the order they are usually useful:

```bash
# 1. Where does the time go, and what is allocated? Single-threaded by default.
dotnet run --project src/StructuredDecisions.Cli -c Release -- profile \
    --model-dir artifacts/models/english --preset triage --text "…" --threads 1 --no-trace

# 2. Which method is on the CPU? (drops --no-trace: in-process sampling via Memory.Introspect)
dotnet run --project src/StructuredDecisions.Cli -c Release -- profile --model-dir … --preset triage --text "…"

# 3. Mica: the forward pass and whole requests against the Python reference
dotnet run --project src/StructuredDecisions.Cli -c Release -- mica-bench --model-dir artifacts/models/mica --stages
dotnet run --project src/StructuredDecisions.Cli -c Release -- mica-bench --model-dir artifacts/models/mica \
    --requests artifacts/dumps/gate-request.json artifacts/dumps/rag-request.json

# 4. Steady-state numbers for one kernel
dotnet run --project benchmarks/StructuredDecisions.Benchmarks -c Release -- --filter '*Gemm*'
```

`LAYA_THREADS=1` (or `--threads 1`) pins the kernels to one thread (per call, pass a `ParallelOptions`
to `SystemOne` / `Forward` - every kernel resolves it through `LayaRuntime.Resolve`); always tune single-threaded
first, because a parallel measurement hides a kernel problem behind memory bandwidth.
`LAYA_VECTOR_BITS=256|512` picks the GEMM kernel.

When a kernel is slower than its instruction mix says it should be, read the disassembly before
theorising:

```bash
DOTNET_JitDisasm="Panel512" DOTNET_TieredCompilation=0 dotnet run -c Release …
```

### Know the roof before chasing a number

"X GFLOP/s" means nothing without the machine's actual ceiling, and the nominal clock is not it.
On the 4-core Xeon this port was tuned on, a loop of pure register-to-register 512-bit FMAs
sustains **151.7 GFLOP/s** — the core drops to 2.37 GHz under AVX-512, while the 256-bit roof is
89.5 GFLOP/s at the full 2.8 GHz. Measure that first (a dozen independent FMA chains, no memory),
then measure the kernel's own instruction mix over an L1-resident buffer. The three numbers —
roof, mix-in-isolation, real kernel — tell you whether you are fighting the instruction mix, the
cache, or nothing at all.

The GEMM currently sits at 68% of the roof, with the mix reaching ~130 in isolation. These were
measured and **rejected**, so do not re-derive them from first principles:

| idea | result |
|---|---|
| Pad the activation row stride to break L1 set aliasing | +5% — not the problem |
| Block the rows so the activations fit L2 | Worse (94 → 80 → 55 as the block shrinks): it evicts the weight panel |
| Wider panels (8 vectors) to halve activation re-reads | Much worse (52–67): register pressure |
| Group panels so a row tile serves several | Worse (76 at group 4): the group's weights thrash L2 |

One panel resident in L2, streamed against every row, is the structure that wins.

## Build & test

```bash
dotnet build StructuredDecisions.slnx -c Release
dotnet test  tests/StructuredDecisions.Tests -c Release           # parity tests skip if no dumps present
dotnet run --project src/StructuredDecisions.Cli -- download --model english
dotnet run --project src/StructuredDecisions.Cli -- predict --preset triage --text "I was charged twice"
```

Model downloads and dumps land in `artifacts/` (gitignored). Nothing in CI downloads a
model; parity tests are skipped when `artifacts/` is empty.

## Things that have bitten us

- `nn.TransformerEncoderLayer` defaults to **relu**, and `norm_first=True` means
  `x + attn(norm(x))`, `x + ff(norm(x))` with no trailing norm.
- ModernBERT's first layer uses `nn.Identity` for `attn_norm`; applying a LayerNorm there
  silently shifts every downstream layer.
- The sliding window is `local_attention // 2` on **each** side (inclusive), i.e. a token
  attends to `[i-64, i+64]`.
- `logits.masked_fill(~marker_mask, -1e4)` uses `-1e4`, not `-inf`; it changes the softmax.
- Temperature is looked up by `"<type>:<bucket>"` (`2`, `3-5`, `6-10`, `11+`) first, then
  falls back to `temperature[qtype]`.
- The multilingual checkpoint's `position_embedding_type: "sans_pos"` is **not** honoured by
  transformers — its ModernBERT reads no such key and always applies RoPE. Honouring it diverges at
  layer 0.
- `tokenizers` matches added tokens in two passes: `normalized: false` entries against the raw
  text, `normalized: true` entries against the normalized text. The multilingual vocabulary's
  `▁▁` runs are non-normalized, so they must never match the spaces the `Replace` normalizer would
  turn into them. Getting this wrong silently changes 10% of tokenizations.
- Added tokens carry literal text and must bypass the byte-level decoder: a space is not in the
  byte-level alphabet, so decoding a whitespace-run token as a model token drops it.
- Transposing the weights only pays with a *panel* layout. A plain `[in, out]` transpose makes the
  inner loop stride across the whole output dimension and measured 3x slower than the naive
  dot-product kernel.
- **Never pass an accumulator vector by value to a helper inside a hot loop.** `PackedMatrix` used
  to hand its eight `Vector512<float>` accumulators to a small `Store` method. The JIT declined to
  inline it (a 64-byte struct argument goes to the stack), the accumulators became address
  exposed, and *every FMA in the inner loop* was followed by a 64-byte spill and a reload on the
  next iteration. The kernel ran at a third of its speed and the instruction mix looked perfectly
  reasonable in the source. Seeding the accumulators with the bias and storing inline fixed it.
  When a kernel is inexplicably slow, `DOTNET_JitDisasm` first.
- Scalar `Math.Exp` per element is not an option in anything the encoder runs. `Gelu` computes an
  `erf`, and a scalar implementation made one elementwise operation 14% of the whole forward pass;
  the vectorized `SimdOps.Exp` brought that stage down 7.6x.
- Rent scratch from `ArrayPool`, do not allocate it. The attention units allocated their rotated
  query and key per (segment, head, layer): 144 MiB of garbage per call, versus 1.8 MiB now.
- A benchmark of an in-place kernel must restore its input **per invocation**, not per iteration.
  Repeated in-place `Gelu` walks its values down to denormals, and denormal arithmetic traps to
  microcode — the benchmark then reports a kernel forty times slower than it is.
- The register tile is swept, not reasoned about. Rows × output-vectors on the 404×1024×5248
  projection: 6×2 = 84 GFLOP/s, 4×4 = 84, 5×4 = 92, **6×4 = 101**, 7×4 = 98, 2×8 = 52. And give
  the broadcast **one** reused temp: a temp per row pushed the live set to 34 of 32 zmm registers
  and the JIT emitted 28 `vmovaps` per reduction step, which are real uops here because 512-bit
  register moves are not move-eliminated on this microarchitecture.
- Attention's inner loops must not end in a horizontal reduction. Gather the keys **transposed** so
  adjacent keys occupy adjacent lanes; a score vector then finishes with a store rather than a
  shuffle chain. Worth 1.6x on the encoder's attention.
- **`InvariantGlobalization` makes `string.Normalize` a no-op outside ASCII.** NFC therefore went
  silently missing: `e` + U+0301 reached BPE as two code points and the ids differed from `tokenizers`.
  `UnicodeNfc` is a managed NFC over tables generated by `tools/gen_unicode_nfc.py`; never go back to
  `string.Normalize` in a tokenizer.
- .NET's regex works on UTF-16 code units, so `\p{L}` never matches a letter outside the BMP. The Qwen3.5
  pre-tokenizer regex is matched by a hand-written scanner over runes (`Qwen35Pattern`); any new
  `Split` pattern that matters needs the same treatment, checked against `tokenizers` on astral text.
- `ForwardTiming.Stage` is idempotent for a reason: `using var stage = …` plus an explicit
  `stage.Dispose()` recorded the stage twice, the second time spanning to the end of the enclosing
  method. It made one stage look six times more expensive than it was and sent me optimizing the
  wrong thing. Instrumentation gets the same scepticism as the code it measures.
