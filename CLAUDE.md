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
.reference/            original Python package, tests, notebook, packaging (read-only spec)
src/Laya/              the port: numerics, tokenizer, ModernBERT, decision head, runtime
src/Laya/D1/           LiquidAI's d1 decision model (LFM2.5-VL-3B): prompt, readout, images
src/Laya.Catalyst/     optional statistical language identifier for the router (Catalyst)
src/Laya.Training/     fine-tuning: backward pass, RLCD objective, AdamW, calibration, dataset
src/Laya.Cli/          `laya` command line tool (predict, route, download, dump, bench, train, evaluate)
tests/Laya.Tests/      xunit tests, including parity tests against dumped PyTorch tensors
tools/                 Python helper scripts used only to produce reference dumps
artifacts/             (gitignored) downloaded models and reference dumps
```

## Packaging

Three packages, `Laya`, `Laya.Catalyst` and `Laya.Training`, published on every push to `main` with a CalVer version by the Azure
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
  moves any layer's output beyond the tolerances in `tests/Laya.Tests/ParityTests.cs` is a
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
`ILanguageClassifier` (`Laya.Catalyst` implements it), and its contract is narrow on purpose: it is
consulted only when the heuristic left a Latin-script state undecided *and* would route it English,
and only for at least `MinimumWords` words. It can move a state to multilingual, never an identified
one away from it. The decision reads `1 − P(en)`, not the top language's probability: Catalyst
splits Indonesian between Indonesian, Malay and Tagalog, so the top-1 probability dips under any
threshold while "not English" stays certain — a top-1 rule made the tests flake one run in three.

## Training

`Laya.Training` ports the fine-tuning notebook. Parity rules apply to it as to inference:

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

## d1 (LiquidAI/d1-3B)

A second engine beside the laya checkpoints: `D1Agent : IDecisionEngine`. Ground truth is the
checkpoint's own Python (`modeling_d1.py`, `runner.py`, `hybrid.py`, `lfm2_vl.py`, `prompt.py`,
`api.py`, downloaded with the weights) running on transformers 5.x; `tools/dump_d1_reference.py`
(text) and `tools/dump_d1_vision_reference.py` (pictures) dump it, `D1ParityTests` compares every
layer. Weights come from Hugging Face (LFM Open License v1.0), not the model host.

What has to be reproduced exactly:

1. **Prompt** (`D1Prompt`, from `prompt.py`): `<|startoftext|><|im_start|>user\n{images}{state}\n\n\nQUESTION:\n{question}<|im_end|>\n<|im_start|>assistant\n`.
   A string state is itself; anything else is `json.dumps(state, ensure_ascii=False, indent=2)`.
   Choice options get single-token codes (the labels if all single letters, else `A`..`Z`, else
   `00`..), with a fallback pool when a code is not one token. Readout: each option scores the max
   logit over its forms (`yes/Yes/YES`, the digit, the code and ` code`), softmax over options.
   The log-normalizer cancels, so only the option tokens' logits are computed.
2. **One question** is the whole prompt encoded at once, one causal chain. **Several** are the
   prefix encoded once (the trunk) and each suffix encoded on its own (branches) — not the joint
   encoding split at the common prefix. `Lfm2Tree` is `hybrid.Tree`.
3. **LFM2 layer**: `h = x + op(rms(x)); h = h + w2(silu(w1 rms(h)) * w3 rms(h))`. Conv: `in_proj`
   emits `[B | C | x]`, `y = out(C * conv(B * x))`, depthwise, 3 causal taps that follow the tree's
   parent chain (a branch's first tokens read the trunk's last). Attention: GQA 32/8, per-head q/k
   RMSNorm *before* RoPE, half-split RoPE over the whole 64-dim head, theta 1e6, angles computed in
   fp32 as torch does. Final norm is `embedding_norm`. LM head tied to `embed_tokens`.
4. **Vision**: `cap_pixels` (Pillow bicubic to <= 1 MP) → `Lfm2VlImageProcessor` (smart resize to
   multiples of 32 holding 64..256 tokens, or 512-px tiles 2..10 + thumbnail when "too large") →
   torchvision bicubic antialias on uint8 → `(x - 127.5) / 127.5` → 16-px patches `[py, px, c]` →
   SigLIP2 NaFlex (pos grid 16x16 resized *bilinear antialias* to the patch grid, 27 pre-LN layers,
   tanh-GELU) → pixel unshuffle 2x2 (channels `[(dy*2+dx)*C + c]`) → Linear, GELU(erf), Linear →
   scattered into the `<image>` slots in order. Placeholder text: `<|image_start|>` (+
   `<|img_row_r_col_c|>` per tile, `<|img_thumbnail|>`) + `<image>` x tokens + `<|image_end|>`.

Things that bit:

- **A third of the conv `in_proj` weights are bf16 subnormals**, and some SwiGLU outputs are too.
  Every x86 FMA touching one takes a microcode assist; neither PyTorch nor .NET sets FTZ/DAZ. The
  reference as shipped spends ~80% of its time there (21 s for a 61-token decision on one core vs
  4 s with `torch.set_flush_denormal(True)`). `BFloat16Matrix` flushes subnormal weights at load
  and subnormal activations while packing; parity is unchanged to the printed digit.
- **torchvision's uint8 bicubic is not float interpolation + rounding.** On CPU it is PyTorch's
  native uint8 kernel: double taps quantized to int16 with the largest precision keeping the biggest
  tap under 2^15, width pass then height pass, each rounded to bytes. Pillow is the same shape with
  22-bit taps. Both are bit-exact in `Resampling`; a float version was off by up to 24 levels.
- Picture parity has looser per-layer bounds than text, on purpose: the tower reaches ~1600 and its
  fp32 round-off (~1e-4 relative) grows on the image rows to ~1e-3 of the residual stream.
  `LanguageModelOverReferenceImageFeaturesMatchesPyTorch` feeds PyTorch's own features and holds the
  text bound, so the language model is exact; log-probs and answers keep the tight bounds.
- transformers 5's `Siglip2VisionModel` has no `.vision_model` (the checkpoint keys still do).
- The full model in fp32 (12.5 GB) plus activations is over this container's 13 GB cgroup; the
  vision dump runs the tower first with the language model in bf16, then swaps.

### The bf16 GEMM, measured (4-core Xeon, AVX-512, one thread)

`BFloat16Matrix` keeps weights in bf16 (5.0 GiB for the text stack instead of 10) in 12-column
panels, packs activations into 16-row groups (an AVX-512 16x16 register transpose), and runs a
32x12 register tile (2 activation vectors x 12 broadcast weights, 24 accumulators) over 256-step
reduction blocks widened into L1/L2 scratch, accumulating into a tile-contiguous buffer that is
register-transposed into the output. Results are bit-identical to one fp32 FMA chain per output.

| measurement | GFLOP/s |
|---|---|
| FMA roof, 512-bit, registers only (`--fma-roof`) | 150-160 |
| the 32x12 tile, both operands in L1 (`--tile 64`) | 145 |
| the tile streaming one operand from L2 | 126-138 |
| the whole product, 608 rows (`--bf16-gemm 608`) | ~100-105 |
| the whole product, 61 rows | ~93 |
| MKL fp32 (`tools/bench_gemm.py`), 608 / 61 rows | 126-137 / 75-85 |

Tried and **rejected** (no gain within the VM's ±10% noise): the old 6x64 PackedMatrix layout with
widening (75-88), software prefetch of the weight stream, row-tile-outermost order with the whole
reduction widened (worse), reduction blocks 64-512 and column blocks 48-2048 (flat). A scattered
`[column][row]` accumulator cost ~25%; the tile-contiguous layout fixed it. Measure with
`--bf16-gemm`'s long runs: 40 ms runs on this VM vary by 20%.

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
dotnet run --project src/Laya.Cli -c Release -- profile \
    --model-dir artifacts/models/english --preset triage --text "…" --threads 1 --no-trace

# 2. Which method is on the CPU? (drops --no-trace: in-process sampling via Memory.Introspect)
dotnet run --project src/Laya.Cli -c Release -- profile --model-dir … --preset triage --text "…"

# 3. Steady-state numbers for one kernel
dotnet run --project benchmarks/Laya.Benchmarks -c Release -- --filter '*Gemm*'
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
dotnet build Laya.slnx -c Release
dotnet test  tests/Laya.Tests -c Release           # parity tests skip if no dumps present
dotnet run --project src/Laya.Cli -- download --model english
dotnet run --project src/Laya.Cli -- predict --preset triage --text "I was charged twice"
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
- `ForwardTiming.Stage` is idempotent for a reason: `using var stage = …` plus an explicit
  `stage.Dispose()` recorded the stage twice, the second time spanning to the end of the enclosing
  method. It made one stage look six times more expensive than it was and sent me optimizing the
  wrong thing. Instrumentation gets the same scepticism as the code it measures.
