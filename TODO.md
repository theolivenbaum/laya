# TODO — laya .NET port

Status key: `[ ]` not started · `[~]` in progress · `[x]` done

## 0. Housekeeping
- [x] Move the Python implementation to `.reference/`
- [x] Disable GitHub Actions workflows (`*.yml` → `*.yml.disabled`)
- [x] Write `CLAUDE.md` (architecture + porting rules)
- [x] Write this `TODO.md`
- [x] Solution scaffolding, `Directory.Build.props`, `net10.0;net11.0` multi-targeting
- [x] `.gitignore` for .NET + `artifacts/`

## 1. Foundations
- [x] `Laya.Numerics` — SIMD kernels: dot, GEMM (row-major × transposed weights),
      LayerNorm, GELU (tanh + erf), softmax, RoPE, add/scale, argmax
- [x] Safetensors reader (mmap-backed, fp16/bf16/fp32 → fp32 views, lazy)
- [x] HuggingFace downloader (`resolve/main` + `api/models/.../tree`, resume, sha check,
      `HF_TOKEN`, per-subfolder `allow_patterns` equivalent)
- [x] `tokenizer.json` reader: byte-level BPE, NFC, GPT-2 regex pre-tokenizer,
      added-token trie, byte encoder/decoder tables

## 2. Model
- [x] `ModernBertConfig` / `LayaConfig` bindings for both JSON files
- [x] Embeddings (tok_embeddings + norm)
- [x] RoPE cache per attention type (theta 160000 global / 10000 local)
- [x] Attention: fused QKV, global + sliding-window (`local_attention/2` each side)
- [x] GeGLU MLP (`Wi` split into gate/input, `Wo`)
- [x] Encoder layer stack (layer 0 attn_norm = identity) + final norm
- [x] Decision head: 2 × pre-norm TransformerEncoderLayer (relu, nhead = d/64)
- [x] `type_emb`, `scorer`, `act_head`, marker gather, `-1e4` masking
- [x] Calibration features + action softmax

## 3. Runtime / API surface
- [x] `BuildSequence` (option rendering, budgets, truncation) — port of `common.py`
- [x] `Agent` (`SystemOne` / `Predict`), temperature buckets, confidence
- [x] `Presets` (triage, email, guard, moderation, router)
- [x] `LanguageDetector` (script ranges, stopword scoring, `Analyse`)
- [x] `Router` (+ `RouteDecision`, LRU, aliases, typed-decisions workflow match)
- [x] `EmailUtils` (`CleanEmailBody`, `EmailState`)
- [x] Scoring rules + calibration helpers (`ProperReward`, `TdLambdaTargets`,
      `EceScore`, `ConfidenceFromProbs`)

## 4. Parity
- [x] `tools/dump_reference.py` — dumps tokenization, per-layer hidden states, head
      outputs, logits and final answers from the PyTorch model
- [x] `laya dump-states` — the same dumps from the C# implementation
- [x] Parity test suite comparing both (max abs / rel error per tensor)
- [x] Tokenizer parity over a corpus of tricky strings (unicode, emoji, whitespace runs)
- [x] End-to-end answer parity on the preset question sets

## 5. CLI
- [x] `download`, `predict`, `route`, `presets`, `dump-states`, `bench`, `tokenize`
- [x] `--standalone` / `--repo` / `--subfolder`, so the per-checkpoint repos are reachable too

## 6. Tests & polish
- [x] Port `.reference/tests/*` to xunit (criteria rendering, router, e2e)
- [x] README rewrite for the .NET package
- [ ] Re-enable CI as a .NET workflow (left disabled per the port brief)

## 7. Upstream sync (NandhaKishorM/laya 0.3.6)
- [x] `.reference/` refreshed to upstream `c752770`
- [x] Latin routing: Armenian, unaccented Romance words, Romanian, shared-word rule, undecided ≠ English
- [x] Router lifecycle lock (#95), temperature clamp (#35), email disclaimer (#94), shortlist (#106)
- [x] `ILanguageClassifier` + `Laya.Catalyst` for the plain-ASCII languages the heuristic cannot see

## 8. Training (`Laya.Training`)
- [x] Backward pass for every layer, checkpointed; gradients match PyTorch autograd (tiny model and
      the real English checkpoint)
- [x] RLCD objective, AdamW, cosine schedule, clipping, temperature calibration (types and buckets)
- [x] typed-decisions loader + datasets-server download; notebook evaluation metrics
- [x] `laya dataset` / `train` / `evaluate`
- [ ] Reuse activation buffers across layers: every layer allocates its tape afresh, and page-faulting
      those arrays is a third of the process's CPU time (`sys` ≈ 0.5 × `user` on a full fine-tune)
- [ ] Save optimizer state with the rolling checkpoint so a run can resume (the notebook cannot either)
- [ ] Data parallelism across processes, the notebook's DDP

## Performance work
- [x] Concatenated-batch forward pass — the weights are streamed once per call, not once per question
- [x] Panel-packed weights + broadcast GEMM kernel
- [x] Explicit AVX-512 path with `LAYA_VECTOR_BITS` override
- [x] `laya profile`: per-stage timings, allocation totals, in-process CPU sampling and allocation
      reports via `Memory.Introspect`
- [x] `benchmarks/Laya.Benchmarks`: BenchmarkDotNet suites for the GEMM shapes, the elementwise
      kernels and the whole forward pass, single- and multi-threaded
- [x] `LayaRuntime.MaxDegreeOfParallelism` / `LAYA_THREADS`, so single-thread work is measurable
- [x] Fix the accumulator spill in the GEMM inner loop (3x, single-threaded)
- [x] Vectorized `Gelu`/`Erf`/`Exp` (7.6x on that stage)
- [x] Pooled scratch buffers: 144 MiB -> 1.8 MiB allocated per call, zero GC collections
- [x] Tune the register block: 6 token rows per pass, measured against 4/8/10/12
- [x] Release the safetensors mapping after load (-0.8 GiB working set)
- [x] Widen the register tile to 6 rows x 4 output vectors (84 -> 101 GFLOP/s), with the broadcast
      on a single reused temp
- [x] Attention: keys transposed into SIMD lanes, so neither inner loop ends in a horizontal
      reduction (encoder attention 1.6x; head attention gathered contiguously as well)
- [x] Measure the machine's actual FMA roof rather than assuming the nominal clock
- [ ] fp16 or bf16 weight storage to halve the 1.57 GiB resident — needs a vectorized widening
      path, and bf16 alone would breach the parity budget
- [ ] Parallel scaling is 2.8x on 4 cores against PyTorch's 3.3x; the pass moves ~8.6 GB of
      activation re-reads and looks bandwidth-bound at 4 threads

### Measured and rejected (do not re-derive)

| idea | result |
|---|---|
| Pad the activation row stride against L1 set aliasing | +5% |
| Row cache blocking | Worse — evicts the weight panel |
| 8-vector panels | Much worse — register pressure |
| Panel grouping to cut activation re-reads | Worse — the group's weights thrash L2 |

The GEMM is at 68% of this machine's 151.7 GFLOP/s 512-bit FMA roof, and the same instruction mix
in isolation reaches ~130. Even a perfect GEMM would leave the forward pass at ~2.7 s, so 2x
against PyTorch is not reachable in fp32 on this hardware.

## Known gaps / deliberate deviations
- The multilingual checkpoint's `encoder/config.json` says `position_embedding_type: "sans_pos"`.
  transformers' ModernBERT has no such switch and always applies RoPE, so the port ignores the key
  too. Honouring it made layer 0 diverge immediately — see `ModernBertConfig.UsesRope`.
- `net11.0` is multi-targeted but shares the `net10.0` kernels: this port was developed against a
  .NET 10 SDK, and writing intrinsics that could not be compiled or measured here would be a parity
  risk. `SimdOps.Capabilities` reports which target framework is actually running.
- GPU execution is out of scope; this port is CPU/SIMD only.
- Training lives in `Laya.Training`, a separate package, as the notebook lives outside the Python
  package. It runs fp32 on the CPU where the notebook ran fp16 autocast on two GPUs.
