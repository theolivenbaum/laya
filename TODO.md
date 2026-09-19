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

## 6. Tests & polish
- [x] Port `.reference/tests/*` to xunit (criteria rendering, router, e2e)
- [x] README rewrite for the .NET package
- [ ] Re-enable CI as a .NET workflow (left disabled per the port brief)

## Performance work (done, with what is left)
- [x] Concatenated-batch forward pass — the weights are streamed once per call, not once per question
- [x] Panel-packed weights + broadcast GEMM kernel
- [x] Explicit AVX-512 path with `LAYA_VECTOR_BITS` override
- [ ] Cache blocking over the reduction dimension; still ~3.5x behind oneDNN
- [ ] Keep weights in fp16 and widen per tile, to halve the weight bandwidth
- [ ] Reuse buffers across layers instead of allocating per forward pass

## Known gaps / deliberate deviations
- The multilingual checkpoint's `encoder/config.json` says `position_embedding_type: "sans_pos"`.
  transformers' ModernBERT has no such switch and always applies RoPE, so the port ignores the key
  too. Honouring it made layer 0 diverge immediately — see `ModernBertConfig.UsesRope`.
- `net11.0` is multi-targeted but shares the `net10.0` kernels: this port was developed against a
  .NET 10 SDK, and writing intrinsics that could not be compiled or measured here would be a parity
  risk. `SimdOps.Capabilities` reports which target framework is actually running.
- GPU execution is out of scope; this port is CPU/SIMD only.
- Training (`proper_reward`, `td_lambda_targets`) is ported for completeness but there is no
  training loop, matching the Python package.
