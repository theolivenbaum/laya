# Laya

**Multilingual, non-autoregressive System 1 decision engine — for .NET.**

Laya answers typed questions (`choice`, `score`, `noul`) about any state — text, an email, a
ticket, a JSON document — in **one forward pass per question set**. Nothing is generated, so
there is nothing to parse and nothing to hallucinate; every answer comes back as a distribution
with a calibrated confidence. It runs on managed CPU SIMD: no Python, no PyTorch, no native
dependency.

```csharp
using Laya;
using Laya.Runtime;

// The Laya.Model.English package: config and tokenizer ship with it, the weights are
// downloaded once on first use.
using var agent = LayaEnglish.Load();

var result = agent.SystemOne("We were billed twice for March, please refund it.", Presets.Triage());

Console.WriteLine(result["intent"].Choice);        // refund
Console.WriteLine(result["intent"].Confidence);    // normalised entropy of the distribution
```

## Checkpoints

This package is the engine. The checkpoints ship separately — each one embeds its config and
tokenizer and downloads its weights once, on first use, into `~/.cache/laya`
(or `$LAYA_HOME` / `$HF_HOME/laya`):

| package | encoder | params | max tokens | weights |
|---|---|---|---|---|
| `Laya.Model.English` | ModernBERT-large | 421M | 512 | 843 MB |
| `Laya.Model.Multilingual` | mmBERT-base | 322M | 1024 | 644 MB |
| `Laya.Model.TypedDecisions` | ModernBERT-large | 421M | 1024 | 843 MB |

`Agent.Load()` still downloads a checkpoint straight from the Hugging Face hub if you would
rather not take a model package, and `Agent.FromDirectory(path)` loads one from disk.

## Links

* Source, docs and issues: <https://github.com/theolivenbaum/laya>
* Models: <https://huggingface.co/convaiinnovations/laya>

Apache-2.0. (c) Copyright 2026 Curiosity GmbH - all rights reserved. Laya itself is Copyright (c) 2025 Convai Innovations.
See `LICENSE` and `NOTICE` in this package.
