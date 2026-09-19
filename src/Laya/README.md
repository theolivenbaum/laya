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

// Downloads the English checkpoint on first use (~843 MB) into ~/.cache/laya.
using var agent = RemoteCheckpoint.English.Load();

var result = agent.SystemOne("We were billed twice for March, please refund it.", Presets.Triage());

Console.WriteLine(result["intent"].Choice);        // refund
Console.WriteLine(result["intent"].Confidence);    // normalised entropy of the distribution
```

## Checkpoints

No model data ships in this package. A checkpoint is downloaded on first use from
<https://models.curiosity.ai/laya/> into `~/.cache/laya` (or `$LAYA_HOME`, or `$HF_HOME/laya`)
and reused from there afterwards:

| `RemoteCheckpoint` | encoder | params | max tokens | download |
|---|---|---|---|---|
| `English` | ModernBERT-large | 421M | 512 | 843 MB |
| `Multilingual` | mmBERT-base | 322M | 1024 | 644 MB |
| `TypedDecisions` | ModernBERT-large | 421M | 1024 | 843 MB |

```csharp
RemoteCheckpoint.Multilingual.Prepare(progress: new Progress<DownloadProgress>(p => …));
RemoteCheckpoint.Multilingual.IsDownloaded();   // cached? then Load() never touches the network
```

Downloads resume where they stopped, and two processes starting at once share one download.
`LAYA_MODEL_BASE_URL` points them at a mirror with the same layout. `Agent.Load()` fetches a
checkpoint from the Hugging Face hub instead, and `Agent.FromDirectory(path)` loads one from disk.

## Links

* Source, docs and issues: <https://github.com/theolivenbaum/laya>
* Models: <https://huggingface.co/convaiinnovations/laya>

Apache-2.0. (c) Copyright 2026 Curiosity GmbH - all rights reserved. Laya itself is Copyright (c) 2025 Convai Innovations.
See `LICENSE` and `NOTICE` in this package.
