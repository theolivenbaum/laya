# Laya

**Typed decisions for .NET, in one forward pass — no Python, no PyTorch, no native dependency.**

Laya answers typed questions (`choice`, `score`, `noul`) about any state — text, an email, a
ticket, a JSON document — and returns a calibrated probability distribution for each answer. Ask
five questions about one ticket and five answers come back in a single call. Nothing is
generated, so there is no prompt to tune, no JSON to parse and nothing to hallucinate. It runs on
the CPU, on managed SIMD, inside your own process.

```csharp
using Laya;
using Laya.Io;
using Laya.Runtime;

// Downloads the English checkpoint on first use (~843 MB) into ~/.cache/laya.
using var agent = RemoteCheckpoint.English.Load();

var result = agent.SystemOne("We were billed twice for March, please refund it.", Presets.Triage());

Console.WriteLine(result["intent"].Choice);             // refund
Console.WriteLine(result["refund_requested"].Noul);    // 0.90
Console.WriteLine(result["intent"].Confidence);        // normalised entropy of the distribution
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
* The original Python implementation, by Convai Innovations: <https://github.com/NandhaKishorM/laya>
* Models: <https://huggingface.co/convaiinnovations/laya>

Apache-2.0. (c) Copyright 2026 Curiosity GmbH - all rights reserved. Laya itself is Copyright (c) 2025 Convai Innovations.
See `LICENSE` and `NOTICE` in this package.
