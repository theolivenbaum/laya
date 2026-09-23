# Laya.Training

Fine-tuning for [Laya](https://github.com/theolivenbaum/laya) decision checkpoints, on the CPU, in
managed code: backpropagation through ModernBERT and the typed decision head, the RLCD objective,
AdamW with cosine annealing, gradient clipping and temperature calibration — the C# port of the
reference typed-decisions fine-tuning notebook.

```csharp
using Laya.Tokenizers;
using Laya.Training;

var tokenizer = HuggingFaceTokenizer.FromDirectory(Path.Combine(source, "tokenizer"));
var cases = DecisionDataset.ReadJsonLines("train.jsonl");   // {state, questions, gold} per line
var items = DecisionDataset.BuildItems(tokenizer, cases, maxLength: 512, headMaxLength: 192);

var trainer = new Trainer(new TrainerOptions { TrainableEncoderLayers = 4 });
var report = trainer.Train(source, [.. items.Select(i => i.Item)], "my-decisions");

using var agent = Agent.FromDirectory("my-decisions");
var metrics = Evaluation.Evaluate(agent, DecisionDataset.ReadJsonLines("test.jsonl"));
```

`DecisionDataset.DownloadAsync` fetches any split of `LocalLLaMA/typed-decisions` (or a dataset of the
same shape) through the Hugging Face datasets-server, with no parquet reader.

The gradients are checked against PyTorch autograd tensor by tensor; see the repository README.
