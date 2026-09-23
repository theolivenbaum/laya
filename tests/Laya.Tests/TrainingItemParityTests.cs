using System.Text.Json;
using Laya.Tokenizers;
using Laya.Training;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// Training sequences against the notebook's <c>build_training_item</c>
/// (<c>tools/dump_training_items.py</c>, 40 cases / 200 items of the typed-decisions train split).
/// </summary>
public class TrainingItemParityTests
{
    internal static string TrainSplit { get; } =
        Path.Combine(TestModels.RepositoryRoot, "artifacts", "data", "LocalLLaMA_typed-decisions.all.train.jsonl");

    [TrainingDataFact]
    public void ItemsMatchTheNotebook()
    {
        string data = TrainSplit;

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestModels.FixtureRoot, "training-items-english.json")));
        var root = document.RootElement;
        var expected = root.GetProperty("items").EnumerateArray().ToList();

        var tokenizer = HuggingFaceTokenizer.FromDirectory(Path.Combine(TestModels.CheckpointDirectory("english"), "tokenizer"));
        var cases = DecisionDataset.ReadJsonLines(data).Take(40).ToList();
        var items = DecisionDataset.BuildItems(tokenizer, cases, root.GetProperty("max_len").GetInt32(),
            root.GetProperty("head_max_len").GetInt32());

        Assert.Equal(expected.Count, items.Count);
        for (int i = 0; i < items.Count; ++i)
        {
            var reference = expected[i];
            var item = items[i];
            string where = $"{item.CaseId}/{item.QuestionId}";
            Assert.Equal(reference.GetProperty("case").GetString(), item.CaseId);
            Assert.Equal(reference.GetProperty("question").GetString(), item.QuestionId);
            Assert.True(reference.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(item.Item.TokenIds), where + ": ids");
            Assert.True(reference.GetProperty("markers").EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(item.Item.MarkerPositions), where + ": markers");
            Assert.Equal(reference.GetProperty("qtype").GetInt32(), item.Item.QuestionType);
            Assert.Equal(reference.GetProperty("label").GetInt32(), item.Item.Label);
            var target = reference.GetProperty("target").EnumerateArray().Select(x => x.GetSingle()).ToArray();
            for (int k = 0; k < target.Length; ++k) Assert.Equal(target[k], item.Item.Target[k], 6);
        }
    }
}

/// <summary>Needs the English checkpoint and the typed-decisions train split (<c>laya dataset --split train</c>).</summary>
internal sealed class TrainingDataFactAttribute : Xunit.FactAttribute
{
    public TrainingDataFactAttribute()
    {
        if (!TestModels.Available("english")) Skip = "the english checkpoint is not in " + TestModels.ModelRoot;
        else if (!File.Exists(TrainingItemParityTests.TrainSplit)) Skip = TrainingItemParityTests.TrainSplit + " is missing; run: laya dataset --split train";
    }
}
