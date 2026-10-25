using System.Text.Json.Nodes;
using StructuredDecisions.Diagnostics;
using StructuredDecisions.Io;
using StructuredDecisions.Laya.Io;
using StructuredDecisions.Mica;
using Xunit;
using Xunit.Abstractions;

namespace StructuredDecisions.Tests;

/// <summary>
/// The real Mica checkpoint against transformers in fp32, layer by layer, for prompts dumped by
/// <c>tools/dump_mica_reference.py checkpoint</c> into <c>artifacts/dumps/&lt;name&gt;/</c>. Skips when the
/// weights or the dumps are not on disk; nothing in CI downloads 9 GB.
/// </summary>
public class MicaCheckpointParityTests(ITestOutputHelper output)
{
    private static readonly Lazy<MicaJudge> Judge = new(() => MicaJudge.FromDirectory(MicaFactAttribute.Directory));

    /// <summary>One loaded checkpoint for every test that needs the weights: loading costs 30 s and 8 GB.</summary>
    internal static MicaJudge SharedJudge => Judge.Value;

    private static string Dumps => Path.Combine(TestModels.RepositoryRoot, "artifacts", "dumps");

    public static IEnumerable<object[]> References()
    {
        if (!MicaFactAttribute.HasWeights || !Directory.Exists(Dumps)) yield break;
        foreach (string directory in Directory.GetDirectories(Dumps).Order(StringComparer.Ordinal))
        {
            if (File.Exists(Path.Combine(directory, "reference.safetensors"))) yield return [Path.GetFileName(directory)];
        }
    }

    [Theory]
    [MemberData(nameof(References))]
    public void EveryLayerMatchesTransformers(string name)
    {
        string directory = Path.Combine(Dumps, name);
        var meta = JsonNode.Parse(File.ReadAllText(Path.Combine(directory, "reference.json")))!;
        using var reference = new SafetensorsFile(Path.Combine(directory, "reference.safetensors"));

        var rows = MicaContract.RowsFromRequest(meta["request"]!.DeepClone());
        var all = rows.Where(r => !r.Multi).Concat(rows.Where(r => r.Multi).SelectMany(MicaContract.InclusionRows)).ToList();
        var row = all[meta["row"]!.GetValue<int>()];
        var judge = Judge.Value;

        // Same prompt text and the same ids as transformers' tokenizer produced for it.
        Assert.Equal(meta["prompt"]!.GetValue<string>(), judge.Prompt.Render(row));
        int[] ids = [.. meta["input_ids"]!.AsArray().Select(n => n!.GetValue<int>())];
        Assert.Equal(ids, judge.PromptTokens(row));

        var recorder = new StateRecorder();
        var (distributions, _) = judge.Predict([row], recorder);

        double worst = 0;
        foreach (var state in recorder.States)
        {
            string tensor = state.Name.Replace("row0.", "", StringComparison.Ordinal);
            if (!reference.Contains(tensor)) continue;
            var expected = reference.ReadFloat32(tensor);
            double error = RelativeError(expected, state.Values);
            double maxAbs = MaxAbsDifference(expected, state.Values);
            output.WriteLine($"{tensor,-20} relative {error:E2}   max |Δ| {maxAbs:E2}");
            worst = Math.Max(worst, error);
        }
        Assert.True(worst < 1e-4, $"worst relative error {worst:E2}");

        var probabilities = meta["probabilities"]!.AsArray().Select(n => n!.GetValue<double>()).ToArray();
        output.WriteLine($"probabilities torch [{string.Join(", ", probabilities.Select(p => p.ToString("F6")))}]");
        output.WriteLine($"probabilities C#    [{string.Join(", ", distributions[0].Select(p => p.ToString("F6")))}]");
        for (int i = 0; i < probabilities.Length; ++i) Assert.True(Math.Abs(probabilities[i] - distributions[0][i]) < 1e-4);
    }

    private static double RelativeError(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double diff = 0, norm = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            double d = expected[i] - actual[i];
            diff += d * d;
            norm += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(diff / Math.Max(norm, 1e-30));
    }

    private static double MaxAbsDifference(float[] expected, float[] actual)
    {
        double max = 0;
        for (int i = 0; i < expected.Length; ++i) max = Math.Max(max, Math.Abs(expected[i] - actual[i]));
        return max;
    }
}
