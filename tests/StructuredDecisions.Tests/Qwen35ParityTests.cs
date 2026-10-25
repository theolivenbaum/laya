using System.Text.Json;
using StructuredDecisions.Diagnostics;
using StructuredDecisions.Io;
using StructuredDecisions.Laya.Io;
using StructuredDecisions.Laya.Models;
using StructuredDecisions.Mica.Models;
using StructuredDecisions.Numerics;
using Xunit;
using Xunit.Abstractions;

namespace StructuredDecisions.Tests;

/// <summary>
/// The Qwen3.5 decoder behind Mica against transformers' <c>Qwen3_5ForCausalLM</c>, on a tiny random model
/// committed as a fixture (<c>tools/dump_mica_reference.py tiny</c>): both layer kinds, grouped KV heads,
/// partial RoPE, more value heads than key heads, and 150 tokens so transformers' chunked DeltaNet crosses
/// a 64-token chunk boundary while the C# side runs the recurrence token by token.
/// </summary>
public class Qwen35ParityTests(ITestOutputHelper output)
{
    private static string Fixture => Path.Combine(TestModels.FixtureRoot, "qwen35-tiny");

    private static (Qwen35Model Model, int[] Ids, SafetensorsFile Reference) Load(string weights)
    {
        var config = Qwen35Config.Load(Path.Combine(weights, "config.json"));
        using var checkpoint = new SafetensorsCheckpoint(weights);
        var model = new Qwen35Model(config, checkpoint);
        using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixture, "reference.json")));
        int[] ids = [.. meta.RootElement.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt32())];
        return (model, ids, new SafetensorsFile(Path.Combine(Fixture, "reference.safetensors")));
    }

    /// <summary>The fixture's weights rewritten as F32, to take the <see cref="PackedMatrix"/> path instead of bf16.</summary>
    private static string Fp32Copy()
    {
        string directory = Path.Combine(Path.GetTempPath(), "laya-qwen35-tiny-fp32");
        Directory.CreateDirectory(directory);
        using var source = new SafetensorsFile(Path.Combine(Fixture, "model.safetensors"));
        SafetensorsWriter.Write(Path.Combine(directory, "model.safetensors"),
            [.. source.Entries.Values.Select(e => new SafetensorsTensor(e.Name, e.Shape, source.ReadFloat32(e.Name), "F32"))]);
        File.Copy(Path.Combine(Fixture, "config.json"), Path.Combine(directory, "config.json"), overwrite: true);
        return directory;
    }

    [Theory]
    [InlineData(false, 1024)]       // bf16 weights, the production path
    [InlineData(true, 1024)]        // fp32 weights through PackedMatrix
    [InlineData(false, 37)]         // a layer read in chunks that do not divide the sequence
    public void EveryLayerMatchesTransformers(bool fp32, int rowsPerChunk)
    {
        var (model, ids, reference) = Load(fp32 ? Fp32Copy() : Fixture);
        using var _ = reference;
        model.RowsPerChunk = rowsPerChunk;
        var recorder = new StateRecorder();
        var last = model.Forward(ids, model.NewState(), recorder);

        // Against fp64, the C# model has to be as close as transformers' own fp32 run is — within 2x on every
        // tensor — and it has to agree with that fp32 run to fp32 rounding.
        foreach (var state in recorder.States)
        {
            if (!reference.Contains(state.Name)) continue;
            var torch32 = reference.ReadFloat32(state.Name);
            var exact = reference.ReadFloat32("fp64." + state.Name);
            double ours = RelativeError(exact, state.Values);
            double theirs = RelativeError(exact, torch32);
            double between = RelativeError(torch32, state.Values);
            output.WriteLine($"{state.Name,-18} vs fp64: C# {ours:E2}  torch fp32 {theirs:E2}   C# vs torch {between:E2}");
            Assert.True(ours <= 2 * theirs + 1e-7, $"{state.Name}: {ours:E2} from fp64, torch fp32 is {theirs:E2}");
            Assert.True(between < 1e-4, $"{state.Name}: {between:E2} from torch fp32");
        }

        // The tied LM head: every logit of the last position.
        var logits = reference.ReadFloat32("logits");
        var exactLogits = reference.ReadFloat32("fp64.logits");
        int vocab = model.Config.VocabSize;
        int[] all = [.. Enumerable.Range(0, vocab)];
        var actual = new float[vocab];
        model.Logits(last, all, actual);
        double logitError = RelativeError(logits, actual);
        double logitExact = RelativeError(exactLogits, actual), torchExact = RelativeError(exactLogits, logits);
        output.WriteLine($"logits vs fp64: C# {logitExact:E2}  torch fp32 {torchExact:E2}   C# vs torch {logitError:E2}");
        Assert.True(logitError < 1e-4, $"logits relative error {logitError:E2}");
        Assert.True(logitExact <= 2 * torchExact + 1e-7, $"logits: {logitExact:E2} from fp64, torch fp32 is {torchExact:E2}");
    }

    [Fact]
    public void ReadingInPiecesIsExactlyReadingAtOnce()
    {
        var (model, ids, reference) = Load(Fixture);
        reference.Dispose();
        var once = model.Forward(ids, model.NewState());

        var state = model.NewState();
        model.Forward(ids.AsSpan(0, 61), state);
        var branch = state.Clone();
        model.Forward(ids.AsSpan(61, 40), state);
        var pieces = model.Forward(ids.AsSpan(101), state);
        Assert.Equal(once, pieces);
        Assert.Equal(ids.Length, state.Length);

        // The clone is untouched by what the original read after it.
        Assert.Equal(61, branch.Length);
        Assert.Equal(once, model.Forward(ids.AsSpan(61), branch));
    }

    /// <summary>‖a − b‖ / ‖a‖, over the whole tensor.</summary>
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
}

public class Qwen35StateTests
{
    [Fact]
    public void RewindingToAMarkReadsTheSameAsAFreshState()
    {
        string fixture = Path.Combine(TestModels.FixtureRoot, "qwen35-tiny");
        using var checkpoint = new SafetensorsCheckpoint(fixture);
        var model = new Qwen35Model(Qwen35Config.Load(Path.Combine(fixture, "config.json")), checkpoint);
        int[] prefix = [.. Enumerable.Range(0, 70).Select(i => (i * 37) % 128)];
        int[] a = [5, 9, 11, 100], b = [7, 7, 7, 7, 7, 7, 1];

        var state = model.NewState();
        model.Forward(prefix, state);
        var mark = state.Save();
        var first = model.Forward(a, state);
        state.Rewind(mark);
        var second = model.Forward(b, state);

        Assert.Equal(model.Forward([.. prefix, .. a], model.NewState()), first);
        Assert.Equal(model.Forward([.. prefix, .. b], model.NewState()), second);
    }
}

public class Qwen35BatchTests
{
    private static Qwen35Model Model()
    {
        string fixture = Path.Combine(TestModels.FixtureRoot, "qwen35-tiny");
        using var checkpoint = new SafetensorsCheckpoint(fixture);
        return new Qwen35Model(Qwen35Config.Load(Path.Combine(fixture, "config.json")), checkpoint);
    }

    private static int[] Tokens(int count, int seed) => [.. Enumerable.Range(0, count).Select(i => (i * 37 + seed * 11) % 128)];

    /// <summary>Sequences read together in one pass get exactly what each gets alone, chunk boundaries included.</summary>
    [Theory]
    [InlineData(1024)]
    [InlineData(23)]
    public void ABatchReadsEachSequenceExactlyAsAlone(int rowsPerChunk)
    {
        var model = Model();
        model.RowsPerChunk = rowsPerChunk;
        int[] prefix = Tokens(50, 1);
        int[][] suffixes = [Tokens(7, 2), Tokens(1, 3), Tokens(30, 4)];

        var expected = suffixes.Select(s => model.Forward([.. prefix, .. s], model.NewState())).ToArray();

        using var shared = model.NewState();
        model.Read(prefix, shared);
        var branches = suffixes.Select(s => new Qwen35Sequence { Tokens = s, State = shared.Clone() }).ToArray();
        model.Forward(branches);
        for (int i = 0; i < suffixes.Length; ++i)
        {
            Assert.Equal(expected[i], branches[i].Hidden);
            Assert.Equal(prefix.Length + suffixes[i].Length, branches[i].State.Length);
        }
    }

    /// <summary>A snapshot taken during a pass is the state a separate read up to that point would leave.</summary>
    [Fact]
    public void CheckpointsAreTheStatesOfShorterReads()
    {
        var model = Model();
        model.RowsPerChunk = 16;
        int[] tokens = Tokens(60, 5), rest = Tokens(9, 6);
        var sequence = new Qwen35Sequence { Tokens = tokens, State = model.NewState(), Readout = false, Checkpoints = [5, 16, 33] };
        model.Forward([sequence]);

        for (int k = 0; k < 3; ++k)
        {
            int at = sequence.Checkpoints![k];
            Assert.Equal(at, sequence.Snapshots![k].Length);
            // Continuing from the snapshot and continuing from a state that read exactly that far agree bit for bit.
            var reference = model.NewState();
            model.Read(tokens.AsSpan(0, at), reference);
            Assert.Equal(model.Forward(rest, reference), model.Forward(rest, sequence.Snapshots[k]));
        }
    }
}
