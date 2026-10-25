using StructuredDecisions.Diagnostics;
using StructuredDecisions.Io;
using StructuredDecisions.Numerics;

namespace StructuredDecisions.Mica.Models;

/// <summary>
/// Recycles the large buffers a <see cref="Qwen35State"/> is made of.
///
/// <para>A state holds 2 MB of DeltaNet state for each of 24 layers. Fresh from the GC, those 48 MB arrive as
/// untouched pages, and the first write to each page faults: 380 ms per state on the VM this was measured on —
/// as long as the whole recurrence of a 200-token prompt. A disposed state hands its buffers back here and
/// the next one reuses pages that are already mapped. Buffers are bucketed by exact length; key/value caches
/// grow in powers of two so their sizes repeat too.</para>
/// </summary>
internal sealed class StateBufferPool
{
    private readonly System.Collections.Concurrent.ConcurrentDictionary<int, System.Collections.Concurrent.ConcurrentBag<float[]>> _buckets = new();

    public float[] Rent(int length, bool clear)
    {
        if (length == 0) return [];
        if (_buckets.TryGetValue(length, out var bucket) && bucket.TryTake(out var buffer))
        {
            if (clear) Array.Clear(buffer);
            return buffer;
        }
        return new float[length];
    }

    public void Return(float[]? buffer)
    {
        if (buffer is null || buffer.Length == 0) return;
        _buckets.GetOrAdd(buffer.Length, _ => []).Add(buffer);
    }
}

/// <summary>
/// What a Qwen3.5 decoder remembers of the tokens it has read: keys and values for each full-attention
/// layer, and the convolution tail plus the delta-rule state for each DeltaNet layer.
///
/// <para>Reading a prompt in two calls through one state gives bit-for-bit the activations of reading it
/// in one: every GEMM row, every attention row and the recurrence are computed the same way either way.
/// That is what lets Mica read the system prompt and the state once and then branch per question with
/// <see cref="Save"/>/<see cref="Rewind"/> or <see cref="Clone"/>.</para>
///
/// <para>Dispose a state when done with it: its buffers go back to the model's pool (see
/// <see cref="StateBufferPool"/>). An undisposed state is simply collected.</para>
/// </summary>
public sealed class Qwen35State : IDisposable
{
    internal readonly float[]?[] Keys;          // per layer: [capacity, kvHeads * headDim], full attention only
    internal readonly float[]?[] Values;
    internal readonly float[]?[] ConvTail;      // per layer: [kernel - 1, convDim], the last pre-convolution rows
    internal readonly float[]?[] Recurrent;     // per layer: [vHeads, dk, dv]
    private readonly StateBufferPool _pool;
    private bool _disposed;

    /// <summary>How many tokens have been read into this state.</summary>
    public int Length { get; internal set; }

    internal Qwen35State(Qwen35Config config, StateBufferPool pool)
    {
        _pool = pool;
        int layers = config.NumHiddenLayers;
        Keys = new float[layers][];
        Values = new float[layers][];
        ConvTail = new float[layers][];
        Recurrent = new float[layers][];
        for (int i = 0; i < layers; ++i)
        {
            if (config.LayerKinds[i] == Qwen35LayerKind.FullAttention)
            {
                Keys[i] = [];
                Values[i] = [];
            }
            else
            {
                ConvTail[i] = pool.Rent((config.LinearConvKernel - 1) * config.LinearConvDim, clear: true);
                Recurrent[i] = pool.Rent(config.LinearNumValueHeads * config.LinearKeyHeadDim * config.LinearValueHeadDim, clear: true);
            }
        }
    }

    private Qwen35State(Qwen35State other)
    {
        _pool = other._pool;
        int layers = other.Keys.Length;
        Keys = new float[layers][];
        Values = new float[layers][];
        ConvTail = new float[layers][];
        Recurrent = new float[layers][];
        for (int i = 0; i < layers; ++i)
        {
            Keys[i] = Copy(other.Keys[i]);
            Values[i] = Copy(other.Values[i]);
            ConvTail[i] = Copy(other.ConvTail[i]);
            Recurrent[i] = Copy(other.Recurrent[i]);
        }
        Length = other.Length;
    }

    private float[]? Copy(float[]? source)
    {
        if (source is null) return null;
        var copy = _pool.Rent(source.Length, clear: false);
        source.AsSpan().CopyTo(copy);
        return copy;
    }

    /// <summary>An independent copy: what is read into one never reaches the other.</summary>
    public Qwen35State Clone()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new(this);
    }

    /// <summary>
    /// A point to come back to: the recurrent and convolution state as they are now. Keys and values are
    /// not copied — rewinding only moves <see cref="Length"/> back, and whatever is read next overwrites
    /// the cache from there. Reading one long shared prefix and then several suffixes therefore costs one
    /// copy of the (small, fixed-size) DeltaNet state per suffix instead of a copy of the whole KV cache.
    /// Dispose the mark to recycle its copy.
    /// </summary>
    public Mark Save()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new(Length, [.. ConvTail.Select(Copy)], [.. Recurrent.Select(Copy)], _pool);
    }

    /// <summary>Returns to a <see cref="Save"/>d point of this same state.</summary>
    public void Rewind(Mark mark)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (mark.Length > Length) throw new ArgumentException("a mark can only rewind, not advance", nameof(mark));
        for (int i = 0; i < ConvTail.Length; ++i)
        {
            mark.ConvTail[i]?.CopyTo(ConvTail[i]!, 0);
            mark.Recurrent[i]?.CopyTo(Recurrent[i]!, 0);
        }
        Length = mark.Length;
    }

    /// <summary>Grows a key/value cache to hold <paramref name="needed"/> floats, keeping the first <paramref name="keep"/>.</summary>
    internal float[] GrowCache(ref float[] buffer, int needed, int keep)
    {
        if (buffer.Length >= needed) return buffer;
        int capacity = (int)Math.Max(needed, Math.Min(int.MaxValue, System.Numerics.BitOperations.RoundUpToPowerOf2((uint)needed)));
        var grown = _pool.Rent(capacity, clear: false);
        buffer.AsSpan(0, keep).CopyTo(grown);
        _pool.Return(buffer);
        buffer = grown;
        return grown;
    }

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <summary>A saved point of a <see cref="Qwen35State"/>.</summary>
    public sealed class Mark : IDisposable
    {
        private readonly StateBufferPool _pool;
        internal Mark(int length, float[]?[] convTail, float[]?[] recurrent, StateBufferPool pool)
        {
            Length = length;
            ConvTail = convTail;
            Recurrent = recurrent;
            _pool = pool;
        }

        public int Length { get; }
        internal float[]?[] ConvTail { get; }
        internal float[]?[] Recurrent { get; }

        public void Dispose()
        {
            foreach (var buffer in ConvTail) _pool.Return(buffer);
            foreach (var buffer in Recurrent) _pool.Return(buffer);
            Array.Clear(ConvTail);
            Array.Clear(Recurrent);
        }
    }

    /// <summary>Bytes held, to judge what keeping a prefix costs.</summary>
    public long Bytes => sizeof(float) * (Keys.Sum(k => (long)(k?.Length ?? 0)) + Values.Sum(v => (long)(v?.Length ?? 0))
        + ConvTail.Sum(c => (long)(c?.Length ?? 0)) + Recurrent.Sum(r => (long)(r?.Length ?? 0)));

    /// <summary>Returns every buffer to the model's pool. The state cannot be used afterwards.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var group in new[] { Keys, Values, ConvTail, Recurrent })
        {
            foreach (var buffer in group) _pool.Return(buffer);
            Array.Clear(group);
        }
    }
}

/// <summary>One sequence of a batched <see cref="Qwen35Model.Forward(IReadOnlyList{Qwen35Sequence}, IStateRecorder?, ParallelOptions?)"/>.</summary>
public sealed class Qwen35Sequence
{
    /// <summary>The tokens to read, after whatever <see cref="State"/> holds.</summary>
    public required ReadOnlyMemory<int> Tokens { get; init; }

    /// <summary>The state this sequence continues; it is advanced by the read.</summary>
    public required Qwen35State State { get; init; }

    /// <summary>Whether to compute <see cref="Hidden"/>. A prefix that is only read for later reads does not need it.</summary>
    public bool Readout { get; init; } = true;

    /// <summary>Token counts (1..<c>Tokens.Length</c>, ascending) after which to snapshot the state, into <see cref="Snapshots"/>.</summary>
    public IReadOnlyList<int>? Checkpoints { get; init; }

    /// <summary>The final-normed hidden state of the last token, when <see cref="Readout"/>.</summary>
    public float[]? Hidden { get; internal set; }

    /// <summary>One independent state per checkpoint, owned by the caller (dispose them).</summary>
    public Qwen35State[]? Snapshots { get; internal set; }
}

/// <summary>
/// The Qwen3.5 text decoder (<c>Qwen3_5TextModel</c> in transformers), forward only, the backbone of Mica.
///
/// <para>Thirty-two pre-norm layers; three in four mix tokens with a Gated DeltaNet and every fourth with
/// gated softmax attention. Every layer ends in a SwiGLU MLP. Mica only ever needs the next-token logits
/// of the last prompt position, and only for its label tokens, so the 248k-row LM head is never
/// multiplied out: <see cref="Logits"/> takes the dot products it needs from the tied embedding.</para>
///
/// <para>Weights stay in the checkpoint's bf16 (<see cref="PackedMatrixBf16"/>); compute is fp32
/// throughout. The DeltaNet runs as its token-by-token recurrence. transformers evaluates the same
/// recurrence chunk-parallel for a prefill — algebraically identical, and within fp32 rounding of it.</para>
/// </summary>
public sealed class Qwen35Model
{
    /// <summary>Rows processed per pass through a layer: bounds the wide intermediates for long prompts.</summary>
    public int RowsPerChunk { get; set; } = 1024;

    private readonly Qwen35Config _config;
    private readonly TokenTable _embeddings;
    private readonly TokenTable _head;
    private readonly float[] _finalNorm;          // 1 + weight
    private readonly Layer[] _layers;
    private readonly float[] _inverseFrequency;

    public Qwen35Config Config => _config;
    public long WeightBytes { get; }

    /// <summary>How the projections and the token table are kept: bf16 (exact) or ggml Q8_0.</summary>
    public WeightPrecision Precision { get; }

    private sealed class Layer
    {
        public required Qwen35LayerKind Kind;
        public required float[] InputNorm;        // 1 + weight
        public required float[] PostAttentionNorm;
        public required ILinear Mlp;              // [gate; up], 2 * intermediate outputs
        public required ILinear Down;

        // Full attention.
        public ILinear? Qkv;                      // [q|gate per head; k; v]
        public ILinear? Output;
        public float[]? QueryNorm, KeyNorm;

        // Gated DeltaNet.
        public ILinear? InProjection;             // [qkv (conv dim); z (value dim); b (v heads); a (v heads)]
        public float[]? ConvWeight;               // [kernel, convDim]: transposed so a tap is contiguous across channels
        public float[]? NegativeA;                // -exp(A_log)
        public float[]? DtBias;
        public float[]? GatedNorm;
        public ILinear? OutProjection;
    }

    /// <param name="precision"><see cref="WeightPrecision.Q8_0"/> quantizes every projection and the tied embedding at load
    /// with ggml's Q8_0 and multiplies them as ggml does (activations quantized too) — what llama.cpp runs from Mica's
    /// <c>Q8_0.gguf</c>; norms, the conv kernel and the DeltaNet's gates stay fp32.</param>
    public Qwen35Model(Qwen35Config config, SafetensorsCheckpoint weights, string? prefix = null,
        WeightPrecision precision = WeightPrecision.Bf16)
    {
        if (!config.AttentionOutputGate) throw new NotSupportedException("Qwen3.5 without attn_output_gate is not implemented.");
        _config = config;
        Precision = precision;
        prefix ??= weights.Contains("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";

        _embeddings = TokenTable.Load(weights, prefix + "embed_tokens.weight", config.HiddenSize, precision);
        _head = config.TieWordEmbeddings || !weights.Contains("lm_head.weight")
            ? _embeddings
            : TokenTable.Load(weights, "lm_head.weight", config.HiddenSize, precision);
        _finalNorm = OnePlus(weights.ReadFloat32(prefix + "norm.weight"));

        int hidden = config.HiddenSize;
        _layers = new Layer[config.NumHiddenLayers];
        for (int i = 0; i < _layers.Length; ++i)
        {
            string p = $"{prefix}layers.{i}.";
            var kind = config.LayerKinds[i];
            var layer = new Layer
            {
                Kind = kind,
                InputNorm = OnePlus(weights.ReadFloat32(p + "input_layernorm.weight")),
                PostAttentionNorm = OnePlus(weights.ReadFloat32(p + "post_attention_layernorm.weight")),
                Mlp = Stack(weights, precision, [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"], hidden),
                Down = Stack(weights, precision, [p + "mlp.down_proj.weight"], config.IntermediateSize),
            };
            if (kind == Qwen35LayerKind.FullAttention)
            {
                layer.Qkv = Stack(weights, precision, [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"], hidden);
                layer.Output = Stack(weights, precision, [p + "self_attn.o_proj.weight"], config.NumAttentionHeads * config.HeadDim);
                layer.QueryNorm = OnePlus(weights.ReadFloat32(p + "self_attn.q_norm.weight"));
                layer.KeyNorm = OnePlus(weights.ReadFloat32(p + "self_attn.k_norm.weight"));
            }
            else
            {
                string a = p + "linear_attn.";
                layer.InProjection = Stack(weights, precision,
                    [a + "in_proj_qkv.weight", a + "in_proj_z.weight", a + "in_proj_b.weight", a + "in_proj_a.weight"], hidden);
                layer.ConvWeight = Transpose(weights.ReadFloat32(a + "conv1d.weight"), config.LinearConvDim, config.LinearConvKernel);
                layer.NegativeA = [.. weights.ReadFloat32(a + "A_log").Select(v => -MathF.Exp(v))];
                layer.DtBias = weights.ReadFloat32(a + "dt_bias");
                layer.GatedNorm = weights.ReadFloat32(a + "norm.weight");
                layer.OutProjection = Stack(weights, precision, [a + "out_proj.weight"], config.LinearValueDim);
            }
            _layers[i] = layer;
        }

        // inv_freq = 1 / theta^(2i / rotary_dim), in fp32 as transformers builds it.
        int half = config.RotaryDim / 2;
        _inverseFrequency = new float[half];
        for (int i = 0; i < half; ++i)
        {
            _inverseFrequency[i] = 1f / MathF.Pow((float)config.RopeTheta, 2f * i / config.RotaryDim);
        }

        WeightBytes = _embeddings.Bytes + (_head == _embeddings ? 0 : _head.Bytes) + _layers.Sum(l =>
            l.Mlp.Bytes + l.Down.Bytes + (l.Qkv?.Bytes ?? 0) + (l.Output?.Bytes ?? 0)
            + (l.InProjection?.Bytes ?? 0) + (l.OutProjection?.Bytes ?? 0));
    }

    /// <summary><c>[rows, columns]</c> to <c>[columns, rows]</c> (here: the conv's <c>[channel, 1, tap]</c> to <c>[tap, channel]</c>).</summary>
    private static float[] Transpose(float[] source, int rows, int columns)
    {
        var result = new float[source.Length];
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < columns; ++c) result[c * rows + r] = source[r * columns + c];
        }
        return result;
    }

    private static float[] OnePlus(float[] weight)
    {
        for (int i = 0; i < weight.Length; ++i) weight[i] = 1f + weight[i];
        return weight;
    }

    /// <summary>
    /// Packs the named <c>[out, in]</c> weights stacked along the output dimension into one projection, so
    /// a layer's sibling projections run as one GEMM. Each output column is computed independently, so
    /// stacking changes no value. bf16 weights stay bf16; anything else is widened to fp32.
    /// </summary>
    private static ILinear Stack(SafetensorsCheckpoint weights, WeightPrecision precision, string[] names, int inFeatures)
    {
        var entries = names.Select(weights.Entry).ToArray();
        foreach (var entry in entries)
        {
            if (entry.Shape.Length != 2 || entry.Shape[1] != inFeatures)
            {
                throw new InvalidDataException($"{entry.Name}: expected [*, {inFeatures}], found [{string.Join(", ", entry.Shape)}].");
            }
        }
        int outFeatures = entries.Sum(e => e.Shape[0]);
        if (entries.All(e => e.DType == "BF16"))
        {
            if (precision == WeightPrecision.Q8_0)
            {
                // Each output row is quantized on its own, so stacking first changes nothing; quantize part by part.
                var parts = names.Select(n => weights.ReadBFloat16Bits(n)).ToArray();
                return parts.Length == 1
                    ? new PackedMatrixQ8_0(parts[0], outFeatures, inFeatures)
                    : new PackedMatrixQ8_0(Concatenate(parts), outFeatures, inFeatures);
            }
            if (names.Length == 1) return new PackedMatrixBf16(weights.ReadBFloat16Bits(names[0]), outFeatures, inFeatures);
            var stacked = new ushort[(long)outFeatures * inFeatures];
            long offset = 0;
            foreach (string name in names)
            {
                var part = weights.ReadBFloat16Bits(name);
                part.CopyTo(stacked.AsSpan((int)offset));
                offset += part.Length;
            }
            return new PackedMatrixBf16(stacked, outFeatures, inFeatures);
        }

        var values = new float[(long)outFeatures * inFeatures];
        long at = 0;
        foreach (string name in names)
        {
            var part = weights.ReadFloat32(name);
            part.CopyTo(values.AsSpan((int)at));
            at += part.Length;
        }
        return precision == WeightPrecision.Q8_0
            ? new PackedMatrixQ8_0(values, outFeatures, inFeatures)
            : new PackedMatrix(values, outFeatures, inFeatures);
    }

    private static ushort[] Concatenate(ushort[][] parts)
    {
        var all = new ushort[parts.Sum(p => (long)p.Length)];
        long at = 0;
        foreach (var part in parts)
        {
            part.CopyTo(all, at);
            at += part.Length;
        }
        return all;
    }

    private readonly StateBufferPool _pool = new();

    /// <summary>A state that has read nothing yet; its buffers come from this model's pool. Dispose it when done.</summary>
    public Qwen35State NewState() => new(_config, _pool);

    /// <summary>
    /// Reads <paramref name="tokens"/> after whatever <paramref name="state"/> already holds and returns
    /// the final-normed hidden state of the last one (<c>[hidden]</c>), which is what the LM head reads.
    /// </summary>
    public float[] Forward(ReadOnlySpan<int> tokens, Qwen35State state, IStateRecorder? recorder = null,
        ParallelOptions? parallel = null)
    {
        var sequence = new Qwen35Sequence { Tokens = tokens.ToArray(), State = state };
        Forward([sequence], recorder, parallel);
        return sequence.Hidden!;
    }

    /// <summary>
    /// Reads <paramref name="tokens"/> into <paramref name="state"/> for later reads to continue from — a shared
    /// prefix — without producing any output.
    /// </summary>
    public void Read(ReadOnlySpan<int> tokens, Qwen35State state, IStateRecorder? recorder = null, ParallelOptions? parallel = null)
        => Forward([new Qwen35Sequence { Tokens = tokens.ToArray(), State = state, Readout = false }], recorder, parallel);

    /// <summary>
    /// Reads several independent sequences in one pass, each continuing its own state.
    ///
    /// <para>The weights are what a pass costs on a CPU: 8 GB of them stream through the cores whether the pass reads
    /// one token or two hundred (a one-token pass takes a third of a second here). So the projections and MLPs run once
    /// over all sequences' rows stacked together, while everything that mixes tokens — attention over each sequence's
    /// own keys, the depthwise convolution, the delta-rule recurrence — runs per sequence on its own state. Every row
    /// of a GEMM is computed independently, so a sequence read in a batch gets exactly the activations it gets alone.
    /// Mica's questions about one state are such a batch: their common prefix is read once, and then all the
    /// questions continue from copies of it together.</para>
    ///
    /// <para>A sequence can ask for <see cref="Qwen35Sequence.Checkpoints"/>: snapshots of its state after some of its
    /// tokens, taken during the pass instead of by stopping the pass there.</para>
    ///
    /// <para>Nothing after the last layer reads any position but a sequence's last, so in the last layer only the rows
    /// whose output is wanted go through attention, <c>o_proj</c> and the MLP — keys and values, which later reads
    /// attend to, are computed for every row. A recorder turns this off (and takes a single sequence), so parity
    /// dumps keep every row of every layer.</para>
    /// </summary>
    public void Forward(IReadOnlyList<Qwen35Sequence> sequences, IStateRecorder? recorder = null, ParallelOptions? parallel = null)
    {
        if (sequences.Count == 0) return;
        if (recorder is not null && sequences.Count != 1) throw new ArgumentException("a recorder records a single sequence");
        int hidden = _config.HiddenSize;
        var batch = new Batch(sequences);
        int rows = batch.Rows;
        foreach (var sequence in sequences)
        {
            if (sequence.Tokens.Length == 0) throw new ArgumentException("nothing to read", nameof(sequences));
            sequence.State.ThrowIfDisposed();
            sequence.Snapshots = sequence.Checkpoints is { Count: > 0 } points ? [.. points.Select(_ => NewState())] : [];
        }

        var x = new float[(long)rows * hidden];
        using (ForwardTiming.Measure("embeddings"))
        {
            for (int b = 0; b < sequences.Count; ++b)
            {
                var tokens = sequences[b].Tokens.Span;
                for (int t = 0; t < tokens.Length; ++t) _embeddings.Row(tokens[t], x.AsSpan((batch.First[b] + t) * hidden, hidden));
            }
        }
        recorder?.Record("embeddings", x, [rows, hidden]);

        var mixed = recorder is null ? null : new float[x.Length];
        for (int i = 0; i < _layers.Length; ++i)
        {
            var layer = _layers[i];
            // Rows whose output this layer has to produce: all of them, except in a last full-attention layer.
            bool prune = recorder is null && i == _layers.Length - 1 && layer.Kind == Qwen35LayerKind.FullAttention;
            for (int chunk = 0; chunk < rows; chunk += RowsPerChunk)
            {
                int count = Math.Min(RowsPerChunk, rows - chunk);
                var pieces = batch.Pieces(chunk, count);
                int[] outputs = prune ? batch.ReadoutRows(chunk, count) : [.. Enumerable.Range(0, count)];
                var slice = x.AsSpan(chunk * hidden, count * hidden);
                var record = mixed is null ? default : mixed.AsSpan(chunk * hidden, count * hidden);
                if (layer.Kind == Qwen35LayerKind.FullAttention) FullAttention(layer, i, slice, count, pieces, outputs, record, parallel);
                else LinearAttention(layer, i, slice, count, pieces, record, parallel);
                if (outputs.Length > 0) Mlp(layer, slice, outputs, parallel);
            }
            if (layer.Kind == Qwen35LayerKind.FullAttention) SnapshotKeysAndValues(i, sequences);
            if (recorder is not null)
            {
                recorder.Record($"layers.{i}.mixer", mixed, [rows, hidden]);
                recorder.Record($"layers.{i}", x, [rows, hidden]);
            }
        }

        for (int b = 0; b < sequences.Count; ++b)
        {
            var sequence = sequences[b];
            int start = sequence.State.Length;
            for (int k = 0; k < sequence.Snapshots!.Length; ++k) sequence.Snapshots[k].Length = start + sequence.Checkpoints![k];
            sequence.State.Length = start + sequence.Tokens.Length;
            if (!sequence.Readout) continue;
            sequence.Hidden = new float[hidden];
            int last = batch.First[b] + sequence.Tokens.Length - 1;
            DecoderKernels.RmsNorm(x.AsSpan(last * hidden, hidden), _finalNorm, _config.RmsNormEps, sequence.Hidden);
        }
        if (recorder is not null)
        {
            var normed = new float[x.Length];
            for (int t = 0; t < rows; ++t)
            {
                DecoderKernels.RmsNorm(x.AsSpan(t * hidden, hidden), _finalNorm, _config.RmsNormEps, normed.AsSpan(t * hidden, hidden));
            }
            recorder.Record("norm", normed, [rows, hidden]);
        }
    }

    /// <summary>Where each sequence's rows sit in a batch, and the slices of a row chunk that belong to each.</summary>
    private sealed class Batch
    {
        public readonly IReadOnlyList<Qwen35Sequence> Sequences;
        public readonly int[] First;
        public readonly int Rows;

        public Batch(IReadOnlyList<Qwen35Sequence> sequences)
        {
            Sequences = sequences;
            First = new int[sequences.Count];
            for (int b = 0; b < sequences.Count; ++b)
            {
                First[b] = Rows;
                Rows += sequences[b].Tokens.Length;
            }
        }

        /// <summary>The (sequence, first chunk row, first sequence row, rows) runs covering chunk rows <c>[chunk, chunk + count)</c>.</summary>
        public List<Piece> Pieces(int chunk, int count)
        {
            var pieces = new List<Piece>();
            for (int b = 0; b < Sequences.Count; ++b)
            {
                int from = Math.Max(chunk, First[b]), to = Math.Min(chunk + count, First[b] + Sequences[b].Tokens.Length);
                if (from < to) pieces.Add(new Piece(b, Sequences[b], from - chunk, from - First[b], to - from));
            }
            return pieces;
        }

        /// <summary>Chunk rows that are the last row of a sequence that wants its output.</summary>
        public int[] ReadoutRows(int chunk, int count)
        {
            var rows = new List<int>();
            for (int b = 0; b < Sequences.Count; ++b)
            {
                int last = First[b] + Sequences[b].Tokens.Length - 1;
                if (Sequences[b].Readout && last >= chunk && last < chunk + count) rows.Add(last - chunk);
            }
            return [.. rows];
        }
    }

    /// <param name="Row">Its first row within the chunk.</param>
    /// <param name="Offset">Its first row within the sequence.</param>
    private readonly record struct Piece(int Index, Qwen35Sequence Sequence, int Row, int Offset, int Rows)
    {
        /// <summary>The absolute position of the piece's first token.</summary>
        public int Position => Sequence.State.Length + Offset;
    }

    /// <summary>The LM head's logits for just <paramref name="ids"/>: <c>hidden · E[id]</c>.</summary>
    public void Logits(ReadOnlySpan<float> hidden, ReadOnlySpan<int> ids, Span<float> logits)
    {
        if (_head.IsQuantized)
        {
            _head.QuantizedLogits(hidden, ids, logits);
            return;
        }
        var buffer = new float[_config.HiddenSize];
        for (int i = 0; i < ids.Length; ++i)
        {
            _head.Row(ids[i], buffer);
            logits[i] = SimdOps.Dot(hidden, buffer);
        }
    }

    /// <summary>The MLP on the chunk rows in <paramref name="rows"/> (all of them, or the few the last layer needs).</summary>
    private void Mlp(Layer layer, Span<float> x, int[] rows, ParallelOptions? parallel)
    {
        int hidden = _config.HiddenSize, inter = _config.IntermediateSize, count = rows.Length;
        using var scratch = new ScratchBuffers();
        int tiled = Tiled(count);
        var normed = scratch.Rent(tiled * hidden);
        var gateUp = scratch.Rent(tiled * 2 * inter);
        var act = scratch.Rent(tiled * inter);
        var down = scratch.Rent(tiled * hidden);

        for (int r = 0; r < count; ++r)
        {
            DecoderKernels.RmsNorm(x.Slice(rows[r] * hidden, hidden), layer.PostAttentionNorm, _config.RmsNormEps, normed.AsSpan(r * hidden, hidden));
        }
        using (ForwardTiming.Measure("mlp.gate_up")) Project(layer.Mlp, normed, count, gateUp, parallel);
        using (ForwardTiming.Measure("mlp.act")) ForRows(count, parallel, t => DecoderKernels.SiluMultiply(
            gateUp.AsSpan(t * 2 * inter, inter), gateUp.AsSpan(t * 2 * inter + inter, inter), act.AsSpan(t * inter, inter)));
        using (ForwardTiming.Measure("mlp.down")) Project(layer.Down, act, count, down, parallel);
        for (int r = 0; r < count; ++r) SimdOps.Add(x.Slice(rows[r] * hidden, hidden), down.AsSpan(r * hidden, hidden));
    }

    /// <summary>
    /// Gated softmax attention over everything each sequence has read: per-head RMSNorm on q and k, RoPE on the first
    /// <see cref="Qwen35Config.RotaryDim"/> dimensions of each head, causal GQA, then <c>· sigmoid(gate)</c>.
    /// Every row's key and value join its sequence's cache; only the chunk rows in <paramref name="outputs"/> attend.
    /// </summary>
    private void FullAttention(Layer layer, int index, Span<float> x, int rows, List<Piece> pieces, int[] outputs,
        Span<float> record, ParallelOptions? parallel)
    {
        var c = _config;
        int hidden = c.HiddenSize, headDim = c.HeadDim, heads = c.NumAttentionHeads, kvHeads = c.NumKeyValueHeads;
        int qWidth = 2 * heads * headDim, kvWidth = kvHeads * headDim, width = qWidth + 2 * kvWidth;

        using var scratch = new ScratchBuffers();
        int tiled = Tiled(rows);
        var normed = scratch.Rent(tiled * hidden);
        var qkv = scratch.Rent(tiled * width);
        var attention = scratch.Rent(Tiled(outputs.Length) * heads * headDim);
        var projected = scratch.Rent(Tiled(outputs.Length) * hidden);

        for (int t = 0; t < rows; ++t)
        {
            DecoderKernels.RmsNorm(x.Slice(t * hidden, hidden), layer.InputNorm, c.RmsNormEps, normed.AsSpan(t * hidden, hidden));
        }
        using (ForwardTiming.Measure("full.qkv")) Project(layer.Qkv!, normed, rows, qkv, parallel);

        // Append each piece's keys and values to its sequence's cache, normed and rotated.
        var rowPiece = new int[rows];
        var keys = new float[pieces.Count][];
        var values = new float[pieces.Count][];
        for (int p = 0; p < pieces.Count; ++p)
        {
            var piece = pieces[p];
            var state = piece.Sequence.State;
            int end = piece.Position + piece.Rows;
            keys[p] = state.GrowCache(ref state.Keys[index]!, end * kvWidth, piece.Position * kvWidth);
            values[p] = state.GrowCache(ref state.Values[index]!, end * kvWidth, piece.Position * kvWidth);
            for (int t = 0; t < piece.Rows; ++t) rowPiece[piece.Row + t] = p;
        }
        ForRows(rows, parallel, t =>
        {
            var piece = pieces[rowPiece[t]];
            var row = qkv.AsSpan(t * width, width);
            int pos = piece.Position + (t - piece.Row);
            Span<float> cos = stackalloc float[c.RotaryDim / 2];
            Span<float> sin = stackalloc float[c.RotaryDim / 2];
            Rotation(pos, cos, sin);
            for (int h = 0; h < heads; ++h)
            {
                var q = row.Slice(h * 2 * headDim, headDim);
                DecoderKernels.RmsNorm(q, layer.QueryNorm, c.RmsNormEps, q);
                Rotate(q, cos, sin);
            }
            for (int h = 0; h < kvHeads; ++h)
            {
                var k = row.Slice(qWidth + h * headDim, headDim);
                DecoderKernels.RmsNorm(k, layer.KeyNorm, c.RmsNormEps, k);
                Rotate(k, cos, sin);
            }
            row.Slice(qWidth, kvWidth).CopyTo(keys[rowPiece[t]].AsSpan(pos * kvWidth, kvWidth));
            row.Slice(qWidth + kvWidth, kvWidth).CopyTo(values[rowPiece[t]].AsSpan(pos * kvWidth, kvWidth));
        });
        if (outputs.Length == 0) return;

        var attentionStage = ForwardTiming.Measure("full.attention");
        float scale = 1f / MathF.Sqrt(headDim);
        int groups = heads / kvHeads;
        int longest = pieces.Max(p => p.Position + p.Rows);
        Parallel.For(0, outputs.Length * heads, DecisionRuntime.Resolve(parallel), () => new float[longest], (item, _, scores) =>
        {
            int o = item / heads, t = outputs[o], h = item % heads, kvh = h / groups;
            var piece = pieces[rowPiece[t]];
            var sequenceKeys = keys[rowPiece[t]];
            var sequenceValues = values[rowPiece[t]];
            int pos = piece.Position + (t - piece.Row);
            var q = qkv.AsSpan(t * width + h * 2 * headDim, headDim);
            float max = float.NegativeInfinity;
            for (int j = 0; j <= pos; ++j)
            {
                float s = SimdOps.Dot(q, sequenceKeys.AsSpan(j * kvWidth + kvh * headDim, headDim)) * scale;
                scores[j] = s;
                if (s > max) max = s;
            }
            float sum = 0f;
            for (int j = 0; j <= pos; ++j)
            {
                float e = MathF.Exp(scores[j] - max);
                scores[j] = e;
                sum += e;
            }
            var output = attention.AsSpan((o * heads + h) * headDim, headDim);
            output.Clear();
            float inv = 1f / sum;
            for (int j = 0; j <= pos; ++j)
            {
                SimdOps.AddScaled(output, sequenceValues.AsSpan(j * kvWidth + kvh * headDim, headDim), scores[j] * inv);
            }
            DecoderKernels.MultiplySigmoid(output, qkv.AsSpan(t * width + h * 2 * headDim + headDim, headDim));
            return scores;
        }, _ => { });
        attentionStage.Dispose();

        using (ForwardTiming.Measure("full.o_proj")) Project(layer.Output!, attention, outputs.Length, projected, parallel);
        for (int o = 0; o < outputs.Length; ++o)
        {
            var source = projected.AsSpan(o * hidden, hidden);
            if (!record.IsEmpty) source.CopyTo(record.Slice(outputs[o] * hidden, hidden));
            SimdOps.Add(x.Slice(outputs[o] * hidden, hidden), source);
        }
    }

    /// <summary>After a full-attention layer: each checkpoint snapshot takes its sequence's keys and values up to it.</summary>
    private void SnapshotKeysAndValues(int index, IReadOnlyList<Qwen35Sequence> sequences)
    {
        int kvWidth = _config.NumKeyValueHeads * _config.HeadDim;
        foreach (var sequence in sequences)
        {
            for (int k = 0; k < sequence.Snapshots!.Length; ++k)
            {
                var snapshot = sequence.Snapshots[k];
                int length = (sequence.State.Length + sequence.Checkpoints![k]) * kvWidth;
                var keys = snapshot.GrowCache(ref snapshot.Keys[index]!, length, 0);
                var values = snapshot.GrowCache(ref snapshot.Values[index]!, length, 0);
                sequence.State.Keys[index].AsSpan(0, length).CopyTo(keys);
                sequence.State.Values[index].AsSpan(0, length).CopyTo(values);
            }
        }
    }

    /// <summary>cos/sin of <c>position · inv_freq</c>, the product taken in fp32 as transformers does.</summary>
    private void Rotation(int position, Span<float> cos, Span<float> sin)
    {
        for (int i = 0; i < cos.Length; ++i)
        {
            float angle = position * _inverseFrequency[i];
            cos[i] = MathF.Cos(angle);
            sin[i] = MathF.Sin(angle);
        }
    }

    /// <summary><c>rotate_half</c> RoPE on the first <c>2 · cos.Length</c> values; the rest pass through.</summary>
    private static void Rotate(Span<float> head, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = cos.Length;
        for (int i = 0; i < half; ++i)
        {
            float a = head[i], b = head[i + half];
            head[i] = a * cos[i] - b * sin[i];
            head[i + half] = b * cos[i] + a * sin[i];
        }
    }

    /// <summary>
    /// Gated DeltaNet: causal depthwise conv + SiLU over [q; k; v], L2-normalised q and k, then per value
    /// head the gated delta rule
    /// <code>
    /// S ← S · exp(g);  S ← S + k ⊗ β (v − Sᵀk);  o = Sᵀq
    /// </code>
    /// with <c>g = −exp(A_log) · softplus(a + dt_bias)</c> and <c>β = sigmoid(b)</c>; the output goes
    /// through <c>RMSNormGated(o, z)</c> and <c>out_proj</c>. Convolution and recurrence run per piece, on its
    /// sequence's own state; a checkpoint inside a piece splits its recurrence there and copies the state out.
    /// </summary>
    private void LinearAttention(Layer layer, int index, Span<float> x, int rows, List<Piece> pieces,
        Span<float> record, ParallelOptions? parallel)
    {
        var c = _config;
        int hidden = c.HiddenSize, convDim = c.LinearConvDim, kernel = c.LinearConvKernel;
        int keyDim = c.LinearKeyDim, valueDim = c.LinearValueDim;
        int vHeads = c.LinearNumValueHeads, kHeads = c.LinearNumKeyHeads, dk = c.LinearKeyHeadDim, dv = c.LinearValueHeadDim;
        int width = convDim + valueDim + 2 * vHeads;
        int tail = kernel - 1;

        using var scratch = new ScratchBuffers();
        int tiled = Tiled(rows);
        var normed = scratch.Rent(tiled * hidden);
        var projected = scratch.Rent(tiled * width);
        var history = scratch.Rent((pieces.Count * tail + rows) * convDim);   // per piece: its sequence's tail, then its rows
        var conv = scratch.Rent(rows * convDim);
        var decay = scratch.Rent(rows * vHeads);
        var beta = scratch.Rent(rows * vHeads);
        var core = scratch.Rent(tiled * valueDim);
        var output = scratch.Rent(tiled * hidden);

        for (int t = 0; t < rows; ++t)
        {
            DecoderKernels.RmsNorm(x.Slice(t * hidden, hidden), layer.InputNorm, c.RmsNormEps, normed.AsSpan(t * hidden, hidden));
        }
        using (ForwardTiming.Measure("linear.in_proj")) Project(layer.InProjection!, normed, rows, projected, parallel);

        // History row of chunk row t: piece p's block starts at (p * tail + piece.Row) rows.
        var historyRow = new int[rows];
        for (int p = 0; p < pieces.Count; ++p)
        {
            var piece = pieces[p];
            int block = p * tail + piece.Row;
            var convTail = piece.Sequence.State.ConvTail[index]!;
            convTail.AsSpan().CopyTo(history.AsSpan(block * convDim, tail * convDim));
            for (int t = 0; t < piece.Rows; ++t)
            {
                projected.AsSpan((piece.Row + t) * width, convDim).CopyTo(history.AsSpan((block + tail + t) * convDim, convDim));
                historyRow[piece.Row + t] = block + t;       // the first of its kernel taps
            }
            // Conv tails for the snapshots inside this piece, then the sequence's own.
            for (int k = 0; k < piece.Sequence.Snapshots!.Length; ++k)
            {
                int at = piece.Sequence.Checkpoints![k] - piece.Offset;
                if (at > 0 && at <= piece.Rows)
                {
                    history.AsSpan((block + at) * convDim, tail * convDim).CopyTo(piece.Sequence.Snapshots[k].ConvTail[index]);
                }
            }
            history.AsSpan((block + piece.Rows) * convDim, tail * convDim).CopyTo(convTail);
        }

        var weight = layer.ConvWeight!;
        var convStage = ForwardTiming.Measure("linear.conv");
        ForRows(rows, parallel, t =>
        {
            var destination = conv.AsSpan(t * convDim, convDim);
            int first = historyRow[t];
            // Vectorized across channels; per channel the taps are summed in order with a separate multiply and
            // add, exactly as the scalar loop does.
            int lanes = System.Numerics.Vector<float>.Count;
            int ch = 0;
            for (; ch <= convDim - lanes; ch += lanes)
            {
                var sum = System.Numerics.Vector<float>.Zero;
                for (int j = 0; j < kernel; ++j)
                {
                    var w = System.Numerics.Vector.LoadUnsafe(in weight[j * convDim + ch]);
                    var h = System.Numerics.Vector.LoadUnsafe(in history[(first + j) * convDim + ch]);
                    sum += w * h;
                }
                System.Numerics.Vector.StoreUnsafe(sum, ref destination[ch]);
            }
            for (; ch < convDim; ++ch)
            {
                float sum = 0f;
                for (int j = 0; j < kernel; ++j) sum += weight[j * convDim + ch] * history[(first + j) * convDim + ch];
                destination[ch] = sum;
            }
            DecoderKernels.Silu(destination);
            float queryScale = 1f / MathF.Sqrt(dk);
            for (int h = 0; h < kHeads; ++h)
            {
                DecoderKernels.L2Normalize(destination.Slice(h * dk, dk), 1e-6f, queryScale);
                DecoderKernels.L2Normalize(destination.Slice(keyDim + h * dk, dk), 1e-6f);
            }
            var row = projected.AsSpan(t * width, width);
            for (int h = 0; h < vHeads; ++h)
            {
                float b = row[convDim + valueDim + h];
                float a = row[convDim + valueDim + vHeads + h];
                beta[t * vHeads + h] = DecoderKernels.Sigmoid(b);
                // Some heads decay hard (exp(A_log) reaches 75 in Mica), and exp(g) then falls below the smallest
                // normal float. Multiplying 16k state values by a subnormal costs a microcode assist each: those few
                // heads took ten times as long as the rest and serialised the whole recurrence. S·g would be below
                // 1e-38 and vanish into the k ⊗ δ added to it, so flushing the decay to 0 changes no representable
                // result.
                float d = MathF.Exp(layer.NegativeA![h] * DecoderKernels.Softplus(a + layer.DtBias![h]));
                decay[t * vHeads + h] = float.IsSubnormal(d) ? 0f : d;
            }
        });
        convStage.Dispose();

        int ratio = vHeads / kHeads;
        var recurrenceStage = ForwardTiming.Measure("linear.recurrence");
        Parallel.For(0, pieces.Count * vHeads, DecisionRuntime.Resolve(parallel), item =>
        {
            var piece = pieces[item / vHeads];
            int h = item % vHeads;
            var stateRows = piece.Sequence.State.Recurrent[index]!.AsSpan(h * dk * dv, dk * dv);
            // Split the piece's rows at the checkpoints that fall inside it, copying the state out at each.
            int done = 0;
            for (int k = 0; k <= piece.Sequence.Snapshots!.Length; ++k)
            {
                bool snapshot = k < piece.Sequence.Snapshots.Length;
                int until = snapshot ? piece.Sequence.Checkpoints![k] - piece.Offset : piece.Rows;
                if (until <= done || until > piece.Rows) continue;
                Recurrence(stateRows, piece.Row + done, until - done, h, h / ratio, conv, decay, beta, core);
                done = until;
                if (snapshot) stateRows.CopyTo(piece.Sequence.Snapshots[k].Recurrent[index].AsSpan(h * dk * dv, dk * dv));
            }
            if (done < piece.Rows) Recurrence(stateRows, piece.Row + done, piece.Rows - done, h, h / ratio, conv, decay, beta, core);
        });
        recurrenceStage.Dispose();

        using (ForwardTiming.Measure("linear.gated_norm")) ForRows(rows, parallel, t =>
        {
            var z = projected.AsSpan(t * width + convDim, valueDim);
            for (int h = 0; h < vHeads; ++h)
            {
                var o = core.AsSpan(t * valueDim + h * dv, dv);
                DecoderKernels.RmsNormGated(o, z.Slice(h * dv, dv), layer.GatedNorm, c.RmsNormEps, o);
            }
        });
        using (ForwardTiming.Measure("linear.out_proj")) Project(layer.OutProjection!, core, rows, output, parallel);
        if (!record.IsEmpty) output.AsSpan(0, rows * hidden).CopyTo(record);
        SimdOps.Add(x, output.AsSpan(0, rows * hidden));
    }

    /// <summary>The gated delta rule for value head <paramref name="h"/> over chunk rows <c>[first, first + count)</c>.</summary>
    private void Recurrence(Span<float> state, int first, int count, int h, int kh, float[] conv, float[] decay, float[] beta, float[] core)
    {
        var c = _config;
        int convDim = c.LinearConvDim, keyDim = c.LinearKeyDim, valueDim = c.LinearValueDim, vHeads = c.LinearNumValueHeads;
        int dk = c.LinearKeyHeadDim, dv = c.LinearValueHeadDim;
        if (dv == 128 && SimdOps.UseVector512)
        {
            unsafe
            {
                fixed (float* sp = state, cp = conv, dp = decay, bp = beta, op = core)
                {
                    float* row = cp + (long)first * convDim;
                    DecoderKernels.GatedDeltaRule128(sp, dk, count, row + kh * dk, row + keyDim + kh * dk, row + 2 * keyDim + h * dv, convDim,
                        dp + (long)first * vHeads + h, bp + (long)first * vHeads + h, vHeads, op + (long)first * valueDim + h * dv, valueDim);
                }
            }
            return;
        }

        Span<float> memory = stackalloc float[dv];
        Span<float> delta = stackalloc float[dv];
        for (int t = first; t < first + count; ++t)
        {
            var row = conv.AsSpan(t * convDim, convDim);
            var q = row.Slice(kh * dk, dk);
            var k = row.Slice(keyDim + kh * dk, dk);
            var v = row.Slice(2 * keyDim + h * dv, dv);
            float g = decay[t * vHeads + h];
            float b = beta[t * vHeads + h];

            memory.Clear();
            for (int i = 0; i < dk; ++i)
            {
                var si = state.Slice(i * dv, dv);
                SimdOps.Scale(si, g);
                SimdOps.AddScaled(memory, si, k[i]);
            }
            for (int j = 0; j < dv; ++j) delta[j] = (v[j] - memory[j]) * b;

            var o = core.AsSpan(t * valueDim + h * dv, dv);
            o.Clear();
            for (int i = 0; i < dk; ++i)
            {
                var si = state.Slice(i * dv, dv);
                SimdOps.AddScaled(si, delta, k[i]);
                SimdOps.AddScaled(o, si, q[i]);
            }
        }
    }

    /// <summary>
    /// Rows rounded up to the GEMM's register tile. The kernel computes six token rows per pass and finishes a
    /// ragged remainder one row at a time at about a third of the speed, so 173 rows (28 tiles and 5 single rows)
    /// cost as much as 180. Projecting a few zero rows extra and discarding them is cheaper, and changes no real row.
    /// </summary>
    private static int Tiled(int rows) => (rows + PackedMatrixBf16.RowTile - 1) / PackedMatrixBf16.RowTile * PackedMatrixBf16.RowTile;

    /// <summary>
    /// <c>output = input · Wᵀ</c> over <paramref name="rows"/> rows, run on <see cref="Tiled"/> rows: the padding rows of
    /// <paramref name="input"/> are zeroed (rented scratch holds anything, NaN included) and their outputs ignored.
    /// </summary>
    private static void Project(ILinear weight, float[] input, int rows, float[] output, ParallelOptions? parallel)
    {
        int tiled = Tiled(rows);
        input.AsSpan(rows * weight.InFeatures, (tiled - rows) * weight.InFeatures).Clear();
        weight.Multiply(input.AsSpan(0, tiled * weight.InFeatures), tiled, [], output.AsSpan(0, tiled * weight.OutFeatures), parallel);
    }

    private static void ForRows(int rows, ParallelOptions? parallel, Action<int> body)
    {
        if (rows < 4 || DecisionRuntime.WorkersOf(parallel) <= 1)
        {
            for (int t = 0; t < rows; ++t) body(t);
            return;
        }
        Parallel.For(0, rows, DecisionRuntime.Resolve(parallel), body);
    }

    /// <summary>
    /// The embedding table (and, tied, the LM head): bf16 when the checkpoint is, or ggml Q8_0 rows — looked up
    /// dequantized (ggml's <c>get_rows</c>), multiplied against a Q8_0-quantized hidden state (ggml's <c>mul_mat</c>).
    /// </summary>
    private sealed class TokenTable
    {
        private readonly ushort[]? _bf16;
        private readonly float[]? _fp32;
        private readonly sbyte[]? _q8;
        private readonly float[]? _q8Scales;
        private readonly int _rows, _width;

        private TokenTable(ushort[]? bf16, float[]? fp32, sbyte[]? q8, float[]? q8Scales, int rows, int width)
        {
            _bf16 = bf16;
            _fp32 = fp32;
            _q8 = q8;
            _q8Scales = q8Scales;
            _rows = rows;
            _width = width;
        }

        public bool IsQuantized => _q8 is not null;

        public long Bytes => _q8 is not null ? _q8.LongLength + _q8Scales!.LongLength * 4
            : _bf16 is not null ? _bf16.LongLength * 2 : _fp32!.LongLength * 4;

        public static TokenTable Load(SafetensorsCheckpoint weights, string name, int width, WeightPrecision precision)
        {
            var entry = weights.Entry(name);
            if (entry.Shape.Length != 2 || entry.Shape[1] != width)
            {
                throw new InvalidDataException($"{name}: expected [*, {width}], found [{string.Join(", ", entry.Shape)}].");
            }
            int rows = entry.Shape[0];
            if (precision == WeightPrecision.Q8_0) return Quantize(weights, name, entry.DType == "BF16", rows, width);
            return entry.DType == "BF16"
                ? new TokenTable(weights.ReadBFloat16Bits(name), null, null, null, rows, width)
                : new TokenTable(null, weights.ReadFloat32(name), null, null, rows, width);
        }

        private static TokenTable Quantize(SafetensorsCheckpoint weights, string name, bool bf16, int rows, int width)
        {
            int blocks = width / Q8_0.BlockSize;
            var q = new sbyte[(long)rows * width];
            var scales = new float[(long)rows * blocks];
            ushort[]? bits = bf16 ? weights.ReadBFloat16Bits(name) : null;
            float[]? values = bf16 ? null : weights.ReadFloat32(name);
            Parallel.For(0, rows, r =>
            {
                Span<float> row = stackalloc float[width];
                if (bits is not null)
                {
                    var source = bits.AsSpan((int)((long)r * width), width);
                    for (int i = 0; i < width; ++i) row[i] = BitConverter.UInt32BitsToSingle((uint)source[i] << 16);
                }
                else values.AsSpan((int)((long)r * width), width).CopyTo(row);
                for (int b = 0; b < blocks; ++b)
                {
                    scales[(long)r * blocks + b] = Q8_0.QuantizeWeightBlock(row.Slice(b * Q8_0.BlockSize, Q8_0.BlockSize),
                        q.AsSpan((int)((long)r * width + b * Q8_0.BlockSize), Q8_0.BlockSize));
                }
            });
            return new TokenTable(null, null, q, scales, rows, width);
        }

        public void Row(int id, Span<float> destination)
        {
            if ((uint)id >= (uint)_rows) throw new ArgumentOutOfRangeException(nameof(id), $"token id {id} is outside the vocabulary.");
            if (_q8 is not null)
            {
                int blocks = _width / Q8_0.BlockSize;
                for (int i = 0; i < _width; ++i)
                {
                    destination[i] = _q8[(long)id * _width + i] * _q8Scales![(long)id * blocks + i / Q8_0.BlockSize];
                }
                return;
            }
            if (_fp32 is not null)
            {
                _fp32.AsSpan(id * _width, _width).CopyTo(destination);
                return;
            }
            var source = _bf16.AsSpan((int)((long)id * _width), _width);
            for (int i = 0; i < _width; ++i) destination[i] = BitConverter.UInt32BitsToSingle((uint)source[i] << 16);
        }

        /// <summary><c>hidden · E[id]</c> as ggml's Q8_0 <c>mul_mat</c> takes it: per block an exact integer dot, scaled by
        /// <c>d_w · d_x</c>, summed in fp32 — the operations <see cref="PackedMatrixQ8_0"/> does.</summary>
        public void QuantizedLogits(ReadOnlySpan<float> hidden, ReadOnlySpan<int> ids, Span<float> logits)
        {
            int blocks = _width / Q8_0.BlockSize;
            var qx = new byte[_width];
            var dx = new float[blocks];
            Q8_0.QuantizeActivations(hidden, 1, _width, _width, qx, dx);
            for (int i = 0; i < ids.Length; ++i)
            {
                int id = ids[i];
                if ((uint)id >= (uint)_rows) throw new ArgumentOutOfRangeException(nameof(ids), $"token id {id} is outside the vocabulary.");
                float acc = 0f;
                for (int b = 0; b < blocks; ++b)
                {
                    int sum = 0;
                    long w = (long)id * _width + b * Q8_0.BlockSize;
                    for (int t = 0; t < Q8_0.BlockSize; ++t) sum += (qx[b * Q8_0.BlockSize + t] - 128) * _q8![w + t];
                    acc = MathF.FusedMultiplyAdd(sum, _q8Scales![(long)id * blocks + b] * dx[b], acc);
                }
                logits[i] = acc;
            }
        }
    }
}
