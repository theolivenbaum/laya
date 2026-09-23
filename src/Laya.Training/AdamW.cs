namespace Laya.Training;

/// <summary>A set of parameters sharing one learning rate.</summary>
public sealed class ParameterGroup(IReadOnlyList<Parameter> parameters, double learningRate)
{
    public IReadOnlyList<Parameter> Parameters { get; } = parameters;

    /// <summary>The rate the schedule starts from.</summary>
    public double BaseLearningRate { get; } = learningRate;

    /// <summary>The rate the next step uses; the schedule writes it.</summary>
    public double LearningRate { get; set; } = learningRate;
}

/// <summary>
/// <c>torch.optim.AdamW</c> with its defaults: decoupled weight decay applied before the Adam update,
/// bias-corrected moments, and <c>ε</c> added to the corrected second moment's square root. A
/// parameter with no gradient this step is updated as if its gradient were zero — which is what the
/// reference's <c>0.0 * act.sum()</c> produces for the action head, so weight decay still reaches it.
/// </summary>
public sealed class AdamW(IReadOnlyList<ParameterGroup> groups, double beta1 = 0.9, double beta2 = 0.999,
    double epsilon = 1e-8, double weightDecay = 0.01)
{
    private readonly Dictionary<Parameter, (float[] M, float[] V)> _state = [];

    public IReadOnlyList<ParameterGroup> Groups { get; } = groups;
    public int Steps { get; private set; }

    public void Step(ParallelOptions? parallel = null)
    {
        var options = LayaRuntime.Resolve(parallel);
        Steps++;
        double correction1 = 1d - Math.Pow(beta1, Steps);
        double correction2 = 1d - Math.Pow(beta2, Steps);
        float b1 = (float)beta1, b2 = (float)beta2;

        foreach (var group in Groups)
        {
            float decay = (float)(1d - group.LearningRate * weightDecay);
            float stepSize = (float)(group.LearningRate / correction1);
            float sqrtCorrection2 = (float)Math.Sqrt(correction2);
            float eps = (float)epsilon;

            foreach (var parameter in group.Parameters)
            {
                if (!parameter.Trainable) continue;
                if (!_state.TryGetValue(parameter, out var moments))
                {
                    moments = (new float[parameter.Length], new float[parameter.Length]);
                    _state[parameter] = moments;
                }
                float[] data = parameter.Data;
                float[]? gradient = parameter.Gradient;
                var (m, v) = moments;

                const int chunk = 1 << 15;
                int chunks = (data.Length + chunk - 1) / chunk;
                Parallel.For(0, chunks, options, c =>
                {
                    int start = c * chunk, end = Math.Min(data.Length, start + chunk);
                    for (int i = start; i < end; ++i)
                    {
                        float g = gradient is null ? 0f : gradient[i];
                        data[i] *= decay;
                        m[i] = b1 * m[i] + (1f - b1) * g;
                        v[i] = b2 * v[i] + (1f - b2) * g * g;
                        data[i] -= stepSize * m[i] / (MathF.Sqrt(v[i]) / sqrtCorrection2 + eps);
                    }
                });
                parameter.Invalidate();
            }
        }
    }

    /// <summary>Bytes the moment estimates occupy.</summary>
    public long StateBytes => _state.Values.Sum(s => 2L * s.M.Length * sizeof(float));
}

/// <summary><c>CosineAnnealingLR</c>: <c>η_min + (η_base − η_min)(1 + cos(π t / T)) / 2</c>.</summary>
public sealed class CosineSchedule(AdamW optimizer, int totalSteps, double minimum = 1e-6)
{
    private int _step;

    public int TotalSteps { get; } = Math.Max(1, totalSteps);

    /// <summary>Advances one step and sets every group's rate, as <c>scheduler.step()</c> does.</summary>
    public void Step()
    {
        _step++;
        foreach (var group in optimizer.Groups) group.LearningRate = At(group.BaseLearningRate, _step);
    }

    public double At(double baseRate, int step) => minimum + (baseRate - minimum) * (1 + Math.Cos(Math.PI * step / TotalSteps)) / 2;
}

/// <summary>Gradient utilities over a whole model.</summary>
public static class Gradients
{
    /// <summary>
    /// <c>clip_grad_norm_</c>: scales every gradient by <c>max / (‖g‖ + 1e-6)</c> when the global L2
    /// norm exceeds <paramref name="maxNorm"/>. Returns the norm before clipping.
    /// </summary>
    public static double ClipByGlobalNorm(IEnumerable<Parameter> parameters, double maxNorm)
    {
        var withGradients = parameters.Where(p => p.Trainable && p.Gradient is not null).ToList();
        double sumSquares = 0d;
        var gate = new Lock();
        foreach (var parameter in withGradients)
        {
            float[] gradient = parameter.Gradient!;
            const int chunk = 1 << 16;
            Parallel.For(0, (gradient.Length + chunk - 1) / chunk, () => 0d, (c, _, local) =>
            {
                int end = Math.Min(gradient.Length, (c + 1) * chunk);
                for (int i = c * chunk; i < end; ++i) local += (double)gradient[i] * gradient[i];
                return local;
            }, local =>
            {
                lock (gate) sumSquares += local;
            });
        }
        double norm = Math.Sqrt(sumSquares);
        double coefficient = maxNorm / (norm + 1e-6);
        if (coefficient < 1d)
        {
            float scale = (float)coefficient;
            foreach (var parameter in withGradients) Numerics.SimdOps.Scale(parameter.Gradient!, scale);
        }
        return norm;
    }
}
