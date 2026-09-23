using Laya.Numerics;
using Laya.Runtime;

namespace Laya.Training;

/// <summary>The loss of one micro-batch and its gradient with respect to every option logit.</summary>
public sealed record ObjectiveResult(double Loss, double PolicyLoss, double CrossEntropy, double MeanReward,
    float[][] LogitGradients);

/// <summary>
/// The training objective of <c>train_ddp.py</c> in the reference notebook: a group-relative policy
/// gradient over noisy logits, scored by a strictly proper reward, plus soft cross-entropy guidance.
///
/// <list type="number">
/// <item>Sample <see cref="GroupSize"/> Gaussian perturbations of each item's logits (σ from the
/// schedule), projected to zero mean over the item's options, and softmax them.</item>
/// <item>Score each sample with <see cref="Calibration.ProperReward"/> (log + spherical score, minus a
/// ranked probability score for <c>score</c> questions) against the soft target.</item>
/// <item>Advantage = reward minus the group mean per item, divided by the standard deviation over the
/// whole batch.</item>
/// <item>Loss = <c>−mean(adv · log π(z))</c> with <c>log π(z) = −Σ (z − l)² / 2σ²</c>, plus
/// <see cref="CrossEntropyWeight"/> × the soft cross-entropy of the unperturbed logits.</item>
/// </list>
///
/// <para>The gradients are analytic: <c>∂ log π / ∂l = ε / σ²</c> for the policy term, and
/// <c>(softmax(l) − t) / N</c> for the cross-entropy.</para>
/// </summary>
public sealed class RlcdObjective
{
    public int GroupSize { get; init; } = 4;
    public double SphericalWeight { get; init; } = 0.75;
    public double RpsWeight { get; init; } = 1.0;
    public double CrossEntropyWeight { get; init; } = 1.0;

    /// <param name="scale">Multiplies the gradients (the reference divides the loss by the accumulation steps).</param>
    public ObjectiveResult Compute(IReadOnlyList<TrainingItem> items, IReadOnlyList<float[]> logits, double sigma,
        Random random, double scale = 1.0)
    {
        var noise = new double[GroupSize][][];
        for (int g = 0; g < GroupSize; ++g)
        {
            noise[g] = new double[items.Count][];
            for (int i = 0; i < items.Count; ++i)
            {
                noise[g][i] = new double[items[i].Options];
                for (int o = 0; o < items[i].Options; ++o) noise[g][i][o] = Gaussian(random);
            }
        }
        return Compute(items, logits, sigma, noise, scale);
    }

    /// <summary>
    /// The same with the standard-normal draws supplied, <c>[group][item][option]</c> — before the
    /// σ scaling and the zero-mean projection, exactly where <c>torch.randn</c> sits in the reference.
    /// </summary>
    public ObjectiveResult Compute(IReadOnlyList<TrainingItem> items, IReadOnlyList<float[]> logits, double sigma,
        double[][][] standardNormal, double scale = 1.0)
    {
        int n = items.Count;
        int groups = standardNormal.Length;
        var gradients = new float[n][];
        var epsilon = new double[groups][][];
        var reward = new double[groups, n];

        for (int g = 0; g < groups; ++g)
        {
            epsilon[g] = new double[n][];
            for (int i = 0; i < n; ++i)
            {
                int k = items[i].Options;
                var e = new double[k];
                double mean = 0d;
                for (int o = 0; o < k; ++o)
                {
                    e[o] = standardNormal[g][i][o] * sigma;
                    mean += e[o];
                }
                mean /= k;
                for (int o = 0; o < k; ++o) e[o] -= mean;
                epsilon[g][i] = e;

                var q = new float[k];
                for (int o = 0; o < k; ++o) q[o] = (float)(logits[i][o] + e[o]);
                SimdOps.Softmax(q);
                var mask = new bool[k];
                Array.Fill(mask, true);
                reward[g, i] = Calibration.ProperReward(q, items[i].Target, mask, (QuestionType)items[i].QuestionType,
                    SphericalWeight, RpsWeight);
            }
        }

        // Group-relative advantage, normalised by the (unbiased) standard deviation over all samples.
        var advantage = new double[groups, n];
        double total = 0d;
        for (int i = 0; i < n; ++i)
        {
            double mean = 0d;
            for (int g = 0; g < groups; ++g) mean += reward[g, i];
            mean /= groups;
            for (int g = 0; g < groups; ++g)
            {
                advantage[g, i] = reward[g, i] - mean;
                total += advantage[g, i];
            }
        }
        int count = groups * n;
        double globalMean = total / count;
        double variance = 0d;
        for (int g = 0; g < groups; ++g)
        {
            for (int i = 0; i < n; ++i) variance += (advantage[g, i] - globalMean) * (advantage[g, i] - globalMean);
        }
        double std = count > 1 ? Math.Sqrt(variance / (count - 1)) : 0d;
        for (int g = 0; g < groups; ++g)
        {
            for (int i = 0; i < n; ++i) advantage[g, i] /= std + 1e-6;
        }

        double policyLoss = 0d, crossEntropy = 0d, rewardSum = 0d;
        double sigmaSquared = sigma * sigma;
        for (int i = 0; i < n; ++i)
        {
            int k = items[i].Options;
            var gradient = new double[k];
            for (int g = 0; g < groups; ++g)
            {
                double logProbability = 0d;
                for (int o = 0; o < k; ++o)
                {
                    double e = epsilon[g][i][o];
                    logProbability -= e * e / (2 * sigmaSquared);
                    gradient[o] -= advantage[g, i] * e / sigmaSquared / count;
                }
                policyLoss -= advantage[g, i] * logProbability / count;
                rewardSum += reward[g, i];
            }

            var p = logits[i].AsSpan(0, k).ToArray();
            SimdOps.Softmax(p);
            float[] target = items[i].Target;
            double targetMass = 0d;
            for (int o = 0; o < k; ++o) targetMass += target[o];
            for (int o = 0; o < k; ++o)
            {
                crossEntropy -= target[o] * Math.Log(Math.Max(p[o], 1e-30)) / n;
                gradient[o] += CrossEntropyWeight * (p[o] * targetMass - target[o]) / n;
            }

            gradients[i] = [.. gradient.Select(v => (float)(v * scale))];
        }

        return new ObjectiveResult(policyLoss + CrossEntropyWeight * crossEntropy, policyLoss, crossEntropy,
            rewardSum / count, gradients);
    }

    private static double Gaussian(Random random)
    {
        // Box–Muller; 1 - NextDouble() keeps the logarithm's argument in (0, 1].
        double u1 = 1d - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2d * Math.Log(u1)) * Math.Cos(2d * Math.PI * u2);
    }
}
