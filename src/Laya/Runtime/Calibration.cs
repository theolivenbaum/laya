namespace Laya.Runtime;

/// <summary>
/// The calibration and scoring helpers from <c>.reference/laya/common.py</c>.
///
/// <para><see cref="ProperReward"/> and <see cref="TdLambdaTargets"/> belong to training rather
/// than inference; they are ported because the Python package exports them, and because they
/// document what the reported probabilities are supposed to mean.</para>
/// </summary>
public static class Calibration
{
    /// <summary>Normalized Shannon entropy confidence: <c>1 - H(p) / log(k)</c>.</summary>
    public static double ConfidenceFromProbabilities(ReadOnlySpan<float> probabilities, int k)
    {
        if (k < 2) return 1d;
        double entropy = 0d;
        for (int i = 0; i < k && i < probabilities.Length; ++i)
        {
            double p = Math.Clamp(probabilities[i], 1e-12, 1.0);
            entropy -= probabilities[i] * Math.Log(p);
        }
        return Math.Clamp(1d - entropy / Math.Log(k), 0d, 1d);
    }

    /// <summary>Expected Calibration Error over equal-width confidence bins.</summary>
    public static double ExpectedCalibrationError(ReadOnlySpan<double> confidence, ReadOnlySpan<bool> correct, int bins = 15)
    {
        if (confidence.Length == 0) return double.NaN;
        if (confidence.Length != correct.Length)
        {
            throw new ArgumentException("confidence and correct must be the same length.", nameof(correct));
        }

        double error = 0d;
        for (int bin = 0; bin < bins; ++bin)
        {
            double low = (double)bin / bins;
            double high = (double)(bin + 1) / bins;

            int count = 0;
            double confidenceSum = 0d;
            double correctSum = 0d;
            for (int i = 0; i < confidence.Length; ++i)
            {
                if (confidence[i] <= low || confidence[i] > high) continue;
                count++;
                confidenceSum += confidence[i];
                if (correct[i]) correctSum += 1d;
            }
            if (count == 0) continue;

            double share = (double)count / confidence.Length;
            error += share * Math.Abs(confidenceSum / count - correctSum / count);
        }
        return error;
    }

    /// <summary>
    /// Strictly proper scoring rule: log score + spherical score, minus a ranked probability score
    /// for ordinal <c>score</c> questions.
    /// </summary>
    public static double ProperReward(ReadOnlySpan<float> reported, ReadOnlySpan<float> target,
        ReadOnlySpan<bool> mask, QuestionType type, double sphericalWeight = 0.5, double rpsWeight = 1.0,
        double logFloor = -9.21)
    {
        int n = reported.Length;
        Span<double> q = n <= 64 ? stackalloc double[n] : new double[n];
        for (int i = 0; i < n; ++i) q[i] = mask.Length > i && !mask[i] ? 0d : reported[i];

        double logScore = 0d;
        double dot = 0d;
        double normSquared = 0d;
        for (int i = 0; i < n; ++i)
        {
            double logQ = Math.Max(Math.Log(Math.Max(q[i], 1e-12)), logFloor);
            logScore += target[i] * logQ;
            dot += target[i] * q[i];
            normSquared += q[i] * q[i];
        }

        double reward = logScore + sphericalWeight * (dot / Math.Max(Math.Sqrt(normSquared), 1e-9));
        if (type != QuestionType.Score) return reward;

        int k = 0;
        for (int i = 0; i < n; ++i)
        {
            if (mask.Length <= i || mask[i]) k++;
        }
        k = Math.Max(k, 2);

        double cumulativeQ = 0d;
        double cumulativeTarget = 0d;
        double rps = 0d;
        for (int i = 0; i < n; ++i)
        {
            cumulativeQ += q[i];
            cumulativeTarget += target[i];
            if (mask.Length <= i || mask[i])
            {
                double delta = cumulativeQ - cumulativeTarget;
                rps += delta * delta;
            }
        }
        return reward - rpsWeight * rps / (k - 1);
    }

    /// <summary>
    /// TD(λ) targets over a multi-turn trajectory: walks the episode backwards, blending the
    /// bootstrapped value of the next step into the return.
    /// </summary>
    public static double[] TdLambdaTargets(ReadOnlySpan<double> trueProbabilities, double finalOutcome, double lambda = 1.0)
    {
        int steps = trueProbabilities.Length;
        var targets = new double[steps];
        if (steps == 0) return targets;

        double g = finalOutcome;
        for (int j = steps - 1; j >= 0; --j)
        {
            if (j < steps - 1) g = (1 - lambda) * trueProbabilities[j + 1] + lambda * g;
            targets[j] = g;
        }
        return targets;
    }
}
