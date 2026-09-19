using Laya.Runtime;
using Xunit;

namespace Laya.Tests;

/// <summary>The calibration and scoring helpers, which are what "calibrated probabilities" means.</summary>
public class CalibrationTests
{
    [Fact]
    public void ConfidenceIsOneForACertainPrediction()
        => Assert.Equal(1d, Calibration.ConfidenceFromProbabilities([1f, 0f, 0f, 0f], 4), 6);

    [Fact]
    public void ConfidenceIsZeroForAUniformPrediction()
        => Assert.Equal(0d, Calibration.ConfidenceFromProbabilities([0.25f, 0.25f, 0.25f, 0.25f], 4), 6);

    [Fact]
    public void ConfidenceIsOneWhenThereIsNothingToChoose()
        => Assert.Equal(1d, Calibration.ConfidenceFromProbabilities([1f], 1));

    [Fact]
    public void ConfidenceFallsAsTheDistributionFlattens()
    {
        double sharp = Calibration.ConfidenceFromProbabilities([0.9f, 0.1f], 2);
        double soft = Calibration.ConfidenceFromProbabilities([0.6f, 0.4f], 2);
        Assert.True(sharp > soft);
        Assert.InRange(soft, 0d, 1d);
    }

    [Fact]
    public void PerfectCalibrationHasZeroError()
    {
        // Every prediction is made with confidence 1 and every one is right.
        double[] confidence = [1, 1, 1, 1];
        bool[] correct = [true, true, true, true];
        Assert.Equal(0d, Calibration.ExpectedCalibrationError(confidence, correct), 6);
    }

    [Fact]
    public void OverconfidenceShowsUpAsError()
    {
        double[] confidence = [0.95, 0.95, 0.95, 0.95];
        bool[] correct = [true, false, false, false];
        Assert.Equal(0.7, Calibration.ExpectedCalibrationError(confidence, correct), 2);
    }

    [Fact]
    public void EmptyCalibrationSetIsNotANumber()
        => Assert.True(double.IsNaN(Calibration.ExpectedCalibrationError([], [])));

    [Fact]
    public void ProperRewardPrefersTheTruth()
    {
        float[] target = [0f, 1f];
        bool[] mask = [true, true];
        double right = Calibration.ProperReward([0.1f, 0.9f], target, mask, QuestionType.Noul);
        double wrong = Calibration.ProperReward([0.9f, 0.1f], target, mask, QuestionType.Noul);
        Assert.True(right > wrong);
    }

    [Fact]
    public void ProperRewardPenalisesDistantOrdinalMisses()
    {
        // For an ordinal question, being wrong by three levels must cost more than being wrong by one.
        float[] target = [0f, 0f, 0f, 1f];
        bool[] mask = [true, true, true, true];
        double near = Calibration.ProperReward([0f, 0f, 0.5f, 0.5f], target, mask, QuestionType.Score);
        double far = Calibration.ProperReward([0.5f, 0f, 0f, 0.5f], target, mask, QuestionType.Score);
        Assert.True(near > far);

        // A choice question has no order, so the ranked term must not apply.
        double nearChoice = Calibration.ProperReward([0f, 0f, 0.5f, 0.5f], target, mask, QuestionType.Choice);
        double farChoice = Calibration.ProperReward([0.5f, 0f, 0f, 0.5f], target, mask, QuestionType.Choice);
        Assert.Equal(nearChoice, farChoice, 6);
    }

    [Fact]
    public void TdLambdaWithLambdaOneIsTheFinalOutcomeEverywhere()
    {
        var targets = Calibration.TdLambdaTargets([0.2, 0.4, 0.6], finalOutcome: 1.0, lambda: 1.0);
        Assert.Equal([1.0, 1.0, 1.0], targets);
    }

    [Fact]
    public void TdLambdaWithLambdaZeroBootstrapsFromTheNextStep()
    {
        var targets = Calibration.TdLambdaTargets([0.2, 0.4, 0.6], finalOutcome: 1.0, lambda: 0.0);
        Assert.Equal(1.0, targets[2], 6);        // last step has nothing to bootstrap from
        Assert.Equal(0.6, targets[1], 6);
        Assert.Equal(0.4, targets[0], 6);
    }

    [Fact]
    public void TdLambdaOnAnEmptyTrajectoryIsEmpty()
        => Assert.Empty(Calibration.TdLambdaTargets([], 1.0));
}
