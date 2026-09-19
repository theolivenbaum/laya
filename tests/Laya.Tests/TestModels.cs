using Xunit;

namespace Laya.Tests;

/// <summary>
/// Locates the checkpoints a test needs. Weights are a gigabyte-scale download, so tests that need
/// them skip rather than fail when <c>artifacts/models/</c> is empty; <c>laya download</c> or
/// <c>LAYA_TEST_MODELS</c> fills it in.
/// </summary>
internal static class TestModels
{
    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    /// <summary>Where checkpoints are looked for: <c>$LAYA_TEST_MODELS</c> or <c>artifacts/models</c>.</summary>
    public static string ModelRoot { get; } =
        Environment.GetEnvironmentVariable("LAYA_TEST_MODELS")
        ?? Path.Combine(RepositoryRoot, "artifacts", "models");

    public static string FixtureRoot { get; } = ResolveFixtureRoot();

    public static string CheckpointDirectory(string checkpoint) => Path.Combine(ModelRoot, checkpoint);

    private static string ResolveFixtureRoot()
    {
        string built = Path.Combine(AppContext.BaseDirectory, "Fixtures");
        return Directory.Exists(built) ? built : Path.Combine(RepositoryRoot, "tests", "Laya.Tests", "Fixtures");
    }

    public static bool Available(string checkpoint)
        => File.Exists(Path.Combine(CheckpointDirectory(checkpoint), "model.safetensors"))
        && File.Exists(Path.Combine(CheckpointDirectory(checkpoint), "rl_agent_config.json"));

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, ".reference"))
                || Directory.Exists(Path.Combine(directory.FullName, ".git")))
            {
                return directory.FullName;
            }
            directory = directory.Parent;
        }
        return Environment.CurrentDirectory;
    }
}

/// <summary>A fact that skips itself when the checkpoint it needs is not on disk.</summary>
internal sealed class ModelFactAttribute : FactAttribute
{
    public ModelFactAttribute(string checkpoint = "english")
    {
        if (!TestModels.Available(checkpoint))
        {
            Skip = $"checkpoint '{checkpoint}' is not in {TestModels.ModelRoot}; " +
                   $"run: dotnet run --project src/Laya.Cli -- download --model {checkpoint} " +
                   "--cache artifacts/models-cache";
        }
    }
}
