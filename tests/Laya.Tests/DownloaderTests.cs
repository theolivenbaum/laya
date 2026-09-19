using Laya.Io;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// The checkpoint file filter, which is what keeps loading one checkpoint from costing the whole
/// 2.3 GB bundle.
///
/// <para>The three checkpoints are published twice: bundled in <c>convaiinnovations/laya</c>, where
/// <c>multilingual/</c> and <c>typed-decisions/</c> are subfolders, and each in its own repo
/// (<c>laya-multilingual</c>, <c>laya-typed-decisions</c>) where the same five files sit at the
/// root. Both layouts go through the same filter, so both are pinned here.</para>
/// </summary>
public class DownloaderTests
{
    /// <summary>The bundle repo's real file list, as the hub API returns it.</summary>
    private static readonly string[] BundleFiles =
    [
        ".gitattributes", "README.md",
        "assets/logo-mark.png", "email_utils.py", "rl_agent_api.py", "rl_common.py",
        "eval/results.json", "eval/results.md", "eval/benchmark_comparison.png",
        "encoder/config.json", "model.safetensors", "rl_agent_config.json",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        "multilingual/encoder/config.json", "multilingual/model.safetensors",
        "multilingual/rl_agent_config.json", "multilingual/tokenizer/tokenizer.json",
        "multilingual/tokenizer/tokenizer_config.json",
        "typed-decisions/encoder/config.json", "typed-decisions/model.safetensors",
        "typed-decisions/rl_agent_config.json", "typed-decisions/tokenizer/tokenizer.json",
        "typed-decisions/tokenizer/tokenizer_config.json",
    ];

    /// <summary>A standalone checkpoint repo, e.g. <c>convaiinnovations/laya-typed-decisions</c>.</summary>
    private static readonly string[] StandaloneFiles =
    [
        ".gitattributes", "README.md",
        "encoder/config.json", "model.safetensors", "rl_agent_config.json",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    ];

    private static readonly string[] ExpectedRoot =
    [
        "encoder/config.json", "model.safetensors", "rl_agent_config.json",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    ];

    [Fact]
    public void TheRootCheckpointDoesNotDragInTheOtherTwo()
    {
        var filter = HuggingFaceDownloader.CheckpointFilter(null);
        string[] selected = [.. BundleFiles.Where(filter)];

        Assert.Equal(ExpectedRoot.Order(), selected.Order());
        // The names at the root are prefixes of the subfolder copies; a sloppy match takes 2.3 GB.
        Assert.DoesNotContain(selected, f => f.StartsWith("multilingual/", StringComparison.Ordinal));
        Assert.DoesNotContain(selected, f => f.StartsWith("typed-decisions/", StringComparison.Ordinal));
        Assert.DoesNotContain(selected, f => f.EndsWith(".py", StringComparison.Ordinal));
        Assert.DoesNotContain(selected, f => f.StartsWith("assets/", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("multilingual")]
    [InlineData("typed-decisions")]
    public void ASubfolderCheckpointTakesOnlyItsOwnFiles(string subfolder)
    {
        var filter = HuggingFaceDownloader.CheckpointFilter(subfolder);
        string[] selected = [.. BundleFiles.Where(filter)];

        Assert.Equal([.. ExpectedRoot.Select(f => subfolder + "/" + f).Order()], selected.Order());
    }

    [Fact]
    public void ATrailingSlashOnTheSubfolderIsAccepted()
    {
        var filter = HuggingFaceDownloader.CheckpointFilter("multilingual/");
        Assert.Equal(5, BundleFiles.Count(filter));
    }

    [Fact]
    public void AStandaloneRepoLooksLikeTheRootOfTheBundle()
    {
        // The separate per-checkpoint repos hold the same five files at their root, so the same
        // null-subfolder filter serves both layouts.
        var filter = HuggingFaceDownloader.CheckpointFilter(null);
        string[] selected = [.. StandaloneFiles.Where(filter)];

        Assert.Equal(ExpectedRoot.Order(), selected.Order());
    }

    [Fact]
    public void StandaloneAndBundleCataloguesCoverTheSameCheckpoints()
    {
        Assert.Equal(Router.DefaultModels.Keys.Order(), Router.StandaloneModels.Keys.Order());

        // Every standalone entry is a repo root; every bundle entry but English is a subfolder.
        Assert.All(Router.StandaloneModels.Values, spec => Assert.Null(spec.Subfolder));
        Assert.Null(Router.DefaultModels["english"].Subfolder);
        Assert.Equal("multilingual", Router.DefaultModels["multilingual"].Subfolder);
        Assert.Equal("typed-decisions", Router.DefaultModels["typed-decisions"].Subfolder);

        // The English checkpoint is the bundle's root, so both catalogues name the same thing.
        Assert.Equal(Router.DefaultModels["english"], Router.StandaloneModels["english"]);
    }

    [Fact]
    public void CacheRootFollowsTheEnvironment()
    {
        string root = HuggingFaceDownloader.DefaultCacheRoot;
        Assert.False(string.IsNullOrWhiteSpace(root));
        Assert.True(Path.IsPathRooted(root));
    }
}
