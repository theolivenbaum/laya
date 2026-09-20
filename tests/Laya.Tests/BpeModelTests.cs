using Laya.Tokenizers;
using Xunit;

namespace Laya.Tests;

/// <summary>
/// The BPE model on a hand-made vocabulary, so the frozen tables and the added-token path are
/// checked without a checkpoint on disk.
/// </summary>
public class BpeModelTests
{
    private static BpeModel Build(IReadOnlyList<(int, string)>? added = null) => new(
        new Dictionary<string, int> { ["a"] = 0, ["b"] = 1, ["c"] = 2, ["ab"] = 3, ["abc"] = 4 },
        [("a", "b"), ("ab", "c")],
        unkToken: null, byteFallback: false, fuseUnk: false, ignoreMerges: false, added);

    [Fact]
    public void MergesFollowRankOrder()
    {
        var model = Build();

        Assert.Equal([4], Encode(model, "abc"));
        Assert.Equal([3, 0], Encode(model, "aba"));
        Assert.Equal([2, 1, 0], Encode(model, "cba"));
    }

    [Fact]
    public void AddedTokensJoinTheVocabularyBeforeItIsFrozen()
    {
        var model = Build([(7, "<mask>"), (0, "a")]);

        Assert.True(model.TryGetId("<mask>", out int mask));
        Assert.Equal(7, mask);
        Assert.Equal("<mask>", model.TokenAt(7));
        Assert.Equal(8, model.Count);

        // An added token already in the vocabulary keeps the vocabulary's id.
        Assert.True(model.TryGetId("a", out int a));
        Assert.Equal(0, a);
    }

    [Fact]
    public void EncodingIsStableAcrossRepeatsAndThreads()
    {
        var model = Build();
        var expected = Encode(model, "abcab");

        // The word cache is filled from whichever thread encodes first; every later reader must see
        // the same ids whether it hit the cache or merged again.
        Parallel.For(0, 64, _ => Assert.Equal(expected, Encode(model, "abcab")));
        Assert.Equal(expected, Encode(model, "abcab"));
    }

    private static int[] Encode(BpeModel model, string word)
    {
        var ids = new List<int>();
        model.Encode(word, ids);
        return [.. ids];
    }
}
