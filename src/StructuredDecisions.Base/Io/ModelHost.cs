namespace StructuredDecisions.Io;

/// <summary>
/// The Curiosity model host every model family downloads from on demand —
/// <c>https://models.curiosity.ai/laya/&lt;checkpoint&gt;/&lt;revision&gt;/&lt;file&gt;</c> — and the variable that
/// points them all at a mirror with the same layout.
/// </summary>
public static class ModelHost
{
    /// <summary>Where checkpoints are fetched from unless a mirror is configured.</summary>
    public const string DefaultBaseUrl = "https://models.curiosity.ai/laya/";

    /// <summary>The environment variable that points downloads at a mirror.</summary>
    public const string BaseUrlVariable = "LAYA_MODEL_BASE_URL";

    /// <summary>The host root in force: <c>$LAYA_MODEL_BASE_URL</c> or <see cref="DefaultBaseUrl"/>, ending in a slash.</summary>
    public static Uri Root
    {
        get
        {
            string root = Environment.GetEnvironmentVariable(BaseUrlVariable) is { Length: > 0 } custom ? custom : DefaultBaseUrl;
            return new Uri(root.EndsWith('/') ? root : root + "/");
        }
    }
}
