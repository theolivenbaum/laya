using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using Laya;

namespace Laya.Benchmarks;

internal static class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine(LayaRuntime.Describe());
        if (args.Length > 0 && args[0] == "--bf16-gemm")
        {
            Bf16GemmProbe.Run(args[1..]);
            return;
        }
        if (ForwardBenchmarks.ModelDirectory() is null)
        {
            Console.WriteLine("note: no checkpoint found, so the forward-pass benchmarks will do nothing. " +
                "Run `laya download --model english --cache artifacts/models-cache`, or set LAYA_TEST_MODELS.");
        }
        Console.WriteLine();

        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly)
            .Run(args, DefaultConfig.Instance.WithOptions(ConfigOptions.DisableOptimizationsValidator));
    }
}
