package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/chinudotdev/gpu-benchmark/internal/config"
	"github.com/chinudotdev/gpu-benchmark/internal/orchestrator"
	"github.com/chinudotdev/gpu-benchmark/internal/platform"
	"github.com/chinudotdev/gpu-benchmark/internal/quality"
	"github.com/chinudotdev/gpu-benchmark/internal/report"
	"github.com/chinudotdev/gpu-benchmark/internal/sysinfo"
	"github.com/spf13/cobra"
)

var (
	version = "dev"
)

func main() {
	rootCmd := &cobra.Command{
		Use:     "gpu-benchmark",
		Short:   "LLM inference benchmark + cost calculator (NVIDIA / AMD / Tenstorrent)",
		Version: version,
	}

	rootCmd.AddCommand(runCmd())
	rootCmd.AddCommand(summarizeCmd())
	rootCmd.AddCommand(sysinfoCmd())
	rootCmd.AddCommand(reportCmd())
	rootCmd.AddCommand(exportCmd())
	rootCmd.AddCommand(preflightCmd())
	rootCmd.AddCommand(compareCmd())
	rootCmd.AddCommand(qualityCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

// ── run ────────────────────────────────────────────────────────────────────

func runCmd() *cobra.Command {
	opts := &orchestrator.Options{}

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Run GPU benchmarks against one or all models",
		Long: `Run GPU inference benchmarks against an OpenAI-compatible endpoint.

Auto-detects hardware platform (NVIDIA, AMD, or Tenstorrent).
Use --platform to override auto-detection.

Loads configuration from .gpu-benchmark.yaml if found (searches cwd → parent dirs → $HOME).
CLI flags always override config file values.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Merge config file → opts (only for flags not explicitly set)
			mergeConfig(cmd, opts)
			return orchestrator.Run(*opts)
		},
	}

	// Model selection
	cmd.Flags().StringVar(&opts.ModelID, "model-id", "", "HuggingFace model ID to benchmark")
	cmd.Flags().BoolVar(&opts.AllModels, "all", false, "Run all models in models.yaml")

	// Cost
	cmd.Flags().Float64Var(&opts.GPURate, "gpu-rate", 2.00, "Hourly GPU cost in USD")
	cmd.Flags().IntVar(&opts.GPUCount, "gpu-count", 1, "Number of GPUs")

	// Benchmark parameters
	cmd.Flags().IntVar(&opts.InputLen, "input-len", 512, "Input token length for benchmark")
	cmd.Flags().IntVar(&opts.OutputLen, "output-len", 256, "Output token length for benchmark")
	cmd.Flags().IntVar(&opts.NumPrompts, "num-prompts", 200, "Number of prompts to send")
	cmd.Flags().IntVar(&opts.MaxModelLen, "max-model-len", 8192, "Max context length to load")
	cmd.Flags().IntVar(&opts.Port, "port", 8000, "Port for vLLM server")
	cmd.Flags().IntVar(&opts.Concurrency, "concurrency", 32, "Concurrent benchmark requests")
	cmd.Flags().IntVar(&opts.Retries, "retries", 2, "Number of retries per request")
	cmd.Flags().IntVar(&opts.WarmupReqs, "warmup", 5, "Number of warmup requests to discard")

	// Platform
	cmd.Flags().StringVar(&opts.Platform, "platform", "", "Override auto-detected platform (nvidia, amd, tenstorrent)")

	// Model config
	cmd.Flags().StringVar(&opts.ModelsYAML, "models-yaml", "./models.yaml", "Path to models.yaml registry")
	cmd.Flags().StringVar(&opts.Quant, "quant", "", "Quantization for --model-id: none, int8, int4, awq, fp8")
	cmd.Flags().StringVar(&opts.HFToken, "hf-token", "", "HuggingFace Hub token (or set HF_TOKEN env var)")

	// Docker
	cmd.Flags().StringVar(&opts.DockerImage, "docker-image", "", "Docker image override")
	cmd.Flags().StringVar(&opts.GPUIDs, "gpu-ids", "", "GPU IDs for Docker (default: all). e.g. \"device=0\" or \"0,1\"")

	// Output / behavior
	cmd.Flags().StringVar(&opts.ResultsDir, "results-dir", "./results", "Directory to write result JSONs")
	cmd.Flags().BoolVar(&opts.Stream, "stream", false, "Use streaming requests for benchmark")
	cmd.Flags().BoolVar(&opts.Force, "force", false, "Re-run benchmarks even if results already exist")
	cmd.Flags().BoolVar(&opts.DryRun, "dry-run", false, "Print commands without executing")

	// Sweep options
	cmd.Flags().StringVar(&opts.ConcurrencySweep, "concurrency-sweep", "", "Comma-separated concurrency levels (e.g. \"1,2,4,8,16,32,64,128\")")
	cmd.Flags().BoolVar(&opts.SeqSweep, "seq-sweep", false, "Run all 5 sequence-length profiles from the framework spec")
	cmd.Flags().IntVar(&opts.Repeat, "repeat", 1, "Repeat each benchmark cell N times for mean ± stddev")
	cmd.Flags().StringVar(&opts.TrafficProfile, "traffic-profile", "", "Traffic profile: single-stream, interactive, high-concurrency, offline-batch")
	cmd.Flags().BoolVar(&opts.VerifyTokens, "verify-tokens", false, "Verify prompt token counts via /tokenize endpoint")

	// Quality gate
	cmd.Flags().BoolVar(&opts.QualityGate, "quality-gate", false, "Run quality gate (lm-eval) after benchmark")
	cmd.Flags().StringVar(&opts.QualityTasks, "quality-tasks", "", "Quality gate tasks: comma-separated (default: mmlu,humaneval,gsm8k)")

	// Pre-run: resolve paths, merge config, validate
	cmd.PreRunE = func(cmd *cobra.Command, args []string) error {
		abs, err := filepath.Abs(opts.ResultsDir)
		if err != nil {
			return err
		}
		opts.ResultsDir = abs

		abs, err = filepath.Abs(opts.ModelsYAML)
		if err != nil {
			return err
		}
		opts.ModelsYAML = abs

		if opts.HFToken == "" {
			opts.HFToken = os.Getenv("HF_TOKEN")
			if opts.HFToken == "" {
				opts.HFToken = os.Getenv("HUGGING_FACE_HUB_TOKEN")
			}
		}

		if !opts.AllModels && opts.ModelID == "" {
			return fmt.Errorf("provide --model-id <id> or --all")
		}

		return nil
	}

	return cmd
}

// mergeConfig loads .gpu-benchmark.yaml and applies values for flags that
// weren't explicitly set by the user. CLI flags always win.
func mergeConfig(cmd *cobra.Command, opts *orchestrator.Options) {
	cfg, err := config.LoadFile()
	if err != nil || cfg == nil {
		return
	}

	// For each config field, only apply if the corresponding flag wasn't changed
	// from its default. cobra doesn't expose "was flag set?" directly for non-PersistentFlags,
	// so we use the Changed method via FlagSet.
	flags := cmd.Flags()

	mergeString := func(name string, target *string) {
		if !flags.Changed(name) && cfgFieldString(name, cfg) != "" {
			*target = cfgFieldString(name, cfg)
		}
	}
	mergeInt := func(name string, target *int, cfgVal int) {
		if !flags.Changed(name) && cfgVal != 0 {
			*target = cfgVal
		}
	}
	mergeFloat64 := func(name string, target *float64, cfgVal float64) {
		if !flags.Changed(name) && cfgVal != 0 {
			*target = cfgVal
		}
	}
	mergeBool := func(name string, target *bool, cfgVal bool) {
		if !flags.Changed(name) && cfgVal {
			*target = cfgVal
		}
	}

	mergeString("model-id", &opts.ModelID)
	mergeString("platform", &opts.Platform)
	mergeString("models-yaml", &opts.ModelsYAML)
	mergeString("quant", &opts.Quant)
	mergeString("hf-token", &opts.HFToken)
	mergeString("docker-image", &opts.DockerImage)
	mergeString("gpu-ids", &opts.GPUIDs)
	mergeString("results-dir", &opts.ResultsDir)
	mergeString("concurrency-sweep", &opts.ConcurrencySweep)
	mergeString("traffic-profile", &opts.TrafficProfile)

	mergeFloat64("gpu-rate", &opts.GPURate, cfg.GPURate)
	mergeInt("gpu-count", &opts.GPUCount, cfg.GPUCount)
	mergeInt("input-len", &opts.InputLen, cfg.InputLen)
	mergeInt("output-len", &opts.OutputLen, cfg.OutputLen)
	mergeInt("num-prompts", &opts.NumPrompts, cfg.NumPrompts)
	mergeInt("max-model-len", &opts.MaxModelLen, cfg.MaxModelLen)
	mergeInt("port", &opts.Port, cfg.Port)
	mergeInt("concurrency", &opts.Concurrency, cfg.Concurrency)
	mergeInt("retries", &opts.Retries, cfg.Retries)
	mergeInt("warmup", &opts.WarmupReqs, cfg.WarmupReqs)
	mergeInt("repeat", &opts.Repeat, cfg.Repeat)

	mergeBool("all", &opts.AllModels, cfg.AllModels)
	mergeBool("stream", &opts.Stream, cfg.Stream != nil && *cfg.Stream)
	mergeBool("force", &opts.Force, cfg.Force)
	mergeBool("dry-run", &opts.DryRun, cfg.DryRun)
	mergeBool("seq-sweep", &opts.SeqSweep, cfg.SeqSweep)
	mergeBool("verify-tokens", &opts.VerifyTokens, cfg.VerifyTokens)
}

// cfgFieldString returns string config values by flag name.
func cfgFieldString(name string, cfg *config.Config) string {
	switch name {
	case "model-id":
		return cfg.ModelID
	case "platform":
		return cfg.Platform
	case "models-yaml":
		return cfg.ModelsYAML
	case "quant":
		return cfg.Quant
	case "hf-token":
		return cfg.HFToken
	case "docker-image":
		return cfg.DockerImage
	case "gpu-ids":
		return cfg.GPUIDs
	case "results-dir":
		return cfg.ResultsDir
	case "concurrency-sweep":
		return cfg.ConcurrencySweep
	case "traffic-profile":
		return cfg.TrafficProfile
	default:
		return ""
	}
}

// ── summarize ──────────────────────────────────────────────────────────────

func summarizeCmd() *cobra.Command {
	var (
		resultsDir string
		format     string
	)

	cmd := &cobra.Command{
		Use:   "summarize",
		Short: "Print cost comparison table from benchmark results",
		RunE: func(cmd *cobra.Command, args []string) error {
			abs, err := filepath.Abs(resultsDir)
			if err != nil {
				return err
			}
			return report.Summarize(abs, report.SummarizeFormat(format))
		},
	}

	cmd.Flags().StringVar(&resultsDir, "results-dir", "./results", "Directory with result JSONs")
	cmd.Flags().StringVar(&format, "format", "table", "Output format: table, csv, json")

	return cmd
}

// ── sysinfo ────────────────────────────────────────────────────────────────

func sysinfoCmd() *cobra.Command {
	var (
		jsonOutput  bool
	)

	cmd := &cobra.Command{
		Use:   "sysinfo",
		Short: "Display current system configuration (GPU, CPU, RAM, OS, Docker)",
		Long: `Display system configuration including all detected accelerators.

Auto-detects all platforms (NVIDIA, AMD, Tenstorrent) simultaneously.
Shows every GPU found regardless of vendor.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			si := sysinfo.Collect(ctx)

			// Probe ALL platforms simultaneously
			allDetected := platform.ProbeAll(ctx)

			image := "vllm/vllm-openai:latest"

			if jsonOutput {
				result := struct {
					Platforms []platform.DetectedPlatform `json:"platforms"`
					*sysinfo.Info
				}{
					Platforms: allDetected,
					Info:      si,
				}
				data, _ := json.MarshalIndent(result, "", "  ")
				fmt.Println(string(data))
			} else {
				printSysinfoPretty(si, allDetected, image)
			}

			return nil
		},
	}

	cmd.Flags().BoolVar(&jsonOutput, "json", false, "Output as JSON instead of pretty table")

	return cmd
}

// ── report ─────────────────────────────────────────────────────────────────

func reportCmd() *cobra.Command {
	var resultsDir string

	cmd := &cobra.Command{
		Use:   "report",
		Short: "Generate a structured Markdown report from benchmark results",
		RunE: func(cmd *cobra.Command, args []string) error {
			abs, err := filepath.Abs(resultsDir)
			if err != nil {
				return err
			}
			path, err := report.GenerateMarkdownReport(abs)
			if err != nil {
				return err
			}
			fmt.Printf("Report written to: %s\n", path)
			return nil
		},
	}

	cmd.Flags().StringVar(&resultsDir, "results-dir", "./results", "Directory with result JSONs")
	return cmd
}

// ── export ─────────────────────────────────────────────────────────────────

func exportCmd() *cobra.Command {
	var (
		resultsDir string
		output     string
		list       bool
	)

	cmd := &cobra.Command{
		Use:   "export",
		Short: "Export results as a reproducible tar.gz artifact",
		RunE: func(cmd *cobra.Command, args []string) error {
			abs, err := filepath.Abs(resultsDir)
			if err != nil {
				return err
			}

			if list {
				if len(args) == 0 {
					return fmt.Errorf("provide archive path to list")
				}
				files, err := report.ListBundle(args[0])
				if err != nil {
					return err
				}
				for _, f := range files {
					fmt.Println(f)
				}
				return nil
			}

			path, err := report.ExportBundle(abs, output)
			if err != nil {
				return err
			}
			fmt.Printf("Exported to: %s\n", path)
			return nil
		},
	}

	cmd.Flags().StringVar(&resultsDir, "results-dir", "./results", "Directory with result JSONs")
	cmd.Flags().StringVar(&output, "output", "", "Output path (default: <results-dir>/gpu-benchmark-export-<timestamp>.tar.gz)")
	cmd.Flags().BoolVar(&list, "list", false, "List contents of an existing archive")

	return cmd
}

// ── preflight ──────────────────────────────────────────────────────────────

func preflightCmd() *cobra.Command {
	var (
		resultsDir string
		minDiskGB  int
	)

	cmd := &cobra.Command{
		Use:   "preflight",
		Short: "Run pre-benchmark checks (Docker, disk space, GPU toolkit)",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()
			abs, err := filepath.Abs(resultsDir)
			if err != nil {
				return err
			}
			results := sysinfo.PreflightChecks(ctx, abs, minDiskGB)
			sysinfo.PrintPreflight(results)

			for _, r := range results {
				if !r.Passed {
					return fmt.Errorf("preflight checks failed")
				}
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&resultsDir, "results-dir", "./results", "Results directory (for disk space check)")
	cmd.Flags().IntVar(&minDiskGB, "min-disk-gb", 50, "Minimum required disk space in GB")

	return cmd
}

// ── compare ────────────────────────────────────────────────────────────────

func compareCmd() *cobra.Command {
	var (
		format    string
		crossover bool
		sla       bool
	)

	cmd := &cobra.Command{
		Use:   "compare <dir_a> <dir_b>",
		Short: "Compare benchmark results from two directories",
		Long: `Compare benchmark results side-by-side.

Shows throughput (TPS), latency, and cost deltas for matching models.
Supports both single-run results and sweep data.

Examples:
  gpu-benchmark compare results-a100 results-h100
  gpu-benchmark compare results-a100 results-h100 --format json
  gpu-benchmark compare results-a100 results-h100 --crossover`,
		Args: cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			dirA, err := filepath.Abs(args[0])
			if err != nil {
				return err
			}
			dirB, err := filepath.Abs(args[1])
			if err != nil {
				return err
			}

			comparisons, err := report.CompareDirs(dirA, dirB)
			if err != nil {
				return err
			}

			switch format {
			case "json":
				return report.CompareJSON(comparisons)
			case "table":
				fallthrough
			default:
				report.PrintComparison(comparisons)
			}

			if crossover {
				report.CrossoverAnalysis(comparisons)
			}

			if sla {
				slaComps, err := report.CompareSLAGoodput(dirA, dirB)
				if err != nil {
					return err
				}
				printSLAComparison(slaComps)
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&format, "format", "table", "Output format: table, json")
	cmd.Flags().BoolVar(&crossover, "crossover", false, "Show crossover analysis (where B overtakes A)")
	cmd.Flags().BoolVar(&sla, "sla", false, "Include SLA-band goodput comparison")

	return cmd
}

func printSLAComparison(comps map[string][]report.SLABandComparison) {
	fmt.Println("  ══ SLA Goodput Comparison ══")
	fmt.Println()

	for model, bands := range comps {
		fmt.Printf("  Model: %s\n", model)
		fmt.Printf("    %-15s %-12s %-12s %-8s\n", "Band", "Goodput A", "Goodput B", "Winner")
		fmt.Printf("    %s\n", "─────────────────────────────────────────")
		for _, b := range bands {
			fmt.Printf("    %-15s %-12.1f %-12.1f %-8s\n",
				b.Band, b.GoodputA, b.GoodputB, b.Winner)
		}
		fmt.Println()
	}
}

// ── quality ────────────────────────────────────────────────────────────────

func qualityCmd() *cobra.Command {
	cfg := &quality.GateConfig{}

	cmd := &cobra.Command{
		Use:   "quality",
		Short: "Run quality gate evaluation (requires running inference server)",
		Long: `Run quality gate evaluation against a running inference server.

Uses lm-evaluation-harness to score model accuracy on standard benchmarks.
Install with: pip install lm-eval

This command assumes an inference server is already running on --host:--port.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			if len(cfg.Tasks) == 0 {
				cfg.Tasks = []string{"mmlu", "humaneval", "gsm8k"}
			}

			cfg.Thresholds = quality.DefaultQualityThreshold()

			result, err := quality.RunGate(ctx, *cfg)
			if err != nil {
				return err
			}

			quality.PrintGateResult(result)

			if !result.Passed {
				return fmt.Errorf("quality gate failed")
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&cfg.ModelID, "model-id", "", "Model ID to evaluate")
	cmd.Flags().StringVar(&cfg.Host, "host", "localhost", "Inference server host")
	cmd.Flags().IntVar(&cfg.Port, "port", 8000, "Inference server port")
	cmd.Flags().StringVar(&cfg.OutputDir, "output-dir", "./results", "Directory to write quality gate results")
	cmd.Flags().BoolVar(&cfg.SkipPrecision, "skip-precision", false, "Skip precision detection")

	return cmd
}

// ── shared helpers ─────────────────────────────────────────────────────────

func printSysinfoPretty(si *sysinfo.Info, detected []platform.DetectedPlatform, dockerImage string) {
	bold := "\033[1m"
	reset := "\033[0m"

	fmt.Println()
	fmt.Printf("  ══ System Configuration ══\n\n")

	if len(detected) > 0 {
		for _, dp := range detected {
			rtLabel := runtimeLabel(dp.Platform.Name())
			for _, dev := range dp.Hardware.Devices {
				fmt.Printf("  %sGPU:%s      %s (%dGB VRAM)\n", bold, reset, dev.Name, dev.VRAM_GB)
				fmt.Printf("             Driver %s | %s %s\n", dev.DriverVersion, rtLabel, dev.RuntimeVersion)
			}
		}
	} else {
		fmt.Printf("  %sGPU:%s      No accelerators detected\n", bold, reset)
	}

	if len(detected) > 1 {
		names := make([]string, len(detected))
		for i, d := range detected {
			names[i] = d.Platform.Name()
		}
		fmt.Printf("  %sNote:%s      Multiple platforms detected: %s\n", bold, reset, strings.Join(names, ", "))
		fmt.Printf("             Use --platform to select one for benchmarking\n")
	}

	sysinfo.PrintPretty(si, dockerImage)
}

func runtimeLabel(plat string) string {
	switch plat {
	case "nvidia":
		return "CUDA"
	case "amd":
		return "ROCm"
	case "tenstorrent":
		return "TT-Metal"
	default:
		return "Runtime"
	}
}
