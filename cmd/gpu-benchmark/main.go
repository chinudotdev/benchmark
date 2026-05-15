package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/chinudotdev/gpu-benchmark/internal/orchestrator"
	"github.com/chinudotdev/gpu-benchmark/internal/report"
	"github.com/spf13/cobra"
)

var (
	version = "1.0.0"
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

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func runCmd() *cobra.Command {
	opts := &orchestrator.Options{}

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Run GPU benchmarks against one or all models",
		RunE: func(cmd *cobra.Command, args []string) error {
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
	cmd.Flags().StringVar(&opts.Platform, "platform", "", "Platform: nvidia, amd, tenstorrent (auto-detect if empty)")

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

	// Resolve paths
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

		// HF token from env if not provided
		if opts.HFToken == "" {
			opts.HFToken = os.Getenv("HF_TOKEN")
			if opts.HFToken == "" {
				opts.HFToken = os.Getenv("HUGGING_FACE_HUB_TOKEN")
			}
		}

		// Validate: need --model-id or --all
		if !opts.AllModels && opts.ModelID == "" {
			return fmt.Errorf("provide --model-id <id> or --all")
		}

		return nil
	}

	return cmd
}

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

func sysinfoCmd() *cobra.Command {
	var (
		platformName string
		jsonOutput   bool
	)

	cmd := &cobra.Command{
		Use:   "sysinfo",
		Short: "Display current system configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			// TODO: Implement standalone sysinfo using platform detection
			fmt.Println("System info collection — use 'gpu-benchmark run --dry-run' for now")
			return nil
		},
	}

	cmd.Flags().StringVar(&platformName, "platform", "", "Platform: nvidia, amd, tenstorrent (auto-detect if empty)")
	cmd.Flags().BoolVar(&jsonOutput, "json", false, "Output as JSON instead of pretty table")

	return cmd
}
