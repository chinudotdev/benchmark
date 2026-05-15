// Package orchestrator ties together platform detection, Docker management,
// model downloading, benchmarking, and result persistence.
package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/chinudotdev/gpu-benchmark/internal/benchmark"
	"github.com/chinudotdev/gpu-benchmark/internal/docker"
	"github.com/chinudotdev/gpu-benchmark/internal/download"
	"github.com/chinudotdev/gpu-benchmark/internal/platform"
	"github.com/chinudotdev/gpu-benchmark/internal/report"
	sysinfopkg "github.com/chinudotdev/gpu-benchmark/internal/sysinfo"
	"github.com/chinudotdev/gpu-benchmark/internal/workload"
)

// Options holds all CLI options for a benchmark run.
type Options struct {
	ModelID     string
	AllModels   bool
	Platform    string // "nvidia", "amd", "tenstorrent", or "" for auto-detect
	GPURate     float64
	GPUCount    int
	InputLen    int
	OutputLen   int
	NumPrompts  int
	MaxModelLen int
	Port        int
	ResultsDir  string
	Concurrency int
	ModelsYAML  string
	Quant       string
	HFToken     string
	DockerImage string
	GPUIDs      string
	Stream      bool
	Force       bool
	DryRun      bool
	Retries     int
	WarmupReqs  int

	// Sweep options (Milestone 3)
	ConcurrencySweep string // comma-separated: "1,2,4,8,16,32,64,128"
	SeqSweep         bool   // run all 5 sequence-length profiles
	Repeat           int    // repeat each cell N times (default: 1)
	TrafficProfile   string // "single-stream", "interactive", "high-concurrency", "offline-batch", or ""
	VerifyTokens     bool   // verify prompt token counts via /tokenize

	// Quality gate (Milestone 5)
	QualityGate  bool   // run quality gate after benchmark
	QualityTasks string // comma-separated task names for quality gate
}

// Run executes the full benchmark pipeline.
func Run(opts Options) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Detect platform
	plat, err := detectPlatform(ctx, opts.Platform)
	if err != nil {
		return fmt.Errorf("platform detection: %w", err)
	}

	log.Printf("Platform: %s", plat.Name())

	// Detect hardware
	hw, err := plat.DetectHardware(ctx)
	if err != nil {
		return fmt.Errorf("hardware detection: %w", err)
	}

	for _, dev := range hw.Devices {
		log.Printf("  GPU: %s (%dGB VRAM, driver=%s)", dev.Name, dev.VRAM_GB, dev.DriverVersion)
	}

	// Check container runtime
	if !opts.DryRun {
		if err := plat.DetectOrInstallRuntime(ctx); err != nil {
			return err
		}
	}

	// Create Docker manager
	dmgr, err := docker.NewManager(ctx)
	if err != nil {
		return fmt.Errorf("docker init: %w", err)
	}

	// Single signal handler: cancel in-flight ops + stop containers + exit
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		fmt.Printf("\nReceived %v, cleaning up...\n", sig)
		cancel()       // stop in-flight operations
		dmgr.Cleanup() // stop containers
		os.Exit(130)
	}()

	// Cleanup on normal exit
	defer dmgr.Cleanup()

	// Pull Docker image
	image := plat.GetDockerImage(opts.DockerImage)
	if image != "" && !opts.DryRun {
		if err := dmgr.Pull(ctx, image); err != nil {
			return fmt.Errorf("docker pull: %w", err)
		}
	}

	// Create results directory
	if err := os.MkdirAll(opts.ResultsDir, 0o755); err != nil {
		return fmt.Errorf("create results dir: %w", err)
	}

	// Write system info
	sysInfo := collectSystemInfo(ctx, hw, plat, image)
	if !opts.DryRun {
		writeSystemInfo(opts.ResultsDir, sysInfo)
	}

	// Load models
	var models []platform.ModelConfig
	if opts.AllModels {
		models, err = workload.LoadModels(opts.ModelsYAML)
		if err != nil {
			return fmt.Errorf("load models: %w", err)
		}
		log.Printf("Found %d models in registry", len(models))
	} else if opts.ModelID != "" {
		models = []platform.ModelConfig{workload.FindModel(opts.ModelID, opts.Quant)}
	} else {
		return fmt.Errorf("provide --model-id <id> or --all")
	}

	// Run benchmarks
	for _, model := range models {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		isSweep := opts.ConcurrencySweep != "" || opts.SeqSweep || opts.Repeat > 1
		if isSweep {
			if err := benchmarkModelSweep(ctx, plat, dmgr, model, opts, hw, sysInfo); err != nil {
				log.Printf("ERROR sweep benchmarking %s: %v", model.Name, err)
			}
		} else {
			if err := benchmarkModel(ctx, plat, dmgr, model, opts, hw, sysInfo); err != nil {
				log.Printf("ERROR benchmarking %s: %v", model.Name, err)
			}
		}
	}

	// Print summary
	printSummary(opts.ResultsDir)
	return nil
}

// benchmarkModel runs the full pipeline for a single model.
func benchmarkModel(
	ctx context.Context,
	plat platform.Platform,
	dmgr *docker.Manager,
	model platform.ModelConfig,
	opts Options,
	hw *platform.HardwareInfo,
	sysInfo *report.SystemInfo,
) error {
	fmt.Println()
	printHeader(fmt.Sprintf("Benchmarking: %s", model.Name))

	// VRAM check
	if len(hw.Devices) > 0 {
		firstGPU := hw.Devices[0]
		if model.MinVRAM_GB > 0 && firstGPU.VRAM_GB < model.MinVRAM_GB && !opts.DryRun {
			log.Printf("Skipping %s — requires %dGB, GPU has %dGB", model.Name, model.MinVRAM_GB, firstGPU.VRAM_GB)
			return nil
		}
	}

	// Check existing result
	resultFile := filepath.Join(opts.ResultsDir, safeName(model.Name)+".json")
	if _, err := os.Stat(resultFile); err == nil && !opts.Force && !opts.DryRun {
		log.Printf("Result already exists: %s — use --force to re-run", resultFile)
		return nil
	}

	// Download model
	if !opts.DryRun {
		download.FixCachePermissions()
		if err := download.Download(ctx, model.ID, opts.HFToken); err != nil {
			return fmt.Errorf("download model: %w", err)
		}
	}

	// Build container config
	runOpts := platform.RunOptions{
		GPURate:     opts.GPURate,
		GPUCount:    opts.GPUCount,
		MaxModelLen: opts.MaxModelLen,
		Port:        opts.Port,
		Quant:       opts.Quant,
		DockerImage: opts.DockerImage,
		GPUIDs:      opts.GPUIDs,
		HFToken:     opts.HFToken,
		Stream:      opts.Stream,
		Force:       opts.Force,
		DryRun:      opts.DryRun,
	}
	containerCfg := plat.ContainerConfig(model, runOpts)

	// Start container
	if opts.DryRun {
		log.Println("[DRY RUN] Would start container:")
		log.Printf("  Image:  %s", containerCfg.Image)
		log.Printf("  Model:  %s", model.ID)
		log.Printf("  Port:   %d", opts.Port)
		log.Println()
		return nil
	}

	// Check port availability
	if docker.IsPortInUse(opts.Port) {
		containerName := fmt.Sprintf("vllm_bench_%d", opts.Port)
		status := dmgr.ContainerStatus(ctx, containerName)
		if status != "" {
			return fmt.Errorf("port %d is already in use by container %s (status: %s)\n  Remove with: docker rm -f %s",
				opts.Port, containerName, status, containerName)
		}
		return fmt.Errorf("port %d is already in use. Stop the process or specify --port", opts.Port)
	}

	// Stop any existing container on this port
	containerName := fmt.Sprintf("vllm_bench_%d", opts.Port)
	dmgr.Stop(containerName)

	// Start container
	if err := dmgr.Run(ctx, containerCfg); err != nil {
		return fmt.Errorf("start container: %w", err)
	}

	// Wait for healthy
	coldStart, err := dmgr.WaitHealthy(ctx, opts.Port, 10*time.Minute)
	if err != nil {
		return fmt.Errorf("server health check: %w", err)
	}

	log.Printf("Cold start time: %v", coldStart.Round(time.Second))

	// Run benchmark
	log.Println("Running benchmark...")
	log.Printf("  Prompts:    %d", opts.NumPrompts)
	log.Printf("  Input len:  %d tokens", opts.InputLen)
	log.Printf("  Output len: %d tokens", opts.OutputLen)
	log.Printf("  Concurrency: %d", opts.Concurrency)
	log.Printf("  Stream:     %v", opts.Stream)

	metrics, err := benchmark.Run(ctx, benchmark.RunnerConfig{
		Host:        "localhost",
		Port:        opts.Port,
		Model:       model.ID,
		NumPrompts:  opts.NumPrompts,
		InputLen:    opts.InputLen,
		OutputLen:   opts.OutputLen,
		Concurrency: opts.Concurrency,
		Stream:      opts.Stream,
		Retries:     opts.Retries,
		WarmupReqs:  opts.WarmupReqs,
	})
	if err != nil {
		dmgr.Stop(containerName)
		return fmt.Errorf("benchmark run: %w", err)
	}

	metrics.ColdStartS = coldStart.Seconds()

	report.PrintMetrics(metrics)

	// Calculate cost
	costVal := report.CalculateCost(metrics.OutputTPS, opts.GPURate, opts.GPUCount)
	costStr := "N/A"
	if costVal != nil {
		costStr = fmt.Sprintf("$%.4f", *costVal)
	}

	// GPU name for result
	gpuName := "unknown"
	if len(hw.Devices) > 0 {
		gpuName = hw.Devices[0].Name
	}

	// Write result
	result := &report.BenchmarkResult{
		ModelID:   model.ID,
		ModelName: model.Name,
		Quant:     model.Quant,
		Platform:  plat.Name(),
		GPU:       gpuName,
		GPUCount:  opts.GPUCount,
		GPURate:   opts.GPURate,
		Benchmark: report.BenchConfig{
			NumPrompts:  opts.NumPrompts,
			InputLen:    opts.InputLen,
			OutputLen:   opts.OutputLen,
			MaxModelLen: opts.MaxModelLen,
			Concurrency: opts.Concurrency,
			Stream:      opts.Stream,
			WarmupReqs:  opts.WarmupReqs,
		},
		Metrics: metrics,
		Cost:    &report.CostResult{CostPer1MTokensUSD: costVal},
		System:  sysInfo,
		Timestamp: time.Now().Format(time.RFC3339),
	}

	if err := report.WriteResult(opts.ResultsDir, result); err != nil {
		log.Printf("Warning: failed to write result: %v", err)
	}

	fmt.Printf("  TPS:              %s tok/s\n", formatFloat(metrics.OutputTPS))
	fmt.Printf("  Cost per 1M tok:  %s\n", costStr)
	fmt.Println()

	// Stop container
	dmgr.Stop(containerName)

	// Settle delay to avoid port reuse race
	time.Sleep(2 * time.Second)

	return nil
}

// benchmarkModelSweep runs the full sweep pipeline for a single model.
// It handles: concurrency sweep, seq-profile sweep, repeat, and matrix orchestration.
func benchmarkModelSweep(
	ctx context.Context,
	plat platform.Platform,
	dmgr *docker.Manager,
	model platform.ModelConfig,
	opts Options,
	hw *platform.HardwareInfo,
	sysInfo *report.SystemInfo,
) error {
	fmt.Println()
	printHeader(fmt.Sprintf("Sweep Benchmarking: %s", model.Name))

	// VRAM check
	if len(hw.Devices) > 0 {
		firstGPU := hw.Devices[0]
		if model.MinVRAM_GB > 0 && firstGPU.VRAM_GB < model.MinVRAM_GB && !opts.DryRun {
			log.Printf("Skipping %s — requires %dGB, GPU has %dGB", model.Name, model.MinVRAM_GB, firstGPU.VRAM_GB)
			return nil
		}
	}

	// Download model
	if !opts.DryRun {
		download.FixCachePermissions()
		if err := download.Download(ctx, model.ID, opts.HFToken); err != nil {
			return fmt.Errorf("download model: %w", err)
		}
	}

	// Build container config
	containerCfg := plat.ContainerConfig(model, platform.RunOptions{
		GPURate: opts.GPURate, GPUCount: opts.GPUCount, MaxModelLen: opts.MaxModelLen,
		Port: opts.Port, Quant: opts.Quant, DockerImage: opts.DockerImage,
		GPUIDs: opts.GPUIDs, HFToken: opts.HFToken, Stream: opts.Stream,
		Force: opts.Force, DryRun: opts.DryRun,
	})

	if opts.DryRun {
		log.Println("[DRY RUN] Would start container and run sweep")
		return nil
	}

	// Start container
	containerName := fmt.Sprintf("vllm_bench_%d", opts.Port)
	dmgr.Stop(containerName)

	if err := dmgr.Run(ctx, containerCfg); err != nil {
		return fmt.Errorf("start container: %w", err)
	}

	// Wait for healthy
	coldStart, err := dmgr.WaitHealthy(ctx, opts.Port, 10*time.Minute)
	if err != nil {
		return fmt.Errorf("server health check: %w", err)
	}
	log.Printf("Cold start time: %v", coldStart.Round(time.Second))

	// Ensure container is stopped when we're done
	defer func() {
		dmgr.Stop(containerName)
		time.Sleep(2 * time.Second)
	}()

	// Load prompt dataset
	dataset, err := benchmark.LoadPromptDataset()
	if err != nil {
		log.Printf("Warning: could not load prompt dataset, using synthetic: %v", err)
		dataset = nil
	}

	// Parse concurrency levels
	concLevels := parseConcurrencyLevels(opts.Concurrency, opts.ConcurrencySweep)

	// Token verification (optional)
	if opts.VerifyTokens && dataset != nil {
		log.Println("Verifying prompt token counts via /tokenize...")
		samplePrompts := dataset.GeneratePrompts(5, opts.InputLen)
		counts, err := benchmark.TokenizeBatch(ctx, "localhost", opts.Port, model.ID, samplePrompts, opts.InputLen, 30.0)
		if err != nil {
			log.Printf("Warning: token verification failed: %v", err)
		} else {
			log.Printf("  Token counts: %v (target: %d)", counts, opts.InputLen)
		}
	}

	// Determine seq profiles to sweep
	var seqProfiles []benchmark.SeqProfileEntry
	if opts.SeqSweep {
		for _, p := range workload.SeqProfiles {
			seqProfiles = append(seqProfiles, benchmark.SeqProfileEntry{
				Name:         p.Name,
				InputTokens:  p.InputTokens,
				OutputTokens: p.OutputTokens,
			})
		}
		log.Printf("Seq profiles: %d profiles", len(seqProfiles))
	} else {
		// Single profile from --input-len / --output-len
		seqProfiles = []benchmark.SeqProfileEntry{{
			Name:         "custom",
			InputTokens:  opts.InputLen,
			OutputTokens: opts.OutputLen,
		}}
	}

	repeat := opts.Repeat
	if repeat < 1 {
		repeat = 1
	}

	log.Println("Running sweep...")
	log.Printf("  Concurrency levels: %v", concLevels)
	log.Printf("  Seq profiles:       %d", len(seqProfiles))
	log.Printf("  Repeats per cell:   %d", repeat)
	log.Printf("  Total cells:        %d", len(seqProfiles)*len(concLevels)*repeat)

	// Create sweep results subdirectory
	sweepDir := filepath.Join(opts.ResultsDir, safeName(model.Name)+"_sweep")
	if err := os.MkdirAll(sweepDir, 0o755); err != nil {
		return fmt.Errorf("create sweep dir: %w", err)
	}

	// Run the sweep
	sweepCfg := benchmark.SweepConfig{
		Host:              "localhost",
		Port:              opts.Port,
		Model:             model.ID,
		Stream:            opts.Stream,
		Retries:           opts.Retries,
		WarmupReqs:        opts.WarmupReqs,
		ConcurrencyLevels: concLevels,
		NumPrompts:        opts.NumPrompts,
		Repeat:            repeat,
		Prompts:           dataset,
	}

	var sweepResults []benchmark.SweepResult
	if opts.SeqSweep {
		sweepResults, err = benchmark.RunSeqSweep(ctx, sweepCfg, seqProfiles)
	} else {
		sweepCfg.InputLen = opts.InputLen
		sweepCfg.OutputLen = opts.OutputLen
		sweepResults, err = benchmark.RunConcurrencySweep(ctx, sweepCfg)
	}
	if err != nil {
		return fmt.Errorf("sweep run: %w", err)
	}

	// GPU name for results
	gpuName := "unknown"
	if len(hw.Devices) > 0 {
		gpuName = hw.Devices[0].Name
	}

	// Write per-cell results
	for _, sr := range sweepResults {
		metricsJSON, _ := json.Marshal(sr.Metrics)
		cellResult := &report.SweepCellResult{
			ModelID:   model.ID,
			ModelName: model.Name,
			Quant:     model.Quant,
			Platform:  plat.Name(),
			GPU:       gpuName,
			GPUCount:  opts.GPUCount,
			GPURate:   opts.GPURate,
			SweepConfig: report.SweepCellConfig{
				InputLen:      sr.InputLen,
				OutputLen:     sr.OutputLen,
				Concurrency:   sr.Concurrency,
				NumPrompts:    opts.NumPrompts,
				SeqProfile:    seqProfileName(seqProfiles, sr.InputLen, sr.OutputLen),
				TrafficProfile: opts.TrafficProfile,
				Stream:        opts.Stream,
			},
			RepeatIdx: sr.RepeatIdx,
			Metrics:   metricsJSON,
			Error:     sr.Error,
			System:    sysInfo,
			Timestamp: time.Now().Format(time.RFC3339),
		}
		if writeErr := report.WriteSweepCell(sweepDir, cellResult); writeErr != nil {
			log.Printf("Warning: failed to write sweep cell: %v", writeErr)
		}
	}

	// Print aggregated results
	aggregated := benchmark.AggregateSweep(sweepResults)
	benchmark.PrintSweepTable(aggregated)

	log.Printf("Sweep complete. Results in: %s/", sweepDir)
	return nil
}

// parseConcurrencyLevels parses the concurrency-sweep flag or falls back to the single concurrency level.
func parseConcurrencyLevels(single int, sweep string) []int {
	if sweep != "" {
		var levels []int
		for _, s := range strings.Split(sweep, ",") {
			s = strings.TrimSpace(s)
			if n, err := strconv.Atoi(s); err == nil && n > 0 {
				levels = append(levels, n)
			}
		}
		if len(levels) > 0 {
			return levels
		}
	}
	return []int{single}
}

// seqProfileName finds the profile name for the given input/output lengths.
func seqProfileName(profiles []benchmark.SeqProfileEntry, inLen, outLen int) string {
	for _, p := range profiles {
		if p.InputTokens == inLen && p.OutputTokens == outLen {
			return p.Name
		}
	}
	return fmt.Sprintf("in%d-out%d", inLen, outLen)
}

// detectPlatform auto-detects or selects the platform.
func detectPlatform(ctx context.Context, name string) (platform.Platform, error) {
	switch strings.ToLower(name) {
	case "nvidia":
		return platform.NewNVIDIAPlatform(), nil
	case "amd":
		return platform.NewAMDPlatform(), nil
	case "tenstorrent":
		return platform.NewTenstorrentPlatform(), nil
	case "":
		// Auto-detect: try each platform
		nvidia := platform.NewNVIDIAPlatform()
		if _, err := nvidia.DetectHardware(ctx); err == nil {
			return nvidia, nil
		}
		amd := platform.NewAMDPlatform()
		if _, err := amd.DetectHardware(ctx); err == nil {
			return amd, nil
		}
		tt := platform.NewTenstorrentPlatform()
		if _, err := tt.DetectHardware(ctx); err == nil {
			return tt, nil
		}
		return nil, fmt.Errorf("no supported accelerators detected (tried NVIDIA, AMD, Tenstorrent)")
	default:
		return nil, fmt.Errorf("unknown platform: %s (use nvidia, amd, or tenstorrent)", name)
	}
}

func collectSystemInfo(ctx context.Context, hw *platform.HardwareInfo, plat platform.Platform, image string) *report.SystemInfo {
	si := sysinfopkg.Collect(ctx)
	return &report.SystemInfo{
		Platform:      plat.Name(),
		Devices:       hw.Devices,
		CPU:           report.CPUInfo{Model: si.CPU.Model, Cores: si.CPU.Cores},
		RAM_GB:        si.RAM_GB,
		OS:            si.OS,
		Kernel:        si.Kernel,
		DockerVersion: si.DockerVersion,
		DockerImage:   image,
		DiskAvailGB:   si.DiskAvailGB,
		CollectedAt:   time.Now().Format(time.RFC3339),
	}
}

func writeSystemInfo(dir string, info *report.SystemInfo) {
	path := filepath.Join(dir, "system_info.json")
	data, _ := json.MarshalIndent(info, "", "  ")
	data = append(data, '\n')
	os.WriteFile(path, data, 0o644)
}

func safeName(name string) string {
	r := strings.NewReplacer("/", "_", "\\", "_", " ", "_")
	return r.Replace(name)
}

func formatFloat(f float64) string {
	return strconv.FormatFloat(f, 'f', 2, 64)
}

func printHeader(s string) {
	line := strings.Repeat("═", len(s)+4)
	fmt.Printf("\n╔%s╗\n", line)
	fmt.Printf("║ %s ║\n", s)
	fmt.Printf("╚%s╝\n\n", line)
}

func printSummary(resultsDir string) {
	printHeader("Benchmark Summary")
	results, err := report.LoadResults(resultsDir)
	if err != nil || len(results) == 0 {
		log.Printf("No results found in %s", resultsDir)
		return
	}

	fmt.Printf("  %-40s  %-12s  %-18s  %-20s\n", "Model", "Quant", "TPS (tok/s)", "Cost/1M tokens")
	sep := strings.Repeat("─", 40) + "  " + strings.Repeat("─", 12) + "  " +
		strings.Repeat("─", 18) + "  " + strings.Repeat("─", 20)
	fmt.Printf("  %s\n", sep)

	for _, r := range results {
		tps := ""
		if r.Metrics != nil {
			tps = strconv.FormatFloat(r.Metrics.OutputTPS, 'f', 2, 64)
		}
		cost := "N/A"
		if r.Cost != nil && r.Cost.CostPer1MTokensUSD != nil {
			cost = fmt.Sprintf("$%.4f", *r.Cost.CostPer1MTokensUSD)
		}
		fmt.Printf("  %-40s  %-12s  %-18s  %-20s\n", r.ModelName, r.Quant, tps, cost)
	}

	fmt.Printf("  %s\n\n", sep)
	fmt.Printf("Full JSON results in: %s/\n\n", resultsDir)
}
