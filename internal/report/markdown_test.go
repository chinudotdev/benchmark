package report

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/chinudotdev/gpu-benchmark/internal/benchmark"
)

func TestGenerateMarkdownReportEmpty(t *testing.T) {
	dir := t.TempDir()
	path, err := GenerateMarkdownReport(dir)
	if err != nil {
		t.Fatalf("GenerateMarkdownReport error: %v", err)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}

	if len(content) == 0 {
		t.Error("report is empty")
	}
	if !containsStr(string(content), "GPU Benchmark Report") {
		t.Error("report missing title")
	}
	if !containsStr(string(content), "No results found") {
		t.Error("report should indicate no results")
	}
}

func TestGenerateMarkdownReportWithResults(t *testing.T) {
	dir := t.TempDir()

	// Write a test result
	result := &BenchmarkResult{
		ModelID:   "test/model-7b",
		ModelName: "model-7b",
		Quant:     "none",
		Platform:  "nvidia",
		GPU:       "A100",
		GPUCount:  1,
		GPURate:   2.00,
		Benchmark: BenchConfig{NumPrompts: 100, InputLen: 512, OutputLen: 256, Concurrency: 32, Stream: true},
		Metrics:   writeTestMetricsWithSLA(),
		Cost:      &CostResult{costPer1M(1.5)},
		Timestamp: "2025-01-01T00:00:00Z",
	}
	WriteResult(dir, result)

	// Write system info
	writeTestSystemInfo(dir)

	path, err := GenerateMarkdownReport(dir)
	if err != nil {
		t.Fatalf("GenerateMarkdownReport error: %v", err)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	report := string(content)
	if !containsStr(report, "System Configuration") {
		t.Error("missing system config section")
	}
	if !containsStr(report, "Benchmark Results") {
		t.Error("missing results section")
	}
	if !containsStr(report, "model-7b") {
		t.Error("missing model name")
	}
	if !containsStr(report, "Throughput") {
		t.Error("missing throughput section")
	}
	if !containsStr(report, "SLA Compliance") {
		t.Error("missing SLA section")
	}
}

func TestGenerateMarkdownReportWithSweep(t *testing.T) {
	dir := t.TempDir()

	// Create sweep subdirectory
	sweepDir := filepath.Join(dir, "model-7b_sweep")
	os.MkdirAll(sweepDir, 0o755)

	cells := []*SweepCellResult{
		{
			ModelName: "model-7b", ModelID: "test/model-7b",
			SweepConfig: SweepCellConfig{InputLen: 128, OutputLen: 128, Concurrency: 32, SeqProfile: "short-chat"},
			RepeatIdx: 0,
			Metrics:   jsonRaw(`{"output_tps":100.5,"ttft_p99":50000000,"tpot_p99":10000000}`),
		},
		{
			ModelName: "model-7b", ModelID: "test/model-7b",
			SweepConfig: SweepCellConfig{InputLen: 4000, OutputLen: 256, Concurrency: 8, SeqProfile: "long-input-rag"},
			RepeatIdx: 0,
			Metrics:   jsonRaw(`{"output_tps":40.2,"ttft_p99":200000000,"tpot_p99":50000000}`),
		},
	}
	for _, cell := range cells {
		WriteSweepCell(sweepDir, cell)
	}

	path, err := GenerateMarkdownReport(dir)
	if err != nil {
		t.Fatalf("GenerateMarkdownReport error: %v", err)
	}

	content, _ := os.ReadFile(path)
	report := string(content)

	if !containsStr(report, "Sweep") {
		t.Error("missing sweep section")
	}
	if !containsStr(report, "short-chat") {
		t.Error("missing short-chat profile")
	}
}

// helpers

func newTestMetrics() *benchmark.Metrics {
	return &benchmark.Metrics{
		TotalRequests:     100,
		SuccessfulReqs:    98,
		FailedReqs:        2,
		TotalTimeS:        30.5,
		TotalOutputTokens: 25000,
		OutputTPS:         819.67,
		RPS:               3.21,
		TokensPerSecPerUser: 95.2,
		TTFTP50:           30000000,
		TTFTP99:           50000000,
		TPOTP50:           8000000,
		TPOTP99:           12000000,
		LatencyP50:        200000000,
		LatencyP99:        500000000,
	}
}

func writeTestMetricsWithSLA() *benchmark.Metrics {
	m := newTestMetrics()
	m.SLA = map[benchmark.SLABand]*benchmark.SLAResult{
		benchmark.SLABandInteractive:    {Band: benchmark.SLABandInteractive, MetCount: 90, TotalCount: 98, GoodputTPS: 750.0},
		benchmark.SLABandConversational: {Band: benchmark.SLABandConversational, MetCount: 95, TotalCount: 98, GoodputTPS: 790.0},
		benchmark.SLABandBatch:          {Band: benchmark.SLABandBatch, MetCount: 98, TotalCount: 98, GoodputTPS: 819.67},
	}
	return m
}

func costPer1M(v float64) *float64 { return &v }

func writeTestSystemInfo(dir string) {
	si := &SystemInfo{
		Platform:      "nvidia",
		OS:            "Ubuntu 22.04",
		Kernel:        "5.15.0",
		CPU:           CPUInfo{Model: "AMD EPYC 7763", Cores: 64},
		RAM_GB:        512,
		DockerVersion: "24.0.7",
		DockerImage:   "vllm/vllm-openai:latest",
		DiskAvailGB:   200,
	}
	data, _ := json.MarshalIndent(si, "", "  ")
	os.WriteFile(filepath.Join(dir, "system_info.json"), data, 0o644)
}

func jsonRaw(s string) json.RawMessage {
	return json.RawMessage(s)
}

func containsStr(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && findStr(s, substr))
}

func findStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
