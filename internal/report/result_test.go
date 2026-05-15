package report

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/chinudotdev/gpu-benchmark/internal/benchmark"
)

func TestWriteAndLoadResults(t *testing.T) {
	 dir := t.TempDir()

	cost := 0.1736
	result := &BenchmarkResult{
		ModelID:   "Qwen/Qwen3-8B",
		ModelName: "Qwen3-8B",
		Quant:     "none",
		Platform:  "nvidia",
		GPU:       "NVIDIA A100-SXM4-80GB",
		GPUCount:  1,
		GPURate:   2.00,
		Benchmark: BenchConfig{
			NumPrompts: 200,
			InputLen:   512,
			OutputLen:  256,
		},
		Metrics: &benchmark.Metrics{
			TotalRequests:     200,
			SuccessfulReqs:    200,
			OutputTPS:         3200.5,
			LatencyP50:        1 * time.Second,
			LatencyP95:        2 * time.Second,
			LatencyP99:        3 * time.Second,
			SLA:               make(map[benchmark.SLABand]*benchmark.SLAResult),
		},
		Cost:      &CostResult{CostPer1MTokensUSD: &cost},
		Timestamp: time.Now().Format(time.RFC3339),
	}

	if err := WriteResult(dir, result); err != nil {
		t.Fatalf("WriteResult() error: %v", err)
	}

	// Check file exists
	path := filepath.Join(dir, "Qwen3-8B.json")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatalf("result file not created: %s", path)
	}

	// Load and verify
	results, err := LoadResults(dir)
	if err != nil {
		t.Fatalf("LoadResults() error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	loaded := results[0]
	if loaded.ModelID != "Qwen/Qwen3-8B" {
		t.Errorf("ModelID = %q", loaded.ModelID)
	}
	if loaded.Metrics.OutputTPS != 3200.5 {
		t.Errorf("OutputTPS = %f", loaded.Metrics.OutputTPS)
	}
	if loaded.Cost == nil || loaded.Cost.CostPer1MTokensUSD == nil {
		t.Fatal("Cost is nil")
	}
	if *loaded.Cost.CostPer1MTokensUSD != 0.1736 {
		t.Errorf("Cost = %f", *loaded.Cost.CostPer1MTokensUSD)
	}
}

func TestLoadResultsEmptyDir(t *testing.T) {
	dir := t.TempDir()
	results, err := LoadResults(dir)
	if err != nil {
		t.Fatalf("LoadResults() error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestLoadResultsSkipsSystemInfo(t *testing.T) {
	dir := t.TempDir()
	// Write system_info.json
	os.WriteFile(filepath.Join(dir, "system_info.json"), []byte("{}"), 0o644)

	results, err := LoadResults(dir)
	if err != nil {
		t.Fatalf("LoadResults() error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results (system_info.json should be skipped), got %d", len(results))
	}
}

func TestLoadResultsSkipsInvalidJSON(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "bad.json"), []byte("not json"), 0o644)

	results, err := LoadResults(dir)
	if err != nil {
		t.Fatalf("LoadResults() error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestWriteResultSpecialChars(t *testing.T) {
	dir := t.TempDir()
	result := &BenchmarkResult{
		ModelID:   "org/model-with/slashes",
		ModelName: "Model With Spaces/Slashes",
		Quant:     "none",
	}

	if err := WriteResult(dir, result); err != nil {
		t.Fatalf("WriteResult() error: %v", err)
	}

	// File name should have / and space replaced with _
	path := filepath.Join(dir, "Model_With_Spaces_Slashes.json")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		files, _ := os.ReadDir(dir)
		t.Fatalf("expected file %s, dir contains: %v", path, files)
	}
}

func TestResultJSONRoundTrip(t *testing.T) {
	dir := t.TempDir()
	cost := 0.5
	result := &BenchmarkResult{
		ModelID:   "test/model",
		ModelName: "TestModel",
		Quant:     "fp8",
		Platform:  "nvidia",
		GPU:       "H100",
		GPUCount:  2,
		GPURate:   3.50,
		Benchmark: BenchConfig{
			NumPrompts: 100,
			InputLen:   128,
			OutputLen:  128,
			Stream:     true,
			WarmupReqs: 5,
		},
		Metrics: &benchmark.Metrics{
			TotalRequests:     100,
			SuccessfulReqs:    98,
			FailedReqs:        2,
			OutputTPS:         5000.0,
			TTFTP99:           200 * time.Millisecond,
			TPOTP99:           15 * time.Millisecond,
			TokensPerSecPerUser: 66.67,
			SLA: map[benchmark.SLABand]*benchmark.SLAResult{
				benchmark.SLABandInteractive: {MetCount: 95, TotalCount: 98, GoodputTPS: 4850.0},
			},
		},
		Cost:      &CostResult{CostPer1MTokensUSD: &cost},
		Timestamp: "2026-05-14T10:00:00Z",
	}

	WriteResult(dir, result)
	results, _ := LoadResults(dir)

	if len(results) != 1 {
		t.Fatal("expected 1 result")
	}

	// Verify key fields survive round-trip
	r := results[0]
	if r.Platform != "nvidia" { t.Errorf("Platform") }
	if r.Benchmark.Stream != true { t.Errorf("Stream") }
	if r.Benchmark.WarmupReqs != 5 { t.Errorf("WarmupReqs") }
	if r.Metrics.TTFTP99 != 200*time.Millisecond { t.Errorf("TTFTP99") }
	if r.Metrics.TPOTP99 != 15*time.Millisecond { t.Errorf("TPOTP99") }
	if sla, ok := r.Metrics.SLA[benchmark.SLABandInteractive]; !ok || sla.MetCount != 95 {
		t.Errorf("SLA interactive MetCount")
	}

	// Verify JSON has expected fields
	path := filepath.Join(dir, "TestModel.json")
	data, _ := os.ReadFile(path)
	var raw map[string]any
	json.Unmarshal(data, &raw)
	if raw["platform"] != "nvidia" { t.Error("JSON missing platform") }
	if bench, ok := raw["benchmark"].(map[string]any); ok {
		if bench["stream"] != true { t.Error("JSON benchmark.stream") }
		if bench["warmup_reqs"] != float64(5) { t.Error("JSON benchmark.warmup_reqs") }
	}
}
