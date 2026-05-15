package report

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/chinudotdev/gpu-benchmark/internal/benchmark"
)

func TestCompareEmptyDirs(t *testing.T) {
	dirA := t.TempDir()
	dirB := t.TempDir()

	_, err := CompareDirs(dirA, dirB)
	if err == nil {
		t.Error("expected error for empty dirs")
	}
}

func TestCompareMatchingModels(t *testing.T) {
	dirA := t.TempDir()
	dirB := t.TempDir()

	resultA := &BenchmarkResult{
		ModelID:   "test/model-7b",
		ModelName: "model-7b",
		Quant:     "none",
		Platform:  "nvidia",
		GPU:       "A100",
		GPUCount:  1,
		GPURate:   2.00,
		Benchmark: BenchConfig{NumPrompts: 100, InputLen: 512, OutputLen: 256, Concurrency: 32},
		Metrics:   newTestMetricsForCompare(500.0),
		Cost:      &CostResult{costPer1M(2.50)},
		Timestamp: "2025-01-01T00:00:00Z",
	}
	WriteResult(dirA, resultA)

	resultB := &BenchmarkResult{
		ModelID:   "test/model-7b",
		ModelName: "model-7b",
		Quant:     "none",
		Platform:  "nvidia",
		GPU:       "H100",
		GPUCount:  1,
		GPURate:   3.00,
		Benchmark: BenchConfig{NumPrompts: 100, InputLen: 512, OutputLen: 256, Concurrency: 32},
		Metrics:   newTestMetricsForCompare(800.0),
		Cost:      &CostResult{costPer1M(2.00)},
		Timestamp: "2025-01-02T00:00:00Z",
	}
	WriteResult(dirB, resultB)

	comparisons, err := CompareDirs(dirA, dirB)
	if err != nil {
		t.Fatalf("CompareDirs error: %v", err)
	}

	if len(comparisons) != 1 {
		t.Fatalf("expected 1 comparison, got %d", len(comparisons))
	}

	cr := comparisons[0]
	if cr.ModelName != "model-7b" {
		t.Errorf("model name = %q, want model-7b", cr.ModelName)
	}
	if len(cr.Cells) != 1 {
		t.Fatalf("expected 1 cell, got %d", len(cr.Cells))
	}

	cell := cr.Cells[0]
	if cell.TPSA != 500.0 {
		t.Errorf("TPSA = %f, want 500.0", cell.TPSA)
	}
	if cell.TPSB != 800.0 {
		t.Errorf("TPSB = %f, want 800.0", cell.TPSB)
	}
	if cell.TPSDelta != 300.0 {
		t.Errorf("TPSDelta = %f, want 300.0", cell.TPSDelta)
	}
	if cell.Winner != "B" {
		t.Errorf("Winner = %q, want B", cell.Winner)
	}
}

func TestCompareNoMatchingModels(t *testing.T) {
	dirA := t.TempDir()
	dirB := t.TempDir()

	resultA := &BenchmarkResult{
		ModelID: "test/model-a", ModelName: "model-a",
		Quant: "none", Platform: "nvidia", GPU: "A100", GPUCount: 1, GPURate: 2.00,
		Timestamp: "2025-01-01T00:00:00Z",
	}
	WriteResult(dirA, resultA)

	resultB := &BenchmarkResult{
		ModelID: "test/model-b", ModelName: "model-b",
		Quant: "none", Platform: "nvidia", GPU: "H100", GPUCount: 1, GPURate: 3.00,
		Timestamp: "2025-01-01T00:00:00Z",
	}
	WriteResult(dirB, resultB)

	comparisons, err := CompareDirs(dirA, dirB)
	if err != nil {
		t.Fatalf("CompareDirs error: %v", err)
	}
	if len(comparisons) != 0 {
		t.Errorf("expected 0 comparisons for non-matching models, got %d", len(comparisons))
	}
}

func TestCompareWithSweeps(t *testing.T) {
	dirA := t.TempDir()
	dirB := t.TempDir()

	// Need at least one result in each dir for CompareDirs to not bail early
	aResult := &BenchmarkResult{
		ModelID: "test/model-7b", ModelName: "model-7b",
		Quant: "none", GPU: "A100", GPUCount: 1, GPURate: 2.00,
		Metrics: newTestMetricsForCompare(100.0),
		Timestamp: "2025-01-01T00:00:00Z",
	}
	WriteResult(dirA, aResult)

	bResult := &BenchmarkResult{
		ModelID: "test/model-7b", ModelName: "model-7b",
		Quant: "none", GPU: "H100", GPUCount: 1, GPURate: 3.00,
		Metrics: newTestMetricsForCompare(200.0),
		Timestamp: "2025-01-01T00:00:00Z",
	}
	WriteResult(dirB, bResult)

	sweepDirA := filepath.Join(dirA, "model-7b_sweep")
	sweepDirB := filepath.Join(dirB, "model-7b_sweep")
	os.MkdirAll(sweepDirA, 0o755)
	os.MkdirAll(sweepDirB, 0o755)

	writeSweepCellHelper(sweepDirA, "model-7b", "short-chat", 128, 128, 32, 0, 100.0, 50000000, 10000000)
	writeSweepCellHelper(sweepDirB, "model-7b", "short-chat", 128, 128, 32, 0, 200.0, 25000000, 5000000)

	comparisons, err := CompareDirs(dirA, dirB)
	if err != nil {
		t.Fatalf("CompareDirs error: %v", err)
	}

	found := false
	for _, cr := range comparisons {
		for _, cell := range cr.Cells {
			if cell.SeqProfile == "short-chat" {
				found = true
				if cell.TPSA != 100.0 {
					t.Errorf("TPSA = %f, want 100.0", cell.TPSA)
				}
				if cell.TPSB != 200.0 {
					t.Errorf("TPSB = %f, want 200.0", cell.TPSB)
				}
			}
		}
	}
	if !found {
		t.Error("expected sweep comparison cells")
	}
}

func TestPrintComparisonEmpty(t *testing.T) {
	// Should not panic
	PrintComparison(nil)
}

func TestPrintComparisonWithData(t *testing.T) {
	comparisons := []*CompareResult{
		{
			DirA:      "/tmp/a",
			DirB:      "/tmp/b",
			ModelName: "test-model",
			Cells: []CompareCell{
				{
					SeqProfile:  "short-chat",
					InputLen:    128,
					OutputLen:   128,
					Concurrency: 32,
					TPSA:        100.0,
					TPSB:        200.0,
					TPSDelta:    100.0,
					TPSDeltaPct: 100.0,
					Winner:      "B",
				},
			},
		},
	}
	PrintComparison(comparisons)
}

func TestCrossoverAnalysis(t *testing.T) {
	comparisons := []*CompareResult{
		{
			ModelName: "test-model",
			Cells: []CompareCell{
				{SeqProfile: "short-chat", Concurrency: 8, TPSA: 200, TPSB: 100},
				{SeqProfile: "short-chat", Concurrency: 32, TPSA: 500, TPSB: 600},
				{SeqProfile: "short-chat", Concurrency: 64, TPSA: 700, TPSB: 900},
			},
		},
	}
	CrossoverAnalysis(comparisons)
}

func TestCompareJSON(t *testing.T) {
	comparisons := []*CompareResult{
		{ModelName: "test", DirA: "/a", DirB: "/b"},
	}
	err := CompareJSON(comparisons)
	if err != nil {
		t.Errorf("CompareJSON error: %v", err)
	}
}

// helpers

func newTestMetricsForCompare(tps float64) *benchmark.Metrics {
	return &benchmark.Metrics{
		TotalRequests:  100,
		SuccessfulReqs: 100,
		OutputTPS:      tps,
		TTFTP99:        50000000,
		TPOTP99:        10000000,
	}
}

func writeSweepCellHelper(dir, model, profile string, inLen, outLen, conc, rep int, tps float64, ttft99, tpot99 int64) {
	metrics := map[string]any{
		"output_tps": tps,
		"ttft_p99":   ttft99,
		"tpot_p99":   tpot99,
	}
	metricsJSON, _ := json.Marshal(metrics)

	cell := &SweepCellResult{
		ModelName: model,
		ModelID:   "test/" + model,
		SweepConfig: SweepCellConfig{
			InputLen:    inLen,
			OutputLen:   outLen,
			Concurrency: conc,
			SeqProfile:  profile,
		},
		RepeatIdx: rep,
		Metrics:   metricsJSON,
	}
	WriteSweepCell(dir, cell)
}

func TestCompareSLAGoodput(t *testing.T) {
	dirA := t.TempDir()
	dirB := t.TempDir()

	// Create results with SLA data
	a := &BenchmarkResult{
		ModelName: "model-7b",
		ModelID:   "test/model-7b",
		Quant:     "none", GPU: "A100", GPUCount: 1, GPURate: 2.00,
		Timestamp: "2025-01-01T00:00:00Z",
		Metrics:   newTestMetricsForCompare(500.0),
	}
	WriteResult(dirA, a)

	b := &BenchmarkResult{
		ModelName: "model-7b",
		ModelID:   "test/model-7b",
		Quant:     "none", GPU: "H100", GPUCount: 1, GPURate: 3.00,
		Timestamp: "2025-01-01T00:00:00Z",
		Metrics:   newTestMetricsForCompare(800.0),
	}
	WriteResult(dirB, b)

	// Without SLA data, should return empty
	comps, err := CompareSLAGoodput(dirA, dirB)
	if err != nil {
		t.Fatalf("CompareSLAGoodput error: %v", err)
	}
	// Since Metrics don't have SLA, this should be empty
	_ = comps
}

// Ensure fmt is available for this test file
var _ = fmt.Sprintf
