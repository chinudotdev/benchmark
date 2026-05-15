package benchmark

import (
	"testing"
)

func TestAggregateSweepEmpty(t *testing.T) {
	results := []SweepResult{}
	aggregated := AggregateSweep(results)
	if len(aggregated) != 0 {
		t.Errorf("expected 0 aggregated results, got %d", len(aggregated))
	}
}

func TestAggregateSweepSingle(t *testing.T) {
	results := []SweepResult{
		{
			Concurrency: 32,
			InputLen:    512,
			OutputLen:   256,
			RepeatIdx:   0,
			Metrics: &Metrics{
				OutputTPS: 100.0,
				TTFTP99:   50000000,  // 50ms in ns
				TPOTP99:   10000000,  // 10ms in ns
			},
		},
	}

	aggregated := AggregateSweep(results)
	if len(aggregated) != 1 {
		t.Fatalf("expected 1 aggregated result, got %d", len(aggregated))
	}

	var agg *SweepAggregate
	for _, a := range aggregated {
		agg = a
		break
	}

	if agg.TPSMean != 100.0 {
		t.Errorf("expected TPSMean=100.0, got %.1f", agg.TPSMean)
	}
	if agg.TPSStddev != 0 {
		t.Errorf("expected TPSStddev=0 for single result, got %.1f", agg.TPSStddev)
	}
	if agg.Repeats != 1 {
		t.Errorf("expected Repeats=1, got %d", agg.Repeats)
	}
	if agg.Concurrency != 32 {
		t.Errorf("expected Concurrency=32, got %d", agg.Concurrency)
	}
}

func TestAggregateSweepRepeated(t *testing.T) {
	results := []SweepResult{
		{
			Concurrency: 16,
			InputLen:    128,
			OutputLen:   128,
			RepeatIdx:   0,
			Metrics:     &Metrics{OutputTPS: 90.0, TTFTP99: 40000000, TPOTP99: 8000000},
		},
		{
			Concurrency: 16,
			InputLen:    128,
			OutputLen:   128,
			RepeatIdx:   1,
			Metrics:     &Metrics{OutputTPS: 110.0, TTFTP99: 60000000, TPOTP99: 12000000},
		},
		{
			Concurrency: 16,
			InputLen:    128,
			OutputLen:   128,
			RepeatIdx:   2,
			Metrics:     &Metrics{OutputTPS: 100.0, TTFTP99: 50000000, TPOTP99: 10000000},
		},
	}

	aggregated := AggregateSweep(results)
	if len(aggregated) != 1 {
		t.Fatalf("expected 1 aggregated result, got %d", len(aggregated))
	}

	var agg *SweepAggregate
	for _, a := range aggregated {
		agg = a
		break
	}

	if agg.TPSMean != 100.0 {
		t.Errorf("expected TPSMean=100.0, got %.1f", agg.TPSMean)
	}
	if agg.Repeats != 3 {
		t.Errorf("expected Repeats=3, got %d", agg.Repeats)
	}
	// Stddev should be ~10.0 for [90, 110, 100]
	if agg.TPSStddev < 9.0 || agg.TPSStddev > 11.0 {
		t.Errorf("expected TPSStddev≈10.0, got %.1f", agg.TPSStddev)
	}
}

func TestAggregateSweepMultipleCells(t *testing.T) {
	results := []SweepResult{
		{Concurrency: 1, InputLen: 128, OutputLen: 128, Metrics: &Metrics{OutputTPS: 50.0}},
		{Concurrency: 8, InputLen: 128, OutputLen: 128, Metrics: &Metrics{OutputTPS: 80.0}},
		{Concurrency: 32, InputLen: 128, OutputLen: 128, Metrics: &Metrics{OutputTPS: 100.0}},
	}

	aggregated := AggregateSweep(results)
	if len(aggregated) != 3 {
		t.Fatalf("expected 3 aggregated results, got %d", len(aggregated))
	}
}

func TestAggregateSweepWithErrors(t *testing.T) {
	results := []SweepResult{
		{
			Concurrency: 16, InputLen: 128, OutputLen: 128,
			Metrics: &Metrics{OutputTPS: 100.0},
		},
		{
			Concurrency: 16, InputLen: 128, OutputLen: 128,
			Error: "request failed", Metrics: nil,
		},
	}

	aggregated := AggregateSweep(results)
	if len(aggregated) != 1 {
		t.Fatalf("expected 1 aggregated result, got %d", len(aggregated))
	}

	var agg *SweepAggregate
	for _, a := range aggregated {
		agg = a
		break
	}

	// Only 1 valid result out of 2
	if agg.TPSMean != 100.0 {
		t.Errorf("expected TPSMean=100.0, got %.1f", agg.TPSMean)
	}
	if agg.Repeats != 2 {
		t.Errorf("expected Repeats=2, got %d", agg.Repeats)
	}
}

func TestSweepAggregateString(t *testing.T) {
	agg := &SweepAggregate{
		Concurrency: 32,
		InputLen:    512,
		OutputLen:   256,
		Repeats:     3,
		TPSMean:     150.5,
		TPSStddev:   12.3,
	}

	s := agg.String()
	if !contains(s, "c=32") || !contains(s, "150.5") || !contains(s, "12.3") {
		t.Errorf("String() = %q, missing expected parts", s)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsSubstr(s, substr))
}

func containsSubstr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
