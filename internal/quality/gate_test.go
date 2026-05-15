package quality

import (
	"testing"
)

func TestDefaultQualityThreshold(t *testing.T) {
	th := DefaultQualityThreshold()
	if th.MMLUMin <= 0 {
		t.Error("MMLUMin should be positive")
	}
	if th.HumanEvalMin <= 0 {
		t.Error("HumanEvalMin should be positive")
	}
	if th.MinTasks < 1 {
		t.Error("MinTasks should be >= 1")
	}
}

func TestEvaluateGate(t *testing.T) {
	tests := []struct {
		name     string
		results  []EvalResult
		threshold QualityThreshold
		want     bool
	}{
		{
			name: "all pass",
			results: []EvalResult{
				{Task: "mmlu", Score: 0.65},
				{Task: "humaneval", Score: 0.45},
			},
			threshold: QualityThreshold{MMLUMin: 0.55, HumanEvalMin: 0.35, MinTasks: 2},
			want:     true,
		},
		{
			name: "one fails",
			results: []EvalResult{
				{Task: "mmlu", Score: 0.70},
				{Task: "humaneval", Score: 0.20}, // below 0.35
			},
			threshold: QualityThreshold{MMLUMin: 0.55, HumanEvalMin: 0.35, MinTasks: 2},
			want:     false,
		},
		{
			name: "min tasks met",
			results: []EvalResult{
				{Task: "mmlu", Score: 0.65},
			},
			threshold: QualityThreshold{MMLUMin: 0.55, MinTasks: 1},
			want:     true,
		},
		{
			name: "empty results",
			results: []EvalResult{},
			threshold: QualityThreshold{MinTasks: 1},
			want:     false,
		},
		{
			name: "error results skipped",
			results: []EvalResult{
				{Task: "mmlu", Score: 0.70},
				{Task: "humaneval", Error: "timeout"},
			},
			threshold: QualityThreshold{MMLUMin: 0.55, MinTasks: 1},
			want:     true,
		},
		{
			name: "unknown task always passes",
			results: []EvalResult{
				{Task: "custom_task", Score: 0.01},
			},
			threshold: QualityThreshold{MinTasks: 1},
			want:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := evaluateGate(tt.results, tt.threshold)
			if got != tt.want {
				t.Errorf("evaluateGate() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestProbeModelConfigNonexistent(t *testing.T) {
	dtype, quant, bits := probeModelConfig("nonexistent/model-xyz-12345")
	if dtype != "" {
		t.Errorf("expected empty dtype for nonexistent model, got %q", dtype)
	}
	if quant != "" {
		t.Errorf("expected empty quant for nonexistent model, got %q", quant)
	}
	if bits != 0 {
		t.Errorf("expected 0 bits for nonexistent model, got %d", bits)
	}
}

func TestGateResultPass(t *testing.T) {
	r := &GateResult{
		ModelID: "test/model",
		Passed:  true,
		Results: []EvalResult{
			{Task: "mmlu", Score: 0.65, Passed: true},
		},
	}
	if !r.Passed {
		t.Error("should be passed")
	}
	if r.Results[0].Score != 0.65 {
		t.Errorf("score = %f, want 0.65", r.Results[0].Score)
	}
}
