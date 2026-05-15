package benchmark

import (
	"testing"
	"time"
)

func TestMeetsSLA(t *testing.T) {
	tests := []struct {
		name  string
		r     RequestResult
		def   SLADef
		want  bool
	}{
		{
			name: "interactive SLA met",
			r:    RequestResult{Success: true, TTFT: 300 * time.Millisecond, TPOT: 20 * time.Millisecond},
			def:  SLADefs[SLABandInteractive],
			want: true,
		},
		{
			name: "interactive TTFT exceeded",
			r:    RequestResult{Success: true, TTFT: 600 * time.Millisecond, TPOT: 20 * time.Millisecond},
			def:  SLADefs[SLABandInteractive],
			want: false,
		},
		{
			name: "interactive TPOT exceeded",
			r:    RequestResult{Success: true, TTFT: 300 * time.Millisecond, TPOT: 40 * time.Millisecond},
			def:  SLADefs[SLABandInteractive],
			want: false,
		},
		{
			name: "conversational SLA met",
			r:    RequestResult{Success: true, TTFT: 1 * time.Second, TPOT: 80 * time.Millisecond},
			def:  SLADefs[SLABandConversational],
			want: true,
		},
		{
			name: "conversational TTFT exceeded",
			r:    RequestResult{Success: true, TTFT: 3 * time.Second, TPOT: 80 * time.Millisecond},
			def:  SLADefs[SLABandConversational],
			want: false,
		},
		{
			name: "batch SLA always met (no bounds)",
			r:    RequestResult{Success: true, TTFT: 10 * time.Second, TPOT: 500 * time.Millisecond},
			def:  SLADefs[SLABandBatch],
			want: true,
		},
		{
			name: "zero TTFT/TPOT with no bounds",
			r:    RequestResult{Success: true, TTFT: 0, TPOT: 0},
			def:  SLADefs[SLABandBatch],
			want: true,
		},
		{
			name: "zero TTFT/TPOT with interactive SLA — edge case",
			r:    RequestResult{Success: true, TTFT: 0, TPOT: 0},
			def:  SLADefs[SLABandInteractive],
			want: true, // 0 means no measurement (non-streaming), so passes
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := meetsSLA(tt.r, tt.def); got != tt.want {
				t.Errorf("meetsSLA() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAggregateResults(t *testing.T) {
	results := []RequestResult{
		{Success: true, OutputTokens: 100, E2ELatency: 1 * time.Second, TTFT: 100 * time.Millisecond, TPOT: 10 * time.Millisecond},
		{Success: true, OutputTokens: 100, E2ELatency: 2 * time.Second, TTFT: 200 * time.Millisecond, TPOT: 20 * time.Millisecond},
		{Success: true, OutputTokens: 100, E2ELatency: 3 * time.Second, TTFT: 300 * time.Millisecond, TPOT: 30 * time.Millisecond},
		{Success: false, OutputTokens: 0, E2ELatency: 5 * time.Second, Error: "timeout"},
	}

	m := aggregateResults(results, 10*time.Second)

	if m.TotalRequests != 4 {
		t.Errorf("TotalRequests = %d, want 4", m.TotalRequests)
	}
	if m.SuccessfulReqs != 3 {
		t.Errorf("SuccessfulReqs = %d, want 3", m.SuccessfulReqs)
	}
	if m.FailedReqs != 1 {
		t.Errorf("FailedReqs = %d, want 1", m.FailedReqs)
	}
	if m.TotalOutputTokens != 300 {
		t.Errorf("TotalOutputTokens = %d, want 300", m.TotalOutputTokens)
	}
	if m.OutputTPS != 30.0 {
		t.Errorf("OutputTPS = %f, want 30.0", m.OutputTPS)
	}
	if m.RPS != 0.3 {
		t.Errorf("RPS = %f, want 0.3", m.RPS)
	}

	// Check SLA compliance exists
	if len(m.SLA) != 3 {
		t.Errorf("len(SLA) = %d, want 3", len(m.SLA))
	}

	// All 3 successful should meet interactive SLA (TTFT <= 500ms, TPOT <= 30ms)
	if sla, ok := m.SLA[SLABandInteractive]; ok {
		if sla.MetCount != 3 {
			t.Errorf("Interactive SLA MetCount = %d, want 3", sla.MetCount)
		}
		if sla.GoodputTPS != 30.0 {
			t.Errorf("Interactive GoodputTPS = %f, want 30.0", sla.GoodputTPS)
		}
	}
}

func TestAggregateResultsEmpty(t *testing.T) {
	m := aggregateResults(nil, 1*time.Second)
	if m.TotalRequests != 0 {
		t.Errorf("TotalRequests = %d, want 0", m.TotalRequests)
	}
	if m.OutputTPS != 0 {
		t.Errorf("OutputTPS = %f, want 0", m.OutputTPS)
	}
}

func TestAggregateResultsAllFailed(t *testing.T) {
	results := []RequestResult{
		{Success: false, Error: "timeout"},
		{Success: false, Error: "connection refused"},
	}
	m := aggregateResults(results, 5*time.Second)
	if m.SuccessfulReqs != 0 {
		t.Errorf("SuccessfulReqs = %d, want 0", m.SuccessfulReqs)
	}
	if m.OutputTPS != 0 {
		t.Errorf("OutputTPS = %f, want 0", m.OutputTPS)
	}
}

func TestPercentile(t *testing.T) {
	durations := []time.Duration{
		10 * time.Millisecond,
		20 * time.Millisecond,
		30 * time.Millisecond,
		40 * time.Millisecond,
		50 * time.Millisecond,
		60 * time.Millisecond,
		70 * time.Millisecond,
		80 * time.Millisecond,
		90 * time.Millisecond,
		100 * time.Millisecond,
	}

	tests := []struct {
		p    float64
		want time.Duration
	}{
		{0.50, 50 * time.Millisecond},
		{0.95, 100 * time.Millisecond},
		{0.99, 100 * time.Millisecond},
		{0.00, 10 * time.Millisecond},
		{1.00, 100 * time.Millisecond},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got := percentile(durations, tt.p)
			if got != tt.want {
				t.Errorf("percentile(%v) = %v, want %v", tt.p, got, tt.want)
			}
		})
	}
}

func TestPercentileEmpty(t *testing.T) {
	if got := percentile(nil, 0.99); got != 0 {
		t.Errorf("percentile(nil) = %v, want 0", got)
	}
}

func TestMedianDuration(t *testing.T) {
	tests := []struct {
		name string
		d    []time.Duration
		want time.Duration
	}{
		{"odd count", []time.Duration{10, 20, 30}, 20},
		{"even count", []time.Duration{10, 20, 30, 40}, 25},
		{"single", []time.Duration{10}, 10},
		{"empty", nil, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := medianDuration(tt.d)
			if got != tt.want {
				t.Errorf("medianDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGeneratePromptsDeterministic(t *testing.T) {
	p1 := generatePrompts(5, 128)
	p2 := generatePrompts(5, 128)

	if len(p1) != 5 || len(p2) != 5 {
		t.Fatal("expected 5 prompts")
	}
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Errorf("prompt %d differs between calls — not deterministic", i)
		}
	}
}

func TestGeneratePromptsLength(t *testing.T) {
	prompts := generatePrompts(10, 512)
	if len(prompts) != 10 {
		t.Fatalf("expected 10 prompts, got %d", len(prompts))
	}
	// Each prompt should be non-empty
	for i, p := range prompts {
		if len(p) == 0 {
			t.Errorf("prompt %d is empty", i)
		}
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		s      string
		maxLen int
		want   string
	}{
		{"short", 10, "short"},
		{"this is a long string", 10, "this is a "},
		{"exact", 5, "exact"},
		{"", 5, ""},
		{"hello 世界 world", 7, "hello 世"}, // multi-byte safe
	}
	for _, tt := range tests {
		got := truncate(tt.s, tt.maxLen)
		if got != tt.want {
			t.Errorf("truncate(%q, %d) = %q, want %q", tt.s, tt.maxLen, got, tt.want)
		}
	}
}
