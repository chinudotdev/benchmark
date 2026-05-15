// Package benchmark defines metric types and the HTTP load generator
// that drives requests against an OpenAI-compatible serving endpoint.
package benchmark

import "time"

// SLABand represents a named latency SLA tier.
type SLABand string

const (
	SLABandInteractive   SLABand = "interactive"
	SLABandConversational SLABand = "conversational"
	SLABandBatch         SLABand = "batch"
)

// SLADef defines the latency targets for an SLA band.
type SLADef struct {
	TTFTTarget time.Duration // P99 TTFT target (0 = no bound)
	TPOTTarget time.Duration // P99 TPOT target (0 = no bound)
}

// SLADefs maps band names to their targets.
var SLADefs = map[SLABand]SLADef{
	SLABandInteractive: {
		TTFTTarget: 500 * time.Millisecond,
		TPOTTarget: 30 * time.Millisecond,
	},
	SLABandConversational: {
		TTFTTarget: 2 * time.Second,
		TPOTTarget: 100 * time.Millisecond,
	},
	SLABandBatch: {}, // no bounds
}

// RequestResult holds timing data for a single benchmark request.
type RequestResult struct {
	Success       bool          `json:"success"`
	OutputTokens  int           `json:"output_tokens"`
	TTFT          time.Duration `json:"ttft"`             // 0 if non-streaming
	TPOT          time.Duration `json:"tpot"`             // median inter-token latency
	E2ELatency    time.Duration `json:"e2e_latency"`
	TokenCount    int           `json:"token_count"`      // number of decode tokens received
	Error         string        `json:"error,omitempty"`
}

// Metrics holds aggregated benchmark results.
type Metrics struct {
	TotalRequests     int                    `json:"total_requests"`
	SuccessfulReqs    int                    `json:"successful_requests"`
	FailedReqs        int                    `json:"failed_requests"`
	TotalTimeS        float64                `json:"total_time_s"`
	TotalOutputTokens int64                  `json:"total_output_tokens"`
	OutputTPS         float64                `json:"output_tps"`
	RPS               float64                `json:"rps"` // requests per second

	// Latency distributions (E2E)
	LatencyP50 time.Duration `json:"latency_p50"`
	LatencyP95 time.Duration `json:"latency_p95"`
	LatencyP99 time.Duration `json:"latency_p99"`

	// TTFT (streaming only)
	TTFTP50 time.Duration `json:"ttft_p50"`
	TTFTP95 time.Duration `json:"ttft_p95"`
	TTFTP99 time.Duration `json:"ttft_p99"`

	// TPOT (streaming only)
	TPOTP50 time.Duration `json:"tpot_p50"`
	TPOTP95 time.Duration `json:"tpot_p95"`
	TPOTP99 time.Duration `json:"tpot_p99"`

	// Per-user decode rate
	TokensPerSecPerUser float64 `json:"tokens_per_sec_per_user"`

	// SLA compliance
	SLA map[SLABand]*SLAResult `json:"sla"`

	// Cold start (measured externally, set by orchestration)
	ColdStartS float64 `json:"cold_start_s,omitempty"`
}

// SLAResult holds throughput filtered by SLA compliance.
type SLAResult struct {
	Band        SLABand `json:"band"`
	MetCount    int     `json:"met_count"`
	TotalCount  int     `json:"total_count"`
	GoodputTPS  float64 `json:"goodput_tps"`
}

// RunnerConfig configures the benchmark load generator.
type RunnerConfig struct {
	Host        string
	Port        int
	Model       string
	NumPrompts  int
	InputLen    int
	OutputLen   int
	Concurrency int
	Stream      bool
	Retries     int
	WarmupReqs  int // number of warmup requests to discard
}
