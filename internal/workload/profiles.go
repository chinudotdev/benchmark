package workload

import (
	"github.com/chinudotdev/gpu-benchmark/internal/benchmark"
)

// SeqProfile defines a named input/output token length pair.
type SeqProfile struct {
	Name        string `json:"name"`
	InputTokens int    `json:"input_tokens"`
	OutputTokens int   `json:"output_tokens"`
	Description string `json:"description"`
}

// Sequence-length profiles from the benchmarking framework spec (Section 4.3).
var SeqProfiles = []SeqProfile{
	{
		Name:         "short-chat",
		InputTokens:  128,
		OutputTokens: 128,
		Description:  "Balanced; matches common chat traffic",
	},
	{
		Name:         "long-input-rag",
		InputTokens:  4000,
		OutputTokens: 256,
		Description:  "Prefill-heavy — large-context retrieval workloads",
	},
	{
		Name:         "long-output-reasoning",
		InputTokens:  512,
		OutputTokens: 4000,
		Description:  "Decode-heavy — bandwidth-bound generation",
	},
	{
		Name:         "very-long-context",
		InputTokens:  16000,
		OutputTokens: 64,
		Description:  "KV-cache pressure and memory headroom limits",
	},
	{
		Name:         "summarisation",
		InputTokens:  770,
		OutputTokens: 73,
		Description:  "MLPerf CNN/DailyMail summarisation profile",
	},
}

// DefaultSeqProfile returns the default profile (short-chat).
func DefaultSeqProfile() SeqProfile {
	return SeqProfiles[0]
}

// TrafficProfile defines a traffic shape for benchmark runs.
type TrafficProfile struct {
	Name        string                  `json:"name"`
	Description string                  `json:"description"`
	SLABand     benchmark.SLABand       `json:"sla_band"`
}

// Traffic profiles from the benchmarking framework spec (Section 4.2).
var TrafficProfiles = []TrafficProfile{
	{
		Name:        "single-stream",
		Description: "One request at a time, no batching",
		SLABand:     benchmark.SLABandInteractive,
	},
	{
		Name:        "interactive",
		Description: "Poisson arrivals, Interactive SLA enforced",
		SLABand:     benchmark.SLABandInteractive,
	},
	{
		Name:        "high-concurrency",
		Description: "Saturating concurrent load, Conversational SLA",
		SLABand:     benchmark.SLABandConversational,
	},
	{
		Name:        "offline-batch",
		Description: "No latency bound, max throughput",
		SLABand:     benchmark.SLABandBatch,
	},
}
