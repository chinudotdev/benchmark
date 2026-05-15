// Package report handles result persistence, summarization, and comparison.
package report

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/chinudotdev/gpu-benchmark/internal/benchmark"
	"github.com/chinudotdev/gpu-benchmark/internal/platform"
)

// BenchmarkResult is the top-level JSON object written per model run.
type BenchmarkResult struct {
	ModelID   string        `json:"model_id"`
	ModelName string        `json:"model_name"`
	Quant     string        `json:"quant"`
	Platform  string        `json:"platform"`   // "nvidia", "amd", "tenstorrent"
	GPU       string        `json:"gpu"`
	GPUCount  int           `json:"gpu_count"`
	GPURate   float64       `json:"gpu_hourly_rate_usd"`

	Benchmark BenchConfig   `json:"benchmark"`
	Metrics   *benchmark.Metrics `json:"metrics"`
	Cost      *CostResult   `json:"cost"`
	System    *SystemInfo   `json:"system"`
	Timestamp string        `json:"timestamp"`
}

// BenchConfig records the benchmark parameters used.
type BenchConfig struct {
	NumPrompts    int    `json:"num_prompts"`
	InputLen      int    `json:"input_len"`
	OutputLen     int    `json:"output_len"`
	MaxModelLen   int    `json:"max_model_len"`
	Concurrency   int    `json:"concurrency"`
	Stream        bool   `json:"stream"`
	WarmupReqs    int    `json:"warmup_reqs"`
	SeqProfile    string `json:"seq_profile,omitempty"`
	TrafficProfile string `json:"traffic_profile,omitempty"`
	RepeatIdx     int    `json:"repeat_idx,omitempty"`
}

// CostResult holds cost estimates.
type CostResult struct {
	CostPer1MTokensUSD *float64 `json:"cost_per_1m_output_tokens_usd"` // null when unavailable
}

// SystemInfo holds the system configuration for reproducibility.
type SystemInfo struct {
	Platform      string              `json:"platform"`
	Devices       []platform.DeviceInfo `json:"devices"`
	CPU           CPUInfo             `json:"cpu"`
	RAM_GB        int                 `json:"ram_gb"`
	OS            string              `json:"os"`
	Kernel        string              `json:"kernel"`
	DockerVersion string              `json:"docker_version"`
	DockerImage   string              `json:"docker_image"`
	DiskAvailGB   int                 `json:"disk_available_gb"`
	CollectedAt   string              `json:"collected_at"`
}

// CPUInfo holds CPU details.
type CPUInfo struct {
	Model string `json:"model"`
	Cores int    `json:"cores"`
}

// WriteResult writes a benchmark result to a JSON file.
func WriteResult(dir string, result *BenchmarkResult) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create results dir: %w", err)
	}

	safeName := strings.NewReplacer("/", "_", "\\", "_", " ", "_").Replace(result.ModelName)
	path := filepath.Join(dir, safeName+".json")

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal result: %w", err)
	}
	data = append(data, '\n')

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write result: %w", err)
	}
	return nil
}

// LoadResults reads all benchmark result JSONs from a directory.
func LoadResults(dir string) ([]*BenchmarkResult, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read results dir: %w", err)
	}

	var results []*BenchmarkResult
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		if entry.Name() == "system_info.json" {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		var result BenchmarkResult
		if err := json.Unmarshal(data, &result); err != nil {
			continue
		}
		results = append(results, &result)
	}

	// Sort by cost ascending
	sort.Slice(results, func(i, j int) bool {
		ci := results[i].Cost.costVal()
		cj := results[j].Cost.costVal()
		return ci < cj
	})

	return results, nil
}

// costVal returns the cost value for sorting (9999 if unavailable).
func (c *CostResult) costVal() float64 {
	if c == nil || c.CostPer1MTokensUSD == nil {
		return 9999
	}
	return *c.CostPer1MTokensUSD
}

// CalculateCost computes cost per 1M output tokens.
func CalculateCost(tps float64, gpuRate float64, gpuCount int) *float64 {
	if tps <= 0 {
		return nil
	}
	totalHourly := gpuRate * float64(gpuCount)
	cost := (totalHourly / tps / 3600) * 1_000_000
	cost = float64(int(cost*10000)) / 10000 // round to 4 decimal places
	return &cost
}

// PrintMetrics prints benchmark metrics to stdout.
func PrintMetrics(m *benchmark.Metrics) {
	line := strings.Repeat("─", 50)
	fmt.Printf("\n%s\n", line)
	fmt.Println("  Results")
	fmt.Println(line)
	fmt.Printf("  Requests:        %d/%d succeeded\n", m.SuccessfulReqs, m.TotalRequests)
	fmt.Printf("  Duration:        %.2fs\n", m.TotalTimeS)
	fmt.Printf("  Output tokens:   %d\n", m.TotalOutputTokens)
	fmt.Printf("  Output TPS:      %.2f tok/s\n", m.OutputTPS)
	fmt.Printf("  RPS:             %.2f\n", m.RPS)
	fmt.Println()

	fmt.Println("  E2E Latency:")
	fmt.Printf("    p50: %s\n", m.LatencyP50.Round(time.Microsecond))
	fmt.Printf("    p95: %s\n", m.LatencyP95.Round(time.Microsecond))
	fmt.Printf("    p99: %s\n", m.LatencyP99.Round(time.Microsecond))

	if m.TTFTP50 > 0 {
		fmt.Println()
		fmt.Println("  TTFT (Time to First Token):")
		fmt.Printf("    p50: %s\n", m.TTFTP50.Round(time.Microsecond))
		fmt.Printf("    p95: %s\n", m.TTFTP95.Round(time.Microsecond))
		fmt.Printf("    p99: %s\n", m.TTFTP99.Round(time.Microsecond))
	}

	if m.TPOTP50 > 0 {
		fmt.Println()
		fmt.Println("  TPOT (Time Per Output Token):")
		fmt.Printf("    p50: %s\n", m.TPOTP50.Round(time.Microsecond))
		fmt.Printf("    p95: %s\n", m.TPOTP95.Round(time.Microsecond))
		fmt.Printf("    p99: %s\n", m.TPOTP99.Round(time.Microsecond))
		fmt.Printf("    t/s/u: %.2f\n", m.TokensPerSecPerUser)
	}

	if len(m.SLA) > 0 {
		fmt.Println()
		fmt.Println("  SLA Compliance:")
		for _, band := range []benchmark.SLABand{
			benchmark.SLABandInteractive,
			benchmark.SLABandConversational,
			benchmark.SLABandBatch,
		} {
			if sla, ok := m.SLA[band]; ok {
				fmt.Printf("    %-15s %d/%d met  (goodput: %.2f tok/s)\n",
					string(band), sla.MetCount, sla.TotalCount, sla.GoodputTPS)
			}
		}
	}

	fmt.Printf("%s\n\n", line)
}
