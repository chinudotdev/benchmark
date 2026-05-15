package report

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SweepCellResult is the JSON object written per sweep cell.
type SweepCellResult struct {
	ModelID    string `json:"model_id"`
	ModelName  string `json:"model_name"`
	Quant      string `json:"quant"`
	Platform   string `json:"platform"`
	GPU        string `json:"gpu"`
	GPUCount   int    `json:"gpu_count"`
	GPURate    float64 `json:"gpu_hourly_rate_usd"`

	SweepConfig SweepCellConfig `json:"sweep_config"`
	RepeatIdx   int             `json:"repeat_idx"`
	Metrics     json.RawMessage `json:"metrics"` // raw to avoid import cycle
	Error       string          `json:"error,omitempty"`

	System    *SystemInfo `json:"system"`
	Timestamp string      `json:"timestamp"`
}

// SweepCellConfig stores the specific parameters for one sweep cell.
type SweepCellConfig struct {
	InputLen      int    `json:"input_len"`
	OutputLen     int    `json:"output_len"`
	Concurrency   int    `json:"concurrency"`
	NumPrompts    int    `json:"num_prompts"`
	SeqProfile    string `json:"seq_profile,omitempty"`
	TrafficProfile string `json:"traffic_profile,omitempty"`
	Stream        bool   `json:"stream"`
}

// WriteSweepCell writes a single sweep cell result to a JSON file.
// Files are named: <model>_<seq-profile>_c<concurrency>_rep<repeat>.json
func WriteSweepCell(dir string, result *SweepCellResult) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create sweep dir: %w", err)
	}

	filename := sweepCellFilename(result)
	path := filepath.Join(dir, filename)

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal sweep result: %w", err)
	}
	data = append(data, '\n')

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write sweep result: %w", err)
	}
	return nil
}

func sweepCellFilename(r *SweepCellResult) string {
	model := safeModelName(r.ModelName)
	profile := r.SweepConfig.SeqProfile
	if profile == "" {
		profile = fmt.Sprintf("in%d-out%d", r.SweepConfig.InputLen, r.SweepConfig.OutputLen)
	}
	return fmt.Sprintf("%s_%s_c%d_rep%d.json", model, profile, r.SweepConfig.Concurrency, r.RepeatIdx)
}

func safeModelName(name string) string {
	r := strings.NewReplacer("/", "_", "\\", "_", " ", "_")
	return r.Replace(name)
}

// LoadSweepCells reads all sweep cell results from a directory.
func LoadSweepCells(dir string) ([]*SweepCellResult, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read sweep dir: %w", err)
	}

	var results []*SweepCellResult
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

		var result SweepCellResult
		if err := json.Unmarshal(data, &result); err != nil {
			continue
		}
		results = append(results, &result)
	}

	// Sort by (input_len, output_len, concurrency, repeat_idx)
	sort.Slice(results, func(i, j int) bool {
		a, b := results[i], results[j]
		if a.SweepConfig.InputLen != b.SweepConfig.InputLen {
			return a.SweepConfig.InputLen < b.SweepConfig.InputLen
		}
		if a.SweepConfig.OutputLen != b.SweepConfig.OutputLen {
			return a.SweepConfig.OutputLen < b.SweepConfig.OutputLen
		}
		if a.SweepConfig.Concurrency != b.SweepConfig.Concurrency {
			return a.SweepConfig.Concurrency < b.SweepConfig.Concurrency
		}
		return a.RepeatIdx < b.RepeatIdx
	})

	return results, nil
}

// SweepSummary is a summary of aggregated sweep results.
type SweepSummary struct {
	ModelID   string           `json:"model_id"`
	ModelName string           `json:"model_name"`
	Cells     []SweepCellSummary `json:"cells"`
}

// SweepCellSummary holds aggregated results for one cell (mean ± stddev across repeats).
type SweepCellSummary struct {
	SeqProfile    string  `json:"seq_profile"`
	InputLen      int     `json:"input_len"`
	OutputLen     int     `json:"output_len"`
	Concurrency   int     `json:"concurrency"`
	Repeats       int     `json:"repeats"`
	TPSMean       float64 `json:"tps_mean"`
	TPSStddev     float64 `json:"tps_stddev"`
	TTFTP99MeanMs float64 `json:"ttft_p99_mean_ms"`
	TPOTP99MeanMs float64 `json:"tpot_p99_mean_ms"`
	GoodputTPS    float64 `json:"goodput_tps_interactive"`
}
