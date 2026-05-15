// Package quality implements quality-gate evaluation for benchmark runs.
// It validates that faster tokens aren't worse tokens by running accuracy
// evaluations and precision checks.
package quality

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// GateResult holds the outcome of a quality gate run.
type GateResult struct {
	ModelID    string           `json:"model_id"`
	Passed     bool             `json:"passed"`
	Thresholds QualityThreshold `json:"thresholds"`
	Results    []EvalResult     `json:"results"`
	Precision  PrecisionInfo    `json:"precision"`
	Timestamp  string           `json:"timestamp"`
	Error      string           `json:"error,omitempty"`
}

// QualityThreshold defines the minimum scores required to pass.
type QualityThreshold struct {
	MMLUMin       float64 `json:"mmlu_min"`        // e.g. 0.60
	HumanEvalMin  float64 `json:"humaneval_min"`   // e.g. 0.40
	GSM8KMin      float64 `json:"gsm8k_min"`       // e.g. 0.50
	MinTasks      int     `json:"min_tasks"`       // minimum tasks to attempt (default: 1)
}

// EvalResult holds the result from a single evaluation task.
type EvalResult struct {
	Task      string  `json:"task"`       // e.g. "mmlu", "humaneval", "gsm8k"
	Score     float64 `json:"score"`      // 0-1
	NumSamples int    `json:"num_samples"`
	Passed    bool    `json:"passed"`
	Error     string  `json:"error,omitempty"`
}

// PrecisionInfo describes the actual compute precision used by the model.
type PrecisionInfo struct {
	ComputeDtype string `json:"compute_dtype"` // e.g. "float16", "bfloat16", "float8"
	QuantMethod  string `json:"quant_method"`  // e.g. "none", "awq", "gptq", "int8"
	QuantBits    int    `json:"quant_bits"`     // e.g. 4, 8, 0
	Source       string `json:"source"`         // how we determined this: "config.json", "server_response", "estimated"
}

// GateConfig configures the quality gate run.
type GateConfig struct {
	ModelID    string
	Port       int
	Host       string
	OutputDir  string // directory to write gate results

	// Which tasks to run
	Tasks []string // e.g. ["mmlu", "humaneval", "gsm8k"]. Empty = all available.

	// Thresholds
	Thresholds QualityThreshold

	// lm-evaluation-harness settings
	LMHarnessPath string // path to lm-eval binary or "lm_eval" for pip-installed
	LMEvalArgs    []string // extra args for lm_eval

	// Precision detection
	SkipPrecision bool // skip precision detection

	// Timeout
	Timeout time.Duration // per-task timeout (default: 30 min)
}

// DefaultQualityThreshold returns sensible defaults.
func DefaultQualityThreshold() QualityThreshold {
	return QualityThreshold{
		MMLUMin:      0.55,
		HumanEvalMin: 0.35,
		GSM8KMin:     0.45,
		MinTasks:     1,
	}
}

// RunGate executes the full quality gate pipeline.
func RunGate(ctx context.Context, cfg GateConfig) (*GateResult, error) {
	result := &GateResult{
		ModelID:    cfg.ModelID,
		Thresholds: cfg.Thresholds,
		Timestamp:  time.Now().Format(time.RFC3339),
	}

	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Minute
	}

	// Step 1: Detect precision
	if !cfg.SkipPrecision {
		precision, err := detectPrecision(ctx, cfg.Host, cfg.Port, cfg.ModelID)
		if err != nil {
			log.Printf("Warning: precision detection failed: %v", err)
			precision = PrecisionInfo{Source: "unknown"}
		}
		result.Precision = precision
	}

	// Step 2: Run evaluation tasks
	tasks := cfg.Tasks
	if len(tasks) == 0 {
		tasks = []string{"mmlu", "humaneval", "gsm8k"}
	}

	for _, task := range tasks {
		func() {
			taskCtx, cancel := context.WithTimeout(ctx, cfg.Timeout)
			defer cancel()

			evalResult, err := runTask(taskCtx, task, cfg)
			if err != nil {
				log.Printf("Warning: task %s failed: %v", task, err)
				result.Results = append(result.Results, EvalResult{
					Task:  task,
					Error: err.Error(),
				})
				return
			}
			result.Results = append(result.Results, *evalResult)
		}()
	}

	// Step 3: Determine pass/fail
	result.Passed = evaluateGate(result.Results, cfg.Thresholds)

	// Write result
	if cfg.OutputDir != "" {
		if err := writeGateResult(cfg.OutputDir, result); err != nil {
			log.Printf("Warning: failed to write gate result: %v", err)
		}
	}

	return result, nil
}

// runTask runs a single evaluation task via lm-evaluation-harness.
func runTask(ctx context.Context, task string, cfg GateConfig) (*EvalResult, error) {
	baseURL := fmt.Sprintf("http://%s:%d/v1", cfg.Host, cfg.Port)

	lmEval := cfg.LMHarnessPath
	if lmEval == "" {
		lmEval = "lm_eval"
	}

	// Check if lm_eval is available
	if _, err := exec.LookPath(lmEval); err != nil {
		return nil, fmt.Errorf("lm_eval not found in PATH — install with: pip install lm-eval")
	}

	args := []string{
		"--model", "openai-completions",
		"--model_args", fmt.Sprintf("model=%s,base_url=%s,num_concurrent=1", cfg.ModelID, baseURL),
		"--tasks", task,
		"--output_path", filepath.Join(cfg.OutputDir, "lm_eval_results"),
		"--log_samples",
	}
	args = append(args, cfg.LMEvalArgs...)

	log.Printf("  Running eval: %s %s", lmEval, strings.Join(args, " "))

	cmd := exec.CommandContext(ctx, lmEval, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("lm_eval task %s: %w", task, err)
	}

	// Parse results from lm_eval output
	score, numSamples, err := parseLMEvalOutput(filepath.Join(cfg.OutputDir, "lm_eval_results"), task)
	if err != nil {
		return nil, fmt.Errorf("parse lm_eval output: %w", err)
	}

	return &EvalResult{
		Task:       task,
		Score:      score,
		NumSamples: numSamples,
	}, nil
}

// parseLMEvalOutput reads the JSON results from lm-evaluation-harness output.
func parseLMEvalOutput(outputDir, task string) (float64, int, error) {
	// lm_eval writes results to <output_dir>/results-<timestamp>.json
	// or a nested directory structure
	entries, err := os.ReadDir(outputDir)
	if err != nil {
		return 0, 0, fmt.Errorf("read output dir: %w", err)
	}

	// Find the most recent results file
	var latestFile string
	var latestTime time.Time
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		info, _ := entry.Info()
		if info.ModTime().After(latestTime) {
			latestTime = info.ModTime()
			latestFile = filepath.Join(outputDir, entry.Name())
		}
	}

	if latestFile == "" {
		// Check subdirectories
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			subDir := filepath.Join(outputDir, entry.Name())
			subEntries, err := os.ReadDir(subDir)
			if err != nil {
				continue
			}
			for _, se := range subEntries {
				if se.IsDir() || !strings.HasSuffix(se.Name(), ".json") {
					continue
				}
				info, _ := se.Info()
				if info.ModTime().After(latestTime) {
					latestTime = info.ModTime()
					latestFile = filepath.Join(subDir, se.Name())
				}
			}
		}
	}

	if latestFile == "" {
		return 0, 0, fmt.Errorf("no results JSON found in %s", outputDir)
	}

	data, err := os.ReadFile(latestFile)
	if err != nil {
		return 0, 0, fmt.Errorf("read results: %w", err)
	}

	// lm_eval results format: {"results": {"task_name": {"acc,none": 0.65, ...}}}
	var results map[string]any
	if err := json.Unmarshal(data, &results); err != nil {
		return 0, 0, fmt.Errorf("parse results JSON: %w", err)
	}

	resultsMap, _ := results["results"].(map[string]any)
	taskResults, _ := resultsMap[task].(map[string]any)

	// Try common metric names
	for _, metricKey := range []string{"acc,none", "acc_norm,none", "exact_match,none", "pass@1,none"} {
		if val, ok := taskResults[metricKey].(float64); ok {
			return val, 0, nil
		}
	}

	return 0, 0, fmt.Errorf("no accuracy metric found for task %s", task)
}

// detectPrecision queries the serving endpoint for actual compute dtype.
func detectPrecision(ctx context.Context, host string, port int, modelID string) (PrecisionInfo, error) {
	info := PrecisionInfo{}

	// Try to fetch /v1/models to get model metadata
	url := fmt.Sprintf("http://%s:%d/v1/models", host, port)
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Use exec curl to avoid importing net/http
	cmd := exec.CommandContext(reqCtx, "curl", "-s", url)
	out, err := cmd.Output()
	if err != nil {
		return info, fmt.Errorf("fetch /v1/models: %w", err)
	}

	var modelsResp struct {
		Data []struct {
			ID   string `json:"id"`
			Meta map[string]any `json:"meta,omitempty"`
		} `json:"data"`
	}
	if err := json.Unmarshal(out, &modelsResp); err != nil {
		return info, fmt.Errorf("parse models response: %w", err)
	}

	// Find our model
	for _, m := range modelsResp.Data {
		if m.ID == modelID || strings.Contains(strings.ToLower(m.ID), strings.ToLower(modelID)) {
			info.Source = "server_response"
			break
		}
	}

	// Try to read config.json from the model cache
	if info.ComputeDtype == "" {
		if dtype, quant, bits := probeModelConfig(modelID); dtype != "" {
			info.ComputeDtype = dtype
			info.QuantMethod = quant
			info.QuantBits = bits
			info.Source = "config.json"
		}
	}

	return info, nil
}

// probeModelConfig attempts to read quantization info from the model's config.json.
func probeModelConfig(modelID string) (dtype, quantMethod string, quantBits int) {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}

	// HuggingFace cache structure: hub/models--<org>--<model>/snapshots/<hash>/config.json
	slug := strings.ReplaceAll(modelID, "/", "--")
	modelDir := filepath.Join(home, ".cache", "huggingface", "hub", "models--"+slug)
	snapshotsDir := filepath.Join(modelDir, "snapshots")

	entries, err := os.ReadDir(snapshotsDir)
	if err != nil {
		return
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		configPath := filepath.Join(snapshotsDir, entry.Name(), "config.json")
		data, err := os.ReadFile(configPath)
		if err != nil {
			continue
		}

		var config map[string]any
		if err := json.Unmarshal(data, &config); err != nil {
			continue
		}

		if quant, ok := config["quantization_config"].(map[string]any); ok {
			if m, ok := quant["quant_method"].(string); ok {
				quantMethod = strings.ToLower(m)
			}
			if b, ok := quant["bits"].(float64); ok {
				quantBits = int(b)
			}
		}

		if dt, ok := config["torch_dtype"].(string); ok {
			dtype = dt
		}

		return // use first snapshot found
	}

	return
}

// evaluateGate determines if results meet the quality thresholds.
func evaluateGate(results []EvalResult, thresholds QualityThreshold) bool {
	passCount := 0

	for _, r := range results {
		if r.Error != "" {
			continue
		}

		var minScore float64
		switch r.Task {
		case "mmlu":
			minScore = thresholds.MMLUMin
		case "humaneval":
			minScore = thresholds.HumanEvalMin
		case "gsm8k":
			minScore = thresholds.GSM8KMin
		default:
			minScore = 0
		}

		if r.Score >= minScore {
			passCount++
		}
	}

	return passCount >= thresholds.MinTasks
}

// PrintGateResult prints a human-readable quality gate result.
func PrintGateResult(result *GateResult) {
	status := "✗ FAIL"
	if result.Passed {
		status = "✓ PASS"
	}

	fmt.Println()
	fmt.Printf("  ══ Quality Gate: %s ══\n", status)
	fmt.Println()
	fmt.Printf("  Model:    %s\n", result.ModelID)
	fmt.Printf("  Timestamp: %s\n", result.Timestamp)
	fmt.Println()

	if result.Precision.ComputeDtype != "" {
		fmt.Printf("  Precision: %s", result.Precision.ComputeDtype)
		if result.Precision.QuantMethod != "" && result.Precision.QuantMethod != "none" {
			fmt.Printf(" (%s %d-bit)", result.Precision.QuantMethod, result.Precision.QuantBits)
		}
		fmt.Printf("  [source: %s]\n", result.Precision.Source)
		fmt.Println()
	}

	fmt.Printf("  %-15s %-10s %-10s %-10s\n", "Task", "Score", "Min", "Status")
	fmt.Printf("  %s\n", strings.Repeat("─", 50))

	for _, r := range result.Results {
		status := "✓"
		if r.Error != "" {
			status = "ERROR"
		}

		var minScore float64
		switch r.Task {
		case "mmlu":
			minScore = result.Thresholds.MMLUMin
		case "humaneval":
			minScore = result.Thresholds.HumanEvalMin
		case "gsm8k":
			minScore = result.Thresholds.GSM8KMin
		}

		score := "ERROR"
		if r.Error == "" {
			score = fmt.Sprintf("%.2f%%", r.Score*100)
			if r.Score < minScore {
				status = "✗"
			}
		}

		fmt.Printf("  %-15s %-10s %-10s %-10s\n", r.Task, score,
			fmt.Sprintf("%.0f%%", minScore*100), status)
		if r.Error != "" {
			fmt.Printf("    Error: %s\n", r.Error)
		}
	}

	fmt.Println()
}

func writeGateResult(dir string, result *GateResult) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(dir, "quality_gate.json")
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal gate result: %w", err)
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}
