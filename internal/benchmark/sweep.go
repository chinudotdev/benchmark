package benchmark

import (
	"context"
	"fmt"
	"log"
	"math"
	"strings"
	"time"
)

// SweepConfig configures a sweep run.
type SweepConfig struct {
	Host        string
	Port        int
	Model       string
	Stream      bool
	Retries     int
	WarmupReqs  int

	// What to sweep
	ConcurrencyLevels []int   // e.g. [1, 2, 4, 8, 16, 32, 64, 128]
	InputLen          int     // used if not doing seq sweep
	OutputLen         int     // used if not doing seq sweep
	NumPrompts        int

	// Repeat
	Repeat int // number of times to repeat each cell

	// Prompt dataset
	Prompts *PromptDataset
}

// SweepResult holds the result for a single sweep cell.
type SweepResult struct {
	Concurrency int            `json:"concurrency"`
	InputLen    int            `json:"input_len"`
	OutputLen   int            `json:"output_len"`
	RepeatIdx   int            `json:"repeat_idx"`
	Metrics     *Metrics       `json:"metrics"`
	Error       string         `json:"error,omitempty"`
}

// RunConcurrencySweep runs benchmarks across multiple concurrency levels.
// For each concurrency level, it repeats Repeat times and returns all results.
func RunConcurrencySweep(ctx context.Context, cfg SweepConfig) ([]SweepResult, error) {
	var results []SweepResult
	total := len(cfg.ConcurrencyLevels) * cfg.Repeat
	completed := 0

	for _, conc := range cfg.ConcurrencyLevels {
		for rep := 0; rep < cfg.Repeat; rep++ {
			select {
			case <-ctx.Done():
				return results, ctx.Err()
			default:
			}

			cellTag := fmt.Sprintf("c=%d rep=%d/%d", conc, rep+1, cfg.Repeat)
			log.Printf("  [%s] %d/%d cells — starting", cellTag, completed+1, total)

			rcfg := RunnerConfig{
				Host:        cfg.Host,
				Port:        cfg.Port,
				Model:       cfg.Model,
				NumPrompts:  cfg.NumPrompts,
				InputLen:    cfg.InputLen,
				OutputLen:   cfg.OutputLen,
				Concurrency: conc,
				Stream:      cfg.Stream,
				Retries:     cfg.Retries,
				WarmupReqs:  cfg.WarmupReqs,
			}

			metrics, err := Run(ctx, rcfg)
			completed++

			sr := SweepResult{
				Concurrency: conc,
				InputLen:    cfg.InputLen,
				OutputLen:   cfg.OutputLen,
				RepeatIdx:   rep,
			}
			if err != nil {
				sr.Error = err.Error()
				log.Printf("  [%s] FAILED: %v", cellTag, err)
			} else {
				sr.Metrics = metrics
				tps := 0.0
				if metrics != nil {
					tps = metrics.OutputTPS
			}
				log.Printf("  [%s] done — %.1f tok/s", cellTag, tps)
			}

			results = append(results, sr)

			// Brief settle between repeats
			if rep < cfg.Repeat-1 {
				time.Sleep(2 * time.Second)
			}
		}
	}

	return results, nil
}

// RunSeqSweep runs benchmarks across sequence-length profiles.
// For each profile, it runs all concurrency levels and repeats.
func RunSeqSweep(ctx context.Context, cfg SweepConfig, profiles []SeqProfileEntry) ([]SweepResult, error) {
	var results []SweepResult
	total := len(profiles) * len(cfg.ConcurrencyLevels) * cfg.Repeat
	completed := 0

	for _, profile := range profiles {
		for _, conc := range cfg.ConcurrencyLevels {
			for rep := 0; rep < cfg.Repeat; rep++ {
				select {
				case <-ctx.Done():
					return results, ctx.Err()
				default:
				}

				cellTag := fmt.Sprintf("%s/c=%d/rep=%d", profile.Name, conc, rep+1)
				log.Printf("  [%s] %d/%d — starting", cellTag, completed+1, total)

				rcfg := RunnerConfig{
					Host:        cfg.Host,
					Port:        cfg.Port,
					Model:       cfg.Model,
					NumPrompts:  cfg.NumPrompts,
					InputLen:    profile.InputTokens,
					OutputLen:   profile.OutputTokens,
					Concurrency: conc,
					Stream:      cfg.Stream,
					Retries:     cfg.Retries,
					WarmupReqs:  cfg.WarmupReqs,
				}

				metrics, err := Run(ctx, rcfg)
				completed++

				sr := SweepResult{
					Concurrency: conc,
					InputLen:    profile.InputTokens,
					OutputLen:   profile.OutputTokens,
					RepeatIdx:   rep,
				}
				if err != nil {
					sr.Error = err.Error()
					log.Printf("  [%s] FAILED: %v", cellTag, err)
				} else {
					sr.Metrics = metrics
					tps := 0.0
					if metrics != nil {
						tps = metrics.OutputTPS
					}
					log.Printf("  [%s] done — %.1f tok/s", cellTag, tps)
				}

				results = append(results, sr)

				// Brief settle between repeats
				if rep < cfg.Repeat-1 {
					time.Sleep(2 * time.Second)
				}
			}
		}
	}

	return results, nil
}

// SeqProfileEntry mirrors workload.SeqProfile to avoid circular import.
type SeqProfileEntry struct {
	Name         string
	InputTokens  int
	OutputTokens int
}

// AggregateSweep aggregates repeated sweep results into mean ± stddev for TPS.
func AggregateSweep(results []SweepResult) map[string]*SweepAggregate {
	// Group by (concurrency, input_len, output_len)
	type key struct {
		conc, inLen, outLen int
	}
	groups := make(map[key][]SweepResult)
	for _, r := range results {
		k := key{r.Concurrency, r.InputLen, r.OutputLen}
		groups[k] = append(groups[k], r)
	}

	aggregated := make(map[string]*SweepAggregate)
	for k, rs := range groups {
		cellLabel := fmt.Sprintf("c=%d/in=%d/out=%d", k.conc, k.inLen, k.outLen)
		agg := &SweepAggregate{
			Concurrency: k.conc,
			InputLen:    k.inLen,
			OutputLen:   k.outLen,
			Repeats:     len(rs),
		}

		var tpsSum, ttftSum, tpotSum float64
		var tpsVals []float64
		validCount := 0

		for _, r := range rs {
			if r.Metrics == nil {
				continue
			}
			validCount++
			tps := r.Metrics.OutputTPS
			tpsSum += tps
			tpsVals = append(tpsVals, tps)
			ttftSum += r.Metrics.TTFTP99.Seconds()
			tpotSum += r.Metrics.TPOTP99.Seconds()
		}

		if validCount > 0 {
			agg.TPSMean = tpsSum / float64(validCount)
			agg.TTFTP99Mean = ttftSum / float64(validCount)
			agg.TPOTP99Mean = tpotSum / float64(validCount)

			if len(tpsVals) > 1 {
				variance := 0.0
				for _, v := range tpsVals {
					diff := v - agg.TPSMean
					variance += diff * diff
				}
				agg.TPSStddev = math.Sqrt(variance / float64(len(tpsVals)-1))
			}
		}

		aggregated[cellLabel] = agg
	}

	return aggregated
}

// SweepAggregate holds aggregated sweep results with statistics.
type SweepAggregate struct {
	Concurrency int     `json:"concurrency"`
	InputLen    int     `json:"input_len"`
	OutputLen   int     `json:"output_len"`
	Repeats     int     `json:"repeats"`
	TPSMean     float64 `json:"tps_mean"`
	TPSStddev   float64 `json:"tps_stddev"`
	TTFTP99Mean float64 `json:"ttft_p99_mean_s"`
	TPOTP99Mean float64 `json:"tpot_p99_mean_s"`
}

func (a *SweepAggregate) String() string {
	tps := fmt.Sprintf("%.1f", a.TPSMean)
	if a.TPSStddev > 0 {
		tps = fmt.Sprintf("%.1f ± %.1f", a.TPSMean, a.TPSStddev)
	}
	return fmt.Sprintf("c=%-3d in=%-5d out=%-5d → %s tok/s  (n=%d)", a.Concurrency, a.InputLen, a.OutputLen, tps, a.Repeats)
}

// PrintSweepTable prints a formatted table of sweep results.
func PrintSweepTable(aggregated map[string]*SweepAggregate) {
	if len(aggregated) == 0 {
		return
	}

	// Collect and sort keys
	var aggs []*SweepAggregate
	for _, a := range aggregated {
		aggs = append(aggs, a)
	}

	line := strings.Repeat("─", 80)
	fmt.Println()
	fmt.Println("  Sweep Results")
	fmt.Println(line)

	for _, a := range aggs {
		fmt.Printf("  %s\n", a.String())
	}

	fmt.Println(line)
	fmt.Println()
}
