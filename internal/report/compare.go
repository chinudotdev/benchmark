package report

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/chinudotdev/gpu-benchmark/internal/benchmark"
)

// CompareResult holds the side-by-side comparison of two result directories.
type CompareResult struct {
	DirA      string             `json:"dir_a"`
	DirB      string             `json:"dir_b"`
	ModelName string             `json:"model_name"`
	Cells     []CompareCell      `json:"cells"`
}

// CompareCell compares a single metric across two directories.
type CompareCell struct {
	SeqProfile  string  `json:"seq_profile"`
	InputLen    int     `json:"input_len"`
	OutputLen   int     `json:"output_len"`
	Concurrency int     `json:"concurrency"`

	// A values
	TPSA    float64 `json:"tps_a"`
	TTFTA99 float64 `json:"ttft_p99_a_ms"` // milliseconds
	TPOTA99 float64 `json:"tpot_p99_a_ms"`
	CostA   float64 `json:"cost_per_1m_a"`

	// B values
	TPSB    float64 `json:"tps_b"`
	TTFTB99 float64 `json:"ttft_p99_b_ms"`
	TPOTB99 float64 `json:"tpot_p99_b_ms"`
	CostB   float64 `json:"cost_per_1m_b"`

	// Deltas (B - A)
	TPSDelta    float64 `json:"tps_delta"`     // positive = B faster
	TPSDeltaPct float64 `json:"tps_delta_pct"` // percentage
	CostDelta   float64 `json:"cost_delta"`
	Winner      string  `json:"winner"` // "A", "B", or "tie"
}

// CompareDirs loads results from two directories and produces a comparison.
// It matches results by model name.
func CompareDirs(dirA, dirB string) ([]*CompareResult, error) {
	resultsA, err := LoadResults(dirA)
	if err != nil {
		return nil, fmt.Errorf("load dir A (%s): %w", dirA, err)
	}
	resultsB, err := LoadResults(dirB)
	if err != nil {
		return nil, fmt.Errorf("load dir B (%s): %w", dirB, err)
	}

	if len(resultsA) == 0 && len(resultsB) == 0 {
		return nil, fmt.Errorf("no results found in either directory")
	}

	// Build lookup by model name
	mapA := make(map[string]*BenchmarkResult)
	for _, r := range resultsA {
		mapA[r.ModelName] = r
	}
	mapB := make(map[string]*BenchmarkResult)
	for _, r := range resultsB {
		mapB[r.ModelName] = r
	}

	// Also load sweep results
	sweepsA := loadAllSweepCells(dirA)
	sweepsB := loadAllSweepCells(dirB)

	var comparisons []*CompareResult

	// Compare single-run results
	allModels := make(map[string]bool)
	for name := range mapA {
		allModels[name] = true
	}
	for name := range mapB {
		allModels[name] = true
	}

	for name := range allModels {
		a, hasA := mapA[name]
		b, hasB := mapB[name]

		if !hasA || !hasB {
			continue // can't compare if one side is missing
		}

		cr := &CompareResult{
			DirA:      dirA,
			DirB:      dirB,
			ModelName: name,
		}

		cell := CompareCell{
			InputLen:    a.Benchmark.InputLen,
			OutputLen:   a.Benchmark.OutputLen,
			Concurrency: a.Benchmark.Concurrency,
		}

		if a.Metrics != nil {
			cell.TPSA = a.Metrics.OutputTPS
			cell.TTFTA99 = a.Metrics.TTFTP99.Seconds() * 1000
			cell.TPOTA99 = a.Metrics.TPOTP99.Seconds() * 1000
		}
		if b.Metrics != nil {
			cell.TPSB = b.Metrics.OutputTPS
			cell.TTFTB99 = b.Metrics.TTFTP99.Seconds() * 1000
			cell.TPOTB99 = b.Metrics.TPOTP99.Seconds() * 1000
		}
		if a.Cost != nil && a.Cost.CostPer1MTokensUSD != nil {
			cell.CostA = *a.Cost.CostPer1MTokensUSD
		}
		if b.Cost != nil && b.Cost.CostPer1MTokensUSD != nil {
			cell.CostB = *b.Cost.CostPer1MTokensUSD
		}

		cr.Cells = append(cr.Cells, cell)
		comparisons = append(comparisons, cr)
	}

	// Compare sweep results
	for modelName := range sweepsA {
		cellsA, hasA := sweepsA[modelName]
		cellsB, hasB := sweepsB[modelName]
		if !hasA || !hasB {
			continue
		}

		cr := &CompareResult{
			DirA:      dirA,
			DirB:      dirB,
			ModelName: modelName,
		}

		// Group by (input, output, concurrency) and average
		aggA := aggregateSweepCells(cellsA)
		aggB := aggregateSweepCells(cellsB)

		type key struct {
			inLen, outLen, conc int
			profile             string
		}
		for k, a := range aggA {
			b, ok := aggB[k]
			if !ok {
				continue
			}

			cell := CompareCell{
				SeqProfile:  k.profile,
				InputLen:    k.inLen,
				OutputLen:   k.outLen,
				Concurrency: k.conc,
				TPSA:        a.tpsMean,
				TTFTA99:     a.ttftP99MeanMs,
				TPOTA99:     a.tpotP99MeanMs,
				TPSB:        b.tpsMean,
				TTFTB99:     b.ttftP99MeanMs,
				TPOTB99:     b.tpotP99MeanMs,
			}

			cr.Cells = append(cr.Cells, cell)
		}

		if len(cr.Cells) > 0 {
			comparisons = append(comparisons, cr)
		}
	}

	// Compute deltas and winners
	for _, cr := range comparisons {
		for i := range cr.Cells {
			cell := &cr.Cells[i]
			cell.TPSDelta = cell.TPSB - cell.TPSA
			if cell.TPSA > 0 {
				cell.TPSDeltaPct = (cell.TPSDelta / cell.TPSA) * 100
			}
			cell.CostDelta = cell.CostB - cell.CostA

			// Winner: higher TPS wins, unless tied within 5%
			if math.Abs(cell.TPSDeltaPct) < 5 {
				cell.Winner = "tie"
			} else if cell.TPSDelta > 0 {
				cell.Winner = "B"
			} else {
				cell.Winner = "A"
			}
		}

		// Sort cells by (input_len, output_len, concurrency)
		sort.Slice(cr.Cells, func(i, j int) bool {
			a, b := cr.Cells[i], cr.Cells[j]
			if a.InputLen != b.InputLen {
				return a.InputLen < b.InputLen
			}
			if a.OutputLen != b.OutputLen {
				return a.OutputLen < b.OutputLen
			}
			return a.Concurrency < b.Concurrency
		})
	}

	return comparisons, nil
}

// PrintComparison prints a formatted comparison table.
func PrintComparison(comparisons []*CompareResult) {
	if len(comparisons) == 0 {
		fmt.Println("No matching models found for comparison.")
		return
	}

	for _, cr := range comparisons {
		fmt.Println()
		fmt.Printf("  ══ %s ══\n", cr.ModelName)
		fmt.Printf("  A: %s\n", cr.DirA)
		fmt.Printf("  B: %s\n\n", cr.DirB)

		fmt.Printf("  %-15s %-5s %-5s %-4s  %-10s %-10s  %-10s  %-8s\n",
			"Profile", "In", "Out", "Cnc", "TPS(A)", "TPS(B)", "Δ TPS", "Winner")
		sep := strings.Repeat("─", 85)
		fmt.Printf("  %s\n", sep)

		for _, cell := range cr.Cells {
			profile := cell.SeqProfile
			if profile == "" {
				profile = fmt.Sprintf("in%d-out%d", cell.InputLen, cell.OutputLen)
			}

			tpsA := "N/A"
			tpsB := "N/A"
			delta := ""
			if cell.TPSA > 0 {
				tpsA = fmt.Sprintf("%.1f", cell.TPSA)
			}
			if cell.TPSB > 0 {
				tpsB = fmt.Sprintf("%.1f", cell.TPSB)
			}
			if cell.TPSDeltaPct != 0 {
				delta = fmt.Sprintf("%+.1f%%", cell.TPSDeltaPct)
			}

			winner := cell.Winner
			switch winner {
			case "A":
				winner = "A ✗"
			case "B":
				winner = "B ✓"
			}

			fmt.Printf("  %-15s %-5d %-5d %-4d  %-10s %-10s  %-10s  %-8s\n",
				truncateString(profile, 15), cell.InputLen, cell.OutputLen, cell.Concurrency,
				tpsA, tpsB, delta, winner)
		}

		fmt.Printf("  %s\n", sep)

		// Summary
		aWins, bWins, ties := 0, 0, 0
		for _, cell := range cr.Cells {
			switch cell.Winner {
			case "A":
				aWins++
			case "B":
				bWins++
			default:
				ties++
			}
		}
		fmt.Printf("  Summary: A wins %d, B wins %d, ties %d\n", aWins, bWins, ties)
	}
	fmt.Println()
}

// CompareJSON writes the comparison as JSON.
func CompareJSON(comparisons []*CompareResult) error {
	data, err := json.MarshalIndent(comparisons, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(data))
	return nil
}

// --- internal helpers ---

type sweepAgg struct {
	tpsMean        float64
	ttftP99MeanMs  float64
	tpotP99MeanMs  float64
}

type aggKey struct {
	inLen, outLen, conc int
	profile             string
}

func loadAllSweepCells(resultsDir string) map[string][]*SweepCellResult {
	entries, err := os.ReadDir(resultsDir)
	if err != nil {
		return nil
	}

	result := make(map[string][]*SweepCellResult)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		subdir := filepath.Join(resultsDir, entry.Name())
		cells, err := LoadSweepCells(subdir)
		if err != nil || len(cells) == 0 {
			continue
		}

		modelName := cells[0].ModelName
		// Strip "_sweep" suffix from dir name to get model name
		if modelName == "" {
			modelName = strings.TrimSuffix(entry.Name(), "_sweep")
		}
		result[modelName] = append(result[modelName], cells...)
	}
	return result
}

func aggregateSweepCells(cells []*SweepCellResult) map[aggKey]*sweepAgg {
	type accum struct {
		tpsSum  float64
		ttftSum float64
		tpotSum float64
		n       int
	}

	groups := make(map[aggKey]*accum)
	for _, c := range cells {
		k := aggKey{
			inLen:   c.SweepConfig.InputLen,
			outLen:  c.SweepConfig.OutputLen,
			conc:    c.SweepConfig.Concurrency,
			profile: c.SweepConfig.SeqProfile,
		}
		a, ok := groups[k]
		if !ok {
			a = &accum{}
			groups[k] = a
		}

		// Parse metrics JSON
		var m struct {
			OutputTPS float64 `json:"output_tps"`
			TTFTP99   int64   `json:"ttft_p99"`
			TPOTP99   int64   `json:"tpot_p99"`
		}
		if err := json.Unmarshal(c.Metrics, &m); err != nil || m.OutputTPS == 0 {
			continue
		}
		a.tpsSum += m.OutputTPS
		a.ttftSum += float64(m.TTFTP99) / 1e6 // ns → ms
		a.tpotSum += float64(m.TPOTP99) / 1e6
		a.n++
	}

	result := make(map[aggKey]*sweepAgg)
	for k, a := range groups {
		if a.n == 0 {
			continue
		}
		result[k] = &sweepAgg{
			tpsMean:        a.tpsSum / float64(a.n),
			ttftP99MeanMs:  a.ttftSum / float64(a.n),
			tpotP99MeanMs:  a.tpotSum / float64(a.n),
		}
	}
	return result
}

// truncateString shortens a string to maxLen runes.
func truncateString(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen])
}

// CrossoverAnalysis finds the concurrency level where B overtakes A in TPS.
func CrossoverAnalysis(comparisons []*CompareResult) {
	fmt.Println()
	fmt.Println("  ══ Crossover Analysis ══")
	fmt.Println()

	for _, cr := range comparisons {
		// Group cells by seq profile
		type profileRow struct {
			conc int
			tpsA float64
			tpsB float64
		}
		profiles := make(map[string][]profileRow)
		for _, cell := range cr.Cells {
			p := cell.SeqProfile
			if p == "" {
				p = fmt.Sprintf("in%d-out%d", cell.InputLen, cell.OutputLen)
			}
			profiles[p] = append(profiles[p], profileRow{
				conc: cell.Concurrency,
				tpsA: cell.TPSA,
				tpsB: cell.TPSB,
			})
		}

		fmt.Printf("  Model: %s\n", cr.ModelName)
		for profile, rows := range profiles {
			// Sort by concurrency
			sort.Slice(rows, func(i, j int) bool { return rows[i].conc < rows[j].conc })

			crossover := 0
			for i := 1; i < len(rows); i++ {
				prev := rows[i-1].tpsA - rows[i-1].tpsB // positive = A ahead
				curr := rows[i].tpsA - rows[i].tpsB
				if prev > 0 && curr <= 0 {
					crossover = rows[i].conc
					break
				}
				if prev < 0 && curr >= 0 {
					crossover = rows[i].conc
					break
				}
			}

			if crossover > 0 {
				fmt.Printf("    %-20s B overtakes A at c=%d\n", profile, crossover)
			} else if len(rows) > 0 {
				// Who wins at highest concurrency?
				last := rows[len(rows)-1]
				if last.tpsB > last.tpsA*1.05 {
					fmt.Printf("    %-20s B wins at all levels (best: c=%d)\n", profile, last.conc)
				} else if last.tpsA > last.tpsB*1.05 {
					fmt.Printf("    %-20s A wins at all levels (best: c=%d)\n", profile, last.conc)
				} else {
					fmt.Printf("    %-20s Tied at all levels\n", profile)
				}
			}
		}
		fmt.Println()
	}
}

// SLABandComparison compares SLA goodput across two directories.
type SLABandComparison struct {
	Band      string  `json:"band"`
	GoodputA  float64 `json:"goodput_tps_a"`
	GoodputB  float64 `json:"goodput_tps_b"`
	Winner    string  `json:"winner"`
}

// CompareSLAGoodput compares SLA-band goodput between two result directories.
func CompareSLAGoodput(dirA, dirB string) (map[string][]SLABandComparison, error) {
	resultsA, _ := LoadResults(dirA)
	resultsB, _ := LoadResults(dirB)

	mapA := make(map[string]*BenchmarkResult)
	for _, r := range resultsA {
		mapA[r.ModelName] = r
	}
	mapB := make(map[string]*BenchmarkResult)
	for _, r := range resultsB {
		mapB[r.ModelName] = r
	}

	comparisons := make(map[string][]SLABandComparison)
	for name, a := range mapA {
		b, ok := mapB[name]
		if !ok {
			continue
		}
		if a.Metrics == nil || b.Metrics == nil {
			continue
		}

		var bands []SLABandComparison
		for _, band := range []benchmark.SLABand{
			benchmark.SLABandInteractive,
			benchmark.SLABandConversational,
			benchmark.SLABandBatch,
		} {
			slaA, hasA := a.Metrics.SLA[band]
			slaB, hasB := b.Metrics.SLA[band]
			if !hasA || !hasB {
				continue
			}

			cmp := SLABandComparison{
				Band:     string(band),
				GoodputA: slaA.GoodputTPS,
				GoodputB: slaB.GoodputTPS,
			}
			if math.Abs(cmp.GoodputA-cmp.GoodputB)/math.Max(cmp.GoodputA, 1) < 0.05 {
				cmp.Winner = "tie"
			} else if cmp.GoodputB > cmp.GoodputA {
				cmp.Winner = "B"
			} else {
				cmp.Winner = "A"
			}
			bands = append(bands, cmp)
		}
		if len(bands) > 0 {
			comparisons[name] = bands
		}
	}

	return comparisons, nil
}
