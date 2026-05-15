package report

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// SummarizeFormat controls the output format for the summarize command.
type SummarizeFormat string

const (
	FormatTable SummarizeFormat = "table"
	FormatCSV   SummarizeFormat = "csv"
	FormatJSON  SummarizeFormat = "json"
)

// Summarize prints results in the requested format.
func Summarize(dir string, format SummarizeFormat) error {
	results, err := LoadResults(dir)
	if err != nil || len(results) == 0 {
		fmt.Printf("No results in %s\n", dir)
		return nil
	}

	switch format {
	case FormatTable:
		printTable(results)
	case FormatCSV:
		printCSV(results)
	case FormatJSON:
		printJSON(results)
	default:
		return fmt.Errorf("unknown format: %s", format)
	}
	return nil
}

func printTable(results []*BenchmarkResult) {
	fmt.Println()
	fmt.Println("  GPU Benchmark Cost Summary")
	fmt.Println()

	// Column definitions
	headers := []string{"Model", "GPU", "Quant", "GPUs", "GPU rate", "TPS", "t/s/u", "$/1M tok", "Date"}
	widths := make([]int, len(headers))
	for i, h := range headers {
		widths[i] = len(h)
	}

	rows := make([][]string, len(results))
	for i, r := range results {
		tpsStr := fmt.Sprintf("%.2f", r.Metrics.OutputTPS)
		tsuStr := ""
		if r.Metrics.TokensPerSecPerUser > 0 {
			tsuStr = fmt.Sprintf("%.2f", r.Metrics.TokensPerSecPerUser)
		}
		costStr := "N/A"
		if r.Cost != nil && r.Cost.CostPer1MTokensUSD != nil {
			costStr = fmt.Sprintf("$%.4f", *r.Cost.CostPer1MTokensUSD)
		}
		dateStr := ""
		if len(r.Timestamp) >= 10 {
			dateStr = r.Timestamp[:10]
		}

		row := []string{
			r.ModelName,
			r.GPU,
			r.Quant,
			fmt.Sprintf("%d", r.GPUCount),
			fmt.Sprintf("$%.2f/hr", r.GPURate),
			tpsStr,
			tsuStr,
			costStr,
			dateStr,
		}
		rows[i] = row

		for j, val := range row {
			if len(val) > widths[j] {
				widths[j] = len(val)
			}
		}
	}

	// Print separator
	sep := make([]string, len(headers))
	for i, w := range widths {
		sep[i] = strings.Repeat("─", w)
	}
	sepLine := "  " + strings.Join(sep, "  ")

	fmt.Println(sepLine)

	// Header
	fmt.Print("  ")
	for i, h := range headers {
		fmt.Printf("%-*s", widths[i], h)
		if i < len(headers)-1 {
			fmt.Print("  ")
		}
	}
	fmt.Println()
	fmt.Println(sepLine)

	// Rows
	for _, row := range rows {
		fmt.Print("  ")
		for j, val := range row {
			fmt.Printf("%-*s", widths[j], val)
			if j < len(row)-1 {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}

	fmt.Println(sepLine)
	fmt.Println()
}

func printCSV(results []*BenchmarkResult) {
	w := csv.NewWriter(os.Stdout)
	defer w.Flush()

	w.Write([]string{
		"model_name", "model_id", "platform", "gpu", "quant",
		"gpu_count", "gpu_rate_usd", "tps", "t_s_u", "cost_per_1m_usd",
		"ttft_p99", "tpot_p99", "timestamp",
	})

	for _, r := range results {
		cost := ""
		if r.Cost != nil && r.Cost.CostPer1MTokensUSD != nil {
			cost = fmt.Sprintf("%.4f", *r.Cost.CostPer1MTokensUSD)
		}
		tps := ""
		if r.Metrics != nil {
			tps = fmt.Sprintf("%.2f", r.Metrics.OutputTPS)
		}
		tsu := ""
		if r.Metrics != nil && r.Metrics.TokensPerSecPerUser > 0 {
			tsu = fmt.Sprintf("%.2f", r.Metrics.TokensPerSecPerUser)
		}
		ttft := ""
		if r.Metrics != nil && r.Metrics.TTFTP99 > 0 {
			ttft = fmt.Sprintf("%d", r.Metrics.TTFTP99.Microseconds())
		}
		tpot := ""
		if r.Metrics != nil && r.Metrics.TPOTP99 > 0 {
			tpot = fmt.Sprintf("%d", r.Metrics.TPOTP99.Microseconds())
		}

		w.Write([]string{
			r.ModelName, r.ModelID, r.Platform, r.GPU, r.Quant,
			fmt.Sprintf("%d", r.GPUCount),
			fmt.Sprintf("%.2f", r.GPURate),
			tps, tsu, cost,
			ttft, tpot, r.Timestamp,
		})
	}
}

func printJSON(results []*BenchmarkResult) {
	data, _ := json.MarshalIndent(results, "", "  ")
	fmt.Println(string(data))
}
