package report

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteAndLoadSweepCells(t *testing.T) {
	dir := t.TempDir()

	cells := []*SweepCellResult{
		{
			ModelID: "test/model-7b", ModelName: "model-7b",
			SweepConfig: SweepCellConfig{InputLen: 128, OutputLen: 128, Concurrency: 32},
			RepeatIdx: 0,
			Metrics:   json.RawMessage(`{"output_tps": 100.5}`),
		},
		{
			ModelID: "test/model-7b", ModelName: "model-7b",
			SweepConfig: SweepCellConfig{InputLen: 128, OutputLen: 128, Concurrency: 32},
			RepeatIdx: 1,
			Metrics:   json.RawMessage(`{"output_tps": 105.2}`),
		},
		{
			ModelID: "test/model-7b", ModelName: "model-7b",
			SweepConfig: SweepCellConfig{InputLen: 4000, OutputLen: 256, Concurrency: 8},
			RepeatIdx: 0,
			Metrics:   json.RawMessage(`{"output_tps": 40.0}`),
		},
	}

	for _, cell := range cells {
		if err := WriteSweepCell(dir, cell); err != nil {
			t.Fatalf("WriteSweepCell error: %v", err)
		}
	}

	loaded, err := LoadSweepCells(dir)
	if err != nil {
		t.Fatalf("LoadSweepCells error: %v", err)
	}

	if len(loaded) != 3 {
		t.Fatalf("expected 3 cells, got %d", len(loaded))
	}

	// Verify sort order: (input_len, output_len, concurrency, repeat_idx)
	// Cells: c=32/in=128 (x2), c=8/in=4000 (x1)
	// Sorted: in=128/c=32/rep=0, in=128/c=32/rep=1, in=4000/c=8/rep=0
	if loaded[0].SweepConfig.InputLen != 128 || loaded[0].RepeatIdx != 0 {
		t.Errorf("first cell should be in=128/rep=0, got in=%d/rep=%d", loaded[0].SweepConfig.InputLen, loaded[0].RepeatIdx)
	}
	if loaded[1].SweepConfig.InputLen != 128 || loaded[1].RepeatIdx != 1 {
		t.Errorf("second cell should be in=128/rep=1, got in=%d/rep=%d", loaded[1].SweepConfig.InputLen, loaded[1].RepeatIdx)
	}
	if loaded[2].SweepConfig.InputLen != 4000 || loaded[2].SweepConfig.Concurrency != 8 {
		t.Errorf("third cell should be in=4000/c=8, got in=%d/c=%d", loaded[2].SweepConfig.InputLen, loaded[2].SweepConfig.Concurrency)
	}
}

func TestSweepCellFilename(t *testing.T) {
	cases := []struct {
		cell     *SweepCellResult
		contains string
	}{
		{
			cell: &SweepCellResult{
				ModelName: "Qwen3-8B",
				SweepConfig: SweepCellConfig{
					SeqProfile:  "short-chat",
					Concurrency: 32,
				},
				RepeatIdx: 0,
			},
			contains: "Qwen3-8B_short-chat_c32_rep0.json",
		},
		{
			cell: &SweepCellResult{
				ModelName: "Qwen3-8B",
				SweepConfig: SweepCellConfig{
					InputLen:    512,
					OutputLen:   256,
					Concurrency: 16,
				},
				RepeatIdx: 2,
			},
			contains: "Qwen3-8B_in512-out256_c16_rep2.json",
		},
	}

	for _, tc := range cases {
		got := sweepCellFilename(tc.cell)
		if got != tc.contains {
			t.Errorf("filename = %q, want %q", got, tc.contains)
		}
	}
}

func TestLoadSweepCellsEmptyDir(t *testing.T) {
	dir := t.TempDir()
	loaded, err := LoadSweepCells(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(loaded) != 0 {
		t.Errorf("expected 0 cells from empty dir, got %d", len(loaded))
	}
}

func TestLoadSweepCellsSkipsSystemInfo(t *testing.T) {
	dir := t.TempDir()
	// Write system_info.json (should be skipped)
	os.WriteFile(filepath.Join(dir, "system_info.json"), []byte("{}"), 0o644)
	// Write a valid sweep cell
	cell := &SweepCellResult{
		ModelID: "test/model", ModelName: "model",
		SweepConfig: SweepCellConfig{InputLen: 128, OutputLen: 128, Concurrency: 1},
		Metrics: json.RawMessage(`{}`),
	}
	WriteSweepCell(dir, cell)

	loaded, err := LoadSweepCells(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded) != 1 {
		t.Errorf("expected 1 cell (system_info.json skipped), got %d", len(loaded))
	}
}
