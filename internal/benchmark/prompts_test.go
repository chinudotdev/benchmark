package benchmark

import (
	"strings"
	"testing"
)

func TestLoadPromptDataset(t *testing.T) {
	ds, err := LoadPromptDataset()
	if err != nil {
		t.Fatalf("LoadPromptDataset() error: %v", err)
	}
	if ds.Len() == 0 {
		t.Fatal("dataset is empty")
	}
	t.Logf("Loaded %d prompts", ds.Len())
}

func TestPromptDatasetGet(t *testing.T) {
	ds, err := LoadPromptDataset()
	if err != nil {
		t.Fatal(err)
	}

	// Get should wrap around
	first := ds.Get(0)
	if first == "" {
		t.Error("Get(0) returned empty string")
	}

	// Wrap around
	wrapped := ds.Get(ds.Len())
	if wrapped != first {
		t.Error("Get(len) should wrap to Get(0)")
	}
}

func TestPromptDatasetSample(t *testing.T) {
	ds, err := LoadPromptDataset()
	if err != nil {
		t.Fatal(err)
	}

	s := ds.Sample()
	if s == "" {
		t.Error("Sample() returned empty string")
	}
}

func TestPromptDatasetSampleN(t *testing.T) {
	ds, err := LoadPromptDataset()
	if err != nil {
		t.Fatal(err)
	}

	samples := ds.SampleN(5)
	if len(samples) != 5 {
		t.Fatalf("SampleN(5) returned %d samples", len(samples))
	}
	for i, s := range samples {
		if s == "" {
			t.Errorf("SampleN(5)[%d] is empty", i)
		}
	}
}

func TestPromptDatasetGeneratePromptsLength(t *testing.T) {
	ds, err := LoadPromptDataset()
	if err != nil {
		t.Fatal(err)
	}

	prompts := ds.GeneratePrompts(10, 256)
	if len(prompts) != 10 {
		t.Fatalf("expected 10 prompts, got %d", len(prompts))
	}

	for i, p := range prompts {
		if p == "" {
			t.Errorf("prompt %d is empty", i)
		}
		// Each prompt should be roughly targetTokens * 4 chars
		// Allow wide tolerance since we're concatenating real sentences
		if len(p) == 0 {
			t.Errorf("prompt %d has zero length", i)
		}
	}
}

func TestPromptDatasetGeneratePromptsShort(t *testing.T) {
	ds, err := LoadPromptDataset()
	if err != nil {
		t.Fatal(err)
	}

	// Very short target (e.g. 10 tokens → ~40 chars)
	prompts := ds.GeneratePrompts(5, 10)
	if len(prompts) != 5 {
		t.Fatalf("expected 5 prompts, got %d", len(prompts))
	}
	for _, p := range prompts {
		if len(p) == 0 {
			t.Error("short prompt is empty")
		}
	}
}

func TestPromptDatasetGeneratePromptsLong(t *testing.T) {
	ds, err := LoadPromptDataset()
	if err != nil {
		t.Fatal(err)
	}

	// Long target (e.g. 4000 tokens → ~16000 chars)
	prompts := ds.GeneratePrompts(3, 4000)
	if len(prompts) != 3 {
		t.Fatalf("expected 3 prompts, got %d", len(prompts))
	}
	for _, p := range prompts {
		if len(p) < 1000 {
			t.Errorf("long prompt too short: %d chars", len(p))
		}
	}
}

func TestGeneratePromptsFromReader(t *testing.T) {
	input := "First prompt line.\n\nSecond prompt line.\n\nThird prompt line.\n"
	ds, err := GeneratePromptsFromReader(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if ds.Len() != 3 {
		t.Fatalf("expected 3 prompts, got %d", ds.Len())
	}
}
