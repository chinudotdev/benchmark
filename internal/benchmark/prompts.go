package benchmark

import (
	"bufio"
	"embed"
	"fmt"
	"io"
	"math/rand"
	"strings"
)

//go:embed prompts.txt
var promptsFS embed.FS

const promptsFile = "prompts.txt"

// PromptDataset holds a corpus of real prompts loaded from the embedded dataset.
type PromptDataset struct {
	prompts []string
	rng     *rand.Rand
}

// LoadPromptDataset loads the embedded prompt dataset.
// The dataset contains one prompt per line.
func LoadPromptDataset() (*PromptDataset, error) {
	f, err := promptsFS.Open(promptsFile)
	if err != nil {
		return nil, fmt.Errorf("open embedded prompts: %w", err)
	}
	defer f.Close()

	var prompts []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(nil, 1024*1024) // 1MB buffer for long lines
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			prompts = append(prompts, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read prompts: %w", err)
	}

	if len(prompts) == 0 {
		return nil, fmt.Errorf("prompt dataset is empty")
	}

	return &PromptDataset{
		prompts: prompts,
		rng:     rand.New(rand.NewSource(42)),
	}, nil
}

// Len returns the number of prompts in the dataset.
func (d *PromptDataset) Len() int {
	return len(d.prompts)
}

// Get returns the prompt at index i.
func (d *PromptDataset) Get(i int) string {
	return d.prompts[i%d.Len()]
}

// Sample returns a random prompt from the dataset.
func (d *PromptDataset) Sample() string {
	return d.prompts[d.rng.Intn(len(d.prompts))]
}

// SampleN returns n random prompts (may contain duplicates if n > dataset size).
func (d *PromptDataset) SampleN(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = d.Sample()
	}
	return out
}

// GeneratePrompts creates n prompts approximately targeting the given token count.
// It selects prompts from the dataset and pads or truncates to approximate the
// target token length (using ~4 chars per token heuristic).
//
// For small target lengths (< 100 tokens), it truncates a random prompt.
// For larger targets, it concatenates prompts until the target is reached.
func (d *PromptDataset) GeneratePrompts(count, targetTokens int) []string {
	result := make([]string, count)
	for i := 0; i < count; i++ {
		result[i] = d.generateOne(targetTokens)
	}
	return result
}

func (d *PromptDataset) generateOne(targetTokens int) string {
	targetChars := targetTokens * 4 // ~4 chars per token

	// For very short targets, just truncate a sample prompt
	if targetChars < 200 {
		p := d.Sample()
		if len(p) > targetChars {
			return p[:targetChars]
		}
		return p
	}

	// For longer targets, concatenate prompts
	var buf strings.Builder
	buf.Grow(targetChars + 256)
	first := true
	for buf.Len() < targetChars {
		if !first {
			buf.WriteString("\n\n")
		}
		buf.WriteString(d.Sample())
		first = false
	}
	return buf.String()
}

// GeneratePromptsFromReader creates a PromptDataset from any reader (for testing).
func GeneratePromptsFromReader(r io.Reader) (*PromptDataset, error) {
	var prompts []string
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			prompts = append(prompts, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return &PromptDataset{
		prompts: prompts,
		rng:     rand.New(rand.NewSource(42)),
	}, nil
}
