// Package workload handles model registry loading and prompt generation.
package workload

import (
	"fmt"
	"os"

	"github.com/chinudotdev/gpu-benchmark/internal/platform"
	"gopkg.in/yaml.v3"
)

// ModelRegistry is the top-level YAML structure for models.yaml.
type ModelRegistry struct {
	Models []platform.ModelConfig `yaml:"models"`
}

// LoadModels reads the model registry from a YAML file.
func LoadModels(path string) ([]platform.ModelConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read models yaml: %w", err)
	}

	var registry ModelRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		return nil, fmt.Errorf("parse models yaml: %w", err)
	}

	// Validate entries
	for i, m := range registry.Models {
		if m.ID == "" {
			return nil, fmt.Errorf("model entry %d: missing id", i)
		}
		if m.Name == "" {
			registry.Models[i].Name = m.ID // default name to ID
		}
		if m.TP == 0 {
			registry.Models[i].TP = 1
		}
		if m.Quant == "" {
			registry.Models[i].Quant = "none"
		}
	}

	return registry.Models, nil
}

// FindModel returns a single model by ID, constructing a ModelConfig
// with defaults if not in the registry.
func FindModel(modelID, quant string) platform.ModelConfig {
	name := modelID
	if parts := splitLast(modelID, "/"); len(parts) == 2 {
		name = parts[1]
	}
	q := quant
	if q == "" {
		q = "none"
	}
	return platform.ModelConfig{
		ID:         modelID,
		Name:       name,
		Quant:      q,
		MinVRAM_GB: 0,
		TP:         1,
		ExtraFlags: "",
	}
}

func splitLast(s, sep string) []string {
	idx := lastIndex(s, sep)
	if idx < 0 {
		return []string{s}
	}
	return []string{s[:idx], s[idx+1:]}
}

func lastIndex(s, sep string) int {
	for i := len(s) - len(sep); i >= 0; i-- {
		if s[i:i+len(sep)] == sep {
			return i
		}
	}
	return -1
}
