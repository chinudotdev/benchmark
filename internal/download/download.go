// Package download handles HuggingFace model weight downloading and caching.
package download

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// IsCached checks if a model is already present in the HuggingFace cache.
func IsCached(modelID string) bool {
	cacheDir := hfCacheDir()
	slug := strings.ReplaceAll(modelID, "/", "--")
	modelPath := filepath.Join(cacheDir, "hub", "models--"+slug)

	info, err := os.Stat(modelPath)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// Download downloads a model via huggingface_hub Python library.
func Download(ctx context.Context, modelID, hfToken string) error {
	if IsCached(modelID) {
		log.Printf("Model already cached: %s", modelID)
		return nil
	}

	log.Printf("Downloading %s (this may take a while)...", modelID)

	// Ensure huggingface_hub is installed
	pipCmd := exec.CommandContext(ctx, "python3", "-m", "pip", "install", "-q", "huggingface_hub[hf_xet]")
	pipCmd.Stdout = nil
	pipCmd.Stderr = nil
	if err := pipCmd.Run(); err != nil {
		log.Printf("Warning: pip install huggingface_hub failed: %v", err)
	}

	// Download via Python
	pyScript := fmt.Sprintf(
		`from huggingface_hub import snapshot_download; snapshot_download("%s", resume_download=True)`,
		modelID,
	)
	cmd := exec.CommandContext(ctx, "python3", "-c", pyScript)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Pass HF token
	env := os.Environ()
	if hfToken != "" {
		env = append(env, fmt.Sprintf("HUGGING_FACE_HUB_TOKEN=%s", hfToken))
	}
	cmd.Env = env

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("model download failed: %w\n  Possible causes:\n  • Model ID doesn't exist\n  • Gated model — pass --hf-token", err)
	}

	log.Printf("Download complete: %s", modelID)
	return nil
}

// FixCachePermissions fixes ownership of the HF cache if it's owned by root.
func FixCachePermissions() error {
	cacheDir := hfCacheDir()
	info, err := os.Stat(cacheDir)
	if err != nil || !info.IsDir() {
		return nil // doesn't exist yet, nothing to fix
	}

	whoami, _ := exec.Command("whoami").Output()
	user := strings.TrimSpace(string(whoami))
	if user == "" {
		user = "ubuntu"
	}

	cmd := exec.Command("sudo", "chown", "-R", user+":"+user, cacheDir)
	return cmd.Run()
}

func hfCacheDir() string {
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".cache", "huggingface")
	}
	return filepath.Join("/root", ".cache", "huggingface")
}
