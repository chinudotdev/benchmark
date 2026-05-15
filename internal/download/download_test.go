package download

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsCachedNonexistent(t *testing.T) {
	cached := IsCached("nonexistent/model-xyz-12345")
	if cached {
		t.Error("should not be cached")
	}
}

func TestFixCachePermissionsNonexistent(t *testing.T) {
	// Should not error when cache dir doesn't exist
	err := FixCachePermissions()
	if err != nil {
		t.Errorf("FixCachePermissions should not error on nonexistent dir: %v", err)
	}
}

func TestFixCachePermissionsWritable(t *testing.T) {
	// Create a temp cache dir and verify it's detected as writable
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, ".cache", "huggingface"), 0o755)

	// This won't use our temp dir (it uses ~/.cache/huggingface),
	// but the function should not panic
	err := FixCachePermissions()
	if err != nil {
		t.Errorf("FixCachePermissions error: %v", err)
	}
}

func TestHFDir(t *testing.T) {
	dir := hfCacheDir()
	if dir == "" {
		t.Error("hfCacheDir returned empty")
	}
	// Should end with huggingface
	if filepath.Base(dir) != "huggingface" {
		t.Errorf("expected base to be 'huggingface', got %q", filepath.Base(dir))
	}
}
