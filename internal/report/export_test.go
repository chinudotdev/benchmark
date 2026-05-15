package report

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExportBundleEmpty(t *testing.T) {
	dir := t.TempDir()
	output := filepath.Join(t.TempDir(), "export.tar.gz")

	path, err := ExportBundle(dir, output)
	if err != nil {
		t.Fatalf("ExportBundle error: %v", err)
	}
	if path != output {
		t.Errorf("path = %q, want %q", path, output)
	}

	// Verify file exists
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("archive not created: %v", err)
	}
}

func TestExportBundleWithResults(t *testing.T) {
	dir := t.TempDir()

	// Write some test files
	os.WriteFile(filepath.Join(dir, "model-7b.json"), []byte(`{"model_id":"test"}`), 0o644)
	os.WriteFile(filepath.Join(dir, "system_info.json"), []byte(`{"platform":"nvidia"}`), 0o644)
	os.WriteFile(filepath.Join(dir, "report.md"), []byte("# Report\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "ignored.log"), []byte("log data"), 0o644) // should be excluded

	output := filepath.Join(t.TempDir(), "export.tar.gz")
	path, err := ExportBundle(dir, output)
	if err != nil {
		t.Fatalf("ExportBundle error: %v", err)
	}

	// List contents
	files, err := ListBundle(path)
	if err != nil {
		t.Fatalf("ListBundle error: %v", err)
	}

	// Should have 3 JSON/MD files + manifest
	found := make(map[string]bool)
	for _, f := range files {
		found[f] = true
	}

	if !found["model-7b.json"] {
		t.Error("missing model-7b.json")
	}
	if !found["system_info.json"] {
		t.Error("missing system_info.json")
	}
	if !found["report.md"] {
		t.Error("missing report.md")
	}
	if !found["manifest.json"] {
		t.Error("missing manifest.json")
	}
	if found["ignored.log"] {
		t.Error("ignored.log should not be in archive")
	}
}

func TestExportBundleAutoName(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "test.json"), []byte(`{}`), 0o644)

	path, err := ExportBundle(dir, "")
	if err != nil {
		t.Fatalf("ExportBundle error: %v", err)
	}
	if path == "" {
		t.Error("expected auto-generated path")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("archive not created: %v", err)
	}
}

func TestExportBundleNonexistentDir(t *testing.T) {
	_, err := ExportBundle("/nonexistent/dir", "")
	if err == nil {
		t.Error("expected error for nonexistent dir")
	}
}
