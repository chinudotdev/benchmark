package report

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ExportBundle creates a reproducible gzipped tar archive of the results directory.
// The archive includes:
//   - All JSON result files
//   - system_info.json
//   - report.md (if it exists)
//   - A manifest.json with metadata about the export
//
// Returns the path to the created archive.
func ExportBundle(resultsDir string, output string) (string, error) {
	// Resolve output path
	if output == "" {
		output = filepath.Join(resultsDir,
			fmt.Sprintf("gpu-benchmark-export-%s.tar.gz",
				time.Now().Format("2006-01-02-150405")))
	}

	// Ensure results dir exists
	info, err := os.Stat(resultsDir)
	if err != nil {
		return "", fmt.Errorf("results dir: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s is not a directory", resultsDir)
	}

	// Create output file
	f, err := os.Create(output)
	if err != nil {
		return "", fmt.Errorf("create archive: %w", err)
	}
	defer f.Close()

	gw := gzip.NewWriter(f)
	defer gw.Close()
	tw := tar.NewWriter(gw)
	defer tw.Close()

	// Track what we included
	var includedFiles []string

	// Walk the results directory
	err = filepath.Walk(resultsDir, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories (they're implied by file paths)
		if fi.IsDir() {
			return nil
		}

		name := filepath.Base(path)
		ext := filepath.Ext(name)

		// Only include relevant files
		switch {
		case ext == ".json":
			// All JSON files
		case ext == ".md":
			// Markdown reports
		case ext == ".csv":
			// CSV exports
		default:
			return nil // skip other files
		}

		// Make path relative to results dir
		relPath, err := filepath.Rel(resultsDir, path)
		if err != nil {
			return err
		}

		// Write tar header
		header, err := tar.FileInfoHeader(fi, "")
		if err != nil {
			return err
		}
		header.Name = relPath
		// Make reproducible: zero out timestamps
		header.ModTime = time.Time{}
		header.AccessTime = time.Time{}
		header.ChangeTime = time.Time{}
		if err := tw.WriteHeader(header); err != nil {
			return fmt.Errorf("write header %s: %w", relPath, err)
		}

		// Write file content
		src, err := os.Open(path)
		if err != nil {
			return err
		}

		if _, err := io.Copy(tw, src); err != nil {
			src.Close()
			return fmt.Errorf("write %s: %w", relPath, err)
		}
		src.Close()

		includedFiles = append(includedFiles, relPath)
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("walk results dir: %w", err)
	}

	// Write manifest
	manifest := map[string]any{
		"version":     "1.0",
		"exported_at": time.Now().Format(time.RFC3339),
		"tool":        "gpu-benchmark",
		"files":       includedFiles,
	}
	manifestJSON, _ := json.MarshalIndent(manifest, "", "  ")
	manifestJSON = append(manifestJSON, '\n')

	tw.WriteHeader(&tar.Header{
		Name:     "manifest.json",
		Size:     int64(len(manifestJSON)),
		Mode:     0o644,
		ModTime:  time.Time{},
	})
	tw.Write(manifestJSON)

	// Flush
	if err := tw.Close(); err != nil {
		return "", fmt.Errorf("close tar: %w", err)
	}
	if err := gw.Close(); err != nil {
		return "", fmt.Errorf("close gzip: %w", err)
	}

	abs, _ := filepath.Abs(output)
	return abs, nil
}

// ListBundle reads a tar.gz export and lists its contents.
func ListBundle(archivePath string) ([]string, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gr.Close()

	tr := tar.NewReader(gr)

	var files []string
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if !strings.HasSuffix(header.Name, "/") {
			files = append(files, header.Name)
		}
	}

	return files, nil
}
