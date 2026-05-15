package sysinfo

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// PreflightResult holds the result of a preflight check.
type PreflightResult struct {
	Name    string `json:"name"`
	Passed  bool   `json:"passed"`
	Message string `json:"message"`
}

// PreflightChecks runs all pre-benchmark checks and returns results.
// Returns an error only if a critical check fails.
func PreflightChecks(ctx context.Context, resultsDir string, minDiskGB int) []PreflightResult {
	var results []PreflightResult

	// Docker availability
	results = append(results, checkDocker(ctx))

	// Disk space
	results = append(results, checkDiskSpace(resultsDir, minDiskGB))

	// nvidia-container-toolkit (if on Linux with NVIDIA)
	results = append(results, checkGPUToolkit(ctx))

	// Python availability (for model downloads)
	results = append(results, checkPython(ctx))

	// Port availability (common default)
	results = append(results, checkHFToken())

	return results
}

func checkDocker(ctx context.Context) PreflightResult {
	cmd := exec.CommandContext(ctx, "docker", "version", "--format", "{{.Client.Version}}")
	out, err := cmd.Output()
	if err != nil {
		return PreflightResult{
			Name:    "docker",
			Passed:  false,
			Message: "Docker is not installed or not in PATH",
		}
	}
	version := strings.TrimSpace(string(out))
	return PreflightResult{
		Name:    "docker",
		Passed:  true,
		Message: fmt.Sprintf("Docker client %s", version),
	}
}

func checkDiskSpace(resultsDir string, minGB int) PreflightResult {
	var path string
	if resultsDir != "" {
		path = resultsDir
	} else {
		path = "."
	}

	// Try to create the results dir first (to check the right filesystem)
	os.MkdirAll(path, 0o755)

	avail := diskAvailGB(path)
	if avail < 0 {
		return PreflightResult{
			Name:    "disk-space",
			Passed:  true, // can't determine, don't block
			Message: "Could not determine available disk space",
		}
	}

	if avail < minGB {
		return PreflightResult{
			Name:    "disk-space",
			Passed:  false,
			Message: fmt.Sprintf("Only %dGB available (need %dGB for model weights + results)", avail, minGB),
		}
	}

	return PreflightResult{
		Name:    "disk-space",
		Passed:  true,
		Message: fmt.Sprintf("%dGB available", avail),
	}
}

func checkGPUToolkit(ctx context.Context) PreflightResult {
	// Only relevant on Linux
	if !isLinux() {
		return PreflightResult{
			Name:    "gpu-toolkit",
			Passed:  true,
			Message: "Skipped (not Linux)",
		}
	}

	// Check for nvidia-container-toolkit
	cmd := exec.CommandContext(ctx, "nvidia-container-cli", "--version")
	if err := cmd.Run(); err != nil {
		// Try alternative check
		cmd := exec.CommandContext(ctx, "dpkg", "-l", "nvidia-container-toolkit")
		if err := cmd.Run(); err != nil {
			return PreflightResult{
				Name:    "gpu-toolkit",
				Passed:  false,
				Message: "nvidia-container-toolkit not found — needed for GPU passthrough",
			}
		}
	}

	return PreflightResult{
		Name:    "gpu-toolkit",
		Passed:  true,
		Message: "nvidia-container-toolkit installed",
	}
}

func checkPython(ctx context.Context) PreflightResult {
	cmd := exec.CommandContext(ctx, "python3", "--version")
	out, err := cmd.Output()
	if err != nil {
		return PreflightResult{
			Name:    "python",
			Passed:  false,
			Message: "python3 not found — needed for HuggingFace model downloads",
		}
	}
	return PreflightResult{
		Name:    "python",
		Passed:  true,
		Message: strings.TrimSpace(string(out)),
	}
}

func checkHFToken() PreflightResult {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	if token == "" {
		return PreflightResult{
			Name:    "hf-token",
			Passed:  true, // not blocking — only needed for gated models
			Message: "Not set (only needed for gated models)",
		}
	}
	return PreflightResult{
		Name:    "hf-token",
		Passed:  true,
		Message: fmt.Sprintf("Set (%d chars)", len(token)),
	}
}

// PrintPreflight prints preflight check results.
func PrintPreflight(results []PreflightResult) {
	fmt.Println()
	fmt.Println("  ══ Preflight Checks ══")
	fmt.Println()

	allPassed := true
	for _, r := range results {
		status := "✓"
		if !r.Passed {
			status = "✗"
			allPassed = false
		}
		fmt.Printf("  %s %-20s %s\n", status, r.Name, r.Message)
	}

	fmt.Println()

	if !allPassed {
		fmt.Println("  ⚠ Some checks failed. Fix them before running benchmarks.")
		fmt.Println()
	}
}

// diskAvailGB returns available disk space in GB for the given path.
func diskAvailGB(path string) int {
	// Use `df` command for portability
	cmd := exec.Command("df", "-g", path)
	out, err := cmd.Output()
	if err != nil {
		// Fallback: try -h flag and parse
		cmd = exec.Command("df", "-h", path)
		out, err = cmd.Output()
		if err != nil {
			return -1
		}
		// Parse "Filesystem Size Used Avail ..."
		lines := strings.Split(string(out), "\n")
		if len(lines) >= 2 {
			fields := strings.Fields(lines[1])
			if len(fields) >= 4 {
				avail := fields[3]
				return parseSizeToGB(avail)
			}
		}
		return -1
	}

	// Parse "Filesystem 1G-blocks Used Available ..."
	lines := strings.Split(string(out), "\n")
	if len(lines) >= 2 {
		fields := strings.Fields(lines[1])
		if len(fields) >= 4 {
			gb, err := strconv.Atoi(fields[3])
			if err == nil {
				return gb
			}
		}
	}
	return -1
}

func parseSizeToGB(s string) int {
	s = strings.ToUpper(strings.TrimSpace(s))
	if strings.HasSuffix(s, "G") {
		gb, err := strconv.Atoi(strings.TrimSuffix(s, "G"))
		if err == nil {
			return gb
		}
	}
	if strings.HasSuffix(s, "T") {
		tb, err := strconv.Atoi(strings.TrimSuffix(s, "T"))
		if err == nil {
			return tb * 1024
		}
	}
	if strings.HasSuffix(s, "M") {
		mb, err := strconv.Atoi(strings.TrimSuffix(s, "M"))
		if err == nil {
			return mb / 1024
		}
	}
	return -1
}

func isLinux() bool {
	return runtime.GOOS == "linux"
}
