// Package sysinfo provides helpers for system command execution
// and full hardware/OS information collection.
package sysinfo

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// LookPath checks if a binary exists on PATH.
func LookPath(name string) (string, error) {
	return exec.LookPath(name)
}

// Exec runs a command and returns its trimmed stdout. Returns an error if
// the command exits non-zero or is not found.
func Exec(ctx context.Context, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, name, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return stdout.String(), &ExecError{
			Cmd:    name,
			Args:   args,
			Err:    err,
			Stderr: stderr.String(),
		}
	}
	return string(bytes.TrimSpace(stdout.Bytes())), nil
}

// ExecWithSudo attempts to run a command, retrying with sudo if it fails.
func ExecWithSudo(ctx context.Context, name string, args ...string) (string, error) {
	out, err := Exec(ctx, name, args...)
	if err == nil {
		return out, nil
	}
	sudoArgs := append([]string{name}, args...)
	return Exec(ctx, "sudo", sudoArgs...)
}

// ExecError provides details about a failed command execution.
type ExecError struct {
	Cmd    string
	Args   []string
	Err    error
	Stderr string
}

func (e *ExecError) Error() string { return e.Err.Error() }
func (e *ExecError) Unwrap() error { return e.Err }

// ── Full system info collection ────────────────────────────────────────────

// Info holds the full system configuration for reproducibility.
type Info struct {
	CPU           CPUInfo  `json:"cpu"`
	RAM_GB        int      `json:"ram_gb"`
	OS            string   `json:"os"`
	Kernel        string   `json:"kernel"`
	DockerVersion string   `json:"docker_version"`
	DiskAvailGB   int      `json:"disk_available_gb"`
}

// CPUInfo holds CPU details.
type CPUInfo struct {
	Model string `json:"model"`
	Cores int    `json:"cores"`
}

// Collect gathers CPU, RAM, OS, kernel, Docker, and disk information.
func Collect(ctx context.Context) *Info {
	info := &Info{}
	info.CPU = collectCPU(ctx)
	info.RAM_GB = collectRAM(ctx)
	info.OS = collectOS(ctx)
	info.Kernel = collectKernel(ctx)
	info.DockerVersion = collectDockerVersion(ctx)
	info.DiskAvailGB = collectDiskGB()
	return info
}

// PrintPretty writes a human-readable system info summary.
func PrintPretty(info *Info, dockerImage string) {
	bold := "\033[1m"
	reset := "\033[0m"

	fmt.Printf("  %sCPU:%s      %s (%d cores)\n", bold, reset, info.CPU.Model, info.CPU.Cores)
	fmt.Printf("  %sRAM:%s      %dGB\n", bold, reset, info.RAM_GB)
	fmt.Printf("  %sOS:%s       %s\n", bold, reset, info.OS)
	fmt.Printf("  %sKernel:%s   %s\n", bold, reset, info.Kernel)
	fmt.Printf("  %sDocker:%s   %s\n", bold, reset, info.DockerVersion)
	if dockerImage != "" {
		fmt.Printf("  %sImage:%s    %s\n", bold, reset, dockerImage)
	}
	fmt.Printf("  %sDisk:%s     %dGB available\n", bold, reset, info.DiskAvailGB)
	fmt.Println()
}

// ── Individual collectors ──────────────────────────────────────────────────

func collectCPU(ctx context.Context) CPUInfo {
	info := CPUInfo{Model: "unknown", Cores: 0}

	if runtime.GOOS == "linux" {
		// Parse /proc/cpuinfo
		if f, err := os.Open("/proc/cpuinfo"); err == nil {
			defer f.Close()
			scanner := bufio.NewScanner(f)
			cores := 0
			for scanner.Scan() {
				line := scanner.Text()
				if strings.HasPrefix(line, "model name") {
					parts := strings.SplitN(line, ":", 2)
					if len(parts) == 2 {
						info.Model = strings.TrimSpace(parts[1])
					}
				}
				if strings.HasPrefix(line, "processor") {
					cores++
				}
			}
			info.Cores = cores
		}
	}

	if runtime.GOOS == "darwin" {
		if out, err := Exec(ctx, "sysctl", "-n", "machdep.cpu.brand_string"); err == nil {
			info.Model = out
		}
		if out, err := Exec(ctx, "sysctl", "-n", "hw.ncpu"); err == nil {
			if n, err := strconv.Atoi(out); err == nil {
				info.Cores = n
			}
		}
	}

	// Fallback for cores
	if info.Cores == 0 {
		if out, err := Exec(ctx, "nproc"); err == nil {
			if n, err := strconv.Atoi(out); err == nil {
				info.Cores = n
			}
		}
	}

	return info
}

func collectRAM(ctx context.Context) int {
	if runtime.GOOS == "linux" {
		if f, err := os.Open("/proc/meminfo"); err == nil {
			defer f.Close()
			scanner := bufio.NewScanner(f)
			if scanner.Scan() {
				line := scanner.Text()
				re := regexp.MustCompile(`MemTotal:\s*(\d+)\s*kB`)
				if m := re.FindStringSubmatch(line); len(m) > 1 {
					if kb, err := strconv.ParseInt(m[1], 10, 64); err == nil {
						return int(kb / 1024 / 1024)
					}
				}
			}
		}
	}

	if runtime.GOOS == "darwin" {
		if out, err := Exec(ctx, "sysctl", "-n", "hw.memsize"); err == nil {
			if bytes, err := strconv.ParseInt(out, 10, 64); err == nil {
				return int(bytes / 1024 / 1024 / 1024)
			}
		}
	}

	return 0
}

func collectOS(ctx context.Context) string {
	if runtime.GOOS == "linux" {
		if data, err := os.ReadFile("/etc/os-release"); err == nil {
			content := string(data)
			re := regexp.MustCompile(`(?m)^PRETTY_NAME="(.+)"`)
			if m := re.FindStringSubmatch(content); len(m) > 1 {
				return m[1]
			}
			// Fallback: assemble from NAME + VERSION
			nameRe := regexp.MustCompile(`(?m)^NAME="(.+)"`)
			verRe := regexp.MustCompile(`(?m)^VERSION="(.+)"`)
			name := firstMatch(nameRe, content)
			ver := firstMatch(verRe, content)
			if name != "" {
				parts := []string{name}
				if ver != "" {
					parts = append(parts, ver)
				}
				return strings.Join(parts, " ")
			}
		}
	}

	if runtime.GOOS == "darwin" {
		if out, err := Exec(ctx, "sw_vers"); err == nil {
			var prod, ver string
			for _, line := range strings.Split(out, "\n") {
				if strings.HasPrefix(line, "ProductName:") {
					prod = strings.TrimSpace(strings.TrimPrefix(line, "ProductName:"))
				}
				if strings.HasPrefix(line, "ProductVersion:") {
					ver = strings.TrimSpace(strings.TrimPrefix(line, "ProductVersion:"))
				}
			}
			if prod != "" {
				parts := []string{prod}
				if ver != "" {
					parts = append(parts, ver)
				}
				return strings.Join(parts, " ")
			}
		}
	}

	return runtime.GOOS
}

func collectKernel(ctx context.Context) string {
	out, err := Exec(ctx, "uname", "-r")
	if err != nil {
		return "unknown"
	}
	return out
}

func collectDockerVersion(ctx context.Context) string {
	out, err := Exec(ctx, "docker", "version", "--format", "{{.Server.Version}}")
	if err != nil {
		// Try with sudo
		out, err = Exec(ctx, "sudo", "docker", "version", "--format", "{{.Server.Version}}")
		if err != nil {
			return "unknown"
		}
	}
	return out
}

func collectDiskGB() int {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "/root"
	}
	target := filepath.Join(home, ".cache", "huggingface")
	if _, err := os.Stat(target); err != nil {
		target = home
	}

	// Try df -BG (Linux: GB units), then -g (macOS/BSD), then -h
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	for _, flag := range []string{"-BG", "-g", "-h"} {
		out, err := Exec(ctx, "df", flag, target)
		if err != nil {
			continue
		}
		lines := strings.Split(out, "\n")
		if len(lines) < 2 {
			continue
		}
		parts := strings.Fields(lines[1])
		if len(parts) < 4 {
			continue
		}
		avail := parts[3]
		// Strip unit suffix (G, Gi, etc.)
		cleaned := regexp.MustCompile(`[A-Za-z]+$`).ReplaceAllString(avail, "")
		if gb, err := strconv.Atoi(cleaned); err == nil {
			return gb
		}
		if gb, err := strconv.ParseFloat(cleaned, 64); err == nil {
			return int(gb)
		}
	}
	return 0
}

func firstMatch(re *regexp.Regexp, s string) string {
	m := re.FindStringSubmatch(s)
	if len(m) > 1 {
		return m[1]
	}
	return ""
}


