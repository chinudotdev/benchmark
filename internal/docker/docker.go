// Package docker handles Docker container lifecycle for benchmark runs.
package docker

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/chinudotdev/gpu-benchmark/internal/platform"
)

// Manager handles Docker operations for benchmark containers.
type Manager struct {
	sudo          bool
	activeContainer string
}

// NewManager creates a new Docker manager, detecting if sudo is needed.
func NewManager(ctx context.Context) (*Manager, error) {
	m := &Manager{}
	// Test if docker works without sudo
	cmd := exec.CommandContext(ctx, "docker", "info")
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Run(); err != nil {
		m.sudo = true
		log.Println("Docker requires sudo — prefixing all docker commands with sudo")
	}
	return m, nil
}

// dockerCmd builds a docker command, prepending sudo if needed.
func (m *Manager) dockerCmd(args ...string) *exec.Cmd {
	if m.sudo {
		all := append([]string{"docker"}, args...)
		return exec.Command("sudo", all...)
	}
	return exec.Command("docker", args...)
}

// Pull pulls a Docker image.
func (m *Manager) Pull(ctx context.Context, image string) error {
	log.Printf("Pulling Docker image: %s", image)
	cmd := m.dockerCmd("pull", image)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker pull failed: %w", err)
	}
	log.Printf("Image ready: %s", image)
	return nil
}

// ContainerStatus returns the status of a container, or empty string if not found.
func (m *Manager) ContainerStatus(ctx context.Context, name string) string {
	cmd := m.dockerCmd("inspect", name, "--format={{.State.Status}}")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// Stop forcefully removes a container.
func (m *Manager) Stop(name string) error {
	cmd := m.dockerCmd("rm", "-f", name)
	cmd.Stdout = nil
	cmd.Stderr = nil
	_ = cmd.Run()
	if m.activeContainer == name {
		m.activeContainer = ""
	}
	return nil
}

// Run starts a container with the given config. Returns the container name.
func (m *Manager) Run(ctx context.Context, cfg *platform.ContainerConfig) error {
	args := []string{"run", "-d"}

	// GPU passthrough flags
	args = append(args, cfg.GPUFlags...)

	args = append(args,
		"--ipc", "host",
		"--name", cfg.Name,
		"-p", fmt.Sprintf("%d:8000", cfg.Port),
		"-v", fmt.Sprintf("%s/.cache/huggingface:/root/.cache/huggingface", homeDir()),
	)

	// Environment variables
	for k, v := range cfg.EnvVars {
		args = append(args, "-e", fmt.Sprintf("%s=%s", k, v))
	}

	// Image + model args
	args = append(args, cfg.Image)
	args = append(args, cfg.ExtraArgs...)

	log.Printf("Starting container: %s (image=%s, port=%d)", cfg.Name, cfg.Image, cfg.Port)

	cmd := m.dockerCmd(args...)
	var stderr bytes.Buffer
	cmd.Stdout = nil
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker run failed: %w\nstderr: %s", err, stderr.String())
	}

	m.activeContainer = cfg.Name
	return nil
}

// WaitHealthy waits for the container to respond OK on the health endpoint.
// Returns the wait duration for cold-start measurement.
func (m *Manager) WaitHealthy(ctx context.Context, port int, maxWait time.Duration) (time.Duration, error) {
	containerName := fmt.Sprintf("vllm_bench_%d", port)
	url := fmt.Sprintf("http://localhost:%d/health", port)
	interval := 5 * time.Second
	elapsed := time.Duration(0)

	log.Printf("Waiting for server (max %v)...", maxWait)

	for elapsed < maxWait {
		select {
		case <-ctx.Done():
			return elapsed, ctx.Err()
		default:
		}

		// Check if container crashed
		status := m.ContainerStatus(ctx, containerName)
		if status == "exited" {
			m.dumpLogs(containerName)
			m.Stop(containerName)
			return elapsed, fmt.Errorf("container exited unexpectedly")
		}

		// Health check
		reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		req, _ := httpNewGetWithContext(reqCtx, url)
		resp, err := httpClientDo(req)
		cancel()

		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				log.Printf("Server ready! (%v)", elapsed.Round(time.Second))
				return elapsed, nil
			}
		}

		log.Printf("  Not ready yet (%v) — retrying in %v...", elapsed.Round(time.Second), interval)
		time.Sleep(interval)
		elapsed += interval
	}

	return elapsed, fmt.Errorf("server did not become ready within %v", maxWait)
}

// dumpLogs prints container logs to stderr.
func (m *Manager) dumpLogs(name string) {
	cmd := m.dockerCmd("logs", name)
	out, _ := cmd.CombinedOutput()
	if len(out) > 0 {
		fmt.Println("  ── Container logs ──")
		fmt.Println(string(out))
		fmt.Println("  ── End logs ──")
	}
}

// ActiveContainer returns the name of the currently running container.
func (m *Manager) ActiveContainer() string {
	return m.activeContainer
}

// Cleanup stops the active container. Called on signal interrupts.
func (m *Manager) Cleanup() {
	if m.activeContainer != "" {
		log.Println("Cleaning up container...")
		m.Stop(m.activeContainer)
	}
}

// IsPortInUse checks if something is already listening on a port.
func IsPortInUse(port int) bool {
	url := fmt.Sprintf("http://localhost:%d/health", port)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	req, _ := httpNewGetWithContext(ctx, url)
	resp, err := httpClientDo(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return true
}

func homeDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return h
	}
	return "/root"
}

// ReadLogs streams container logs. Returns a scanner for reading line by line.
func (m *Manager) ReadLogs(ctx context.Context, name string) (*bufio.Scanner, error) {
	cmd := m.dockerCmd("logs", name)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return bufio.NewScanner(stdout), nil
}
