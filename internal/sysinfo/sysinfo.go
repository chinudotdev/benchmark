// Package sysinfo provides helpers for system command execution
// and hardware information collection.
package sysinfo

import (
	"bytes"
	"context"
	"os/exec"
	"time"
)

// LookPath checks if a binary exists on PATH.
func LookPath(name string) (string, error) {
	return exec.LookPath(name)
}

// Exec runs a command and returns its stdout. Returns an error if
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

// ExecWithSudo attempts to run a command, retrying with sudo if it fails
// due to permission issues.
func ExecWithSudo(ctx context.Context, name string, args ...string) (string, error) {
	out, err := Exec(ctx, name, args...)
	if err == nil {
		return out, nil
	}
	// Retry with sudo
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

func (e *ExecError) Error() string {
	return e.Err.Error()
}

func (e *ExecError) Unwrap() error {
	return e.Err
}
