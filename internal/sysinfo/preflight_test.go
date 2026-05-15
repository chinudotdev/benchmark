package sysinfo

import (
	"context"
	"testing"
)

func TestPreflightChecks(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	results := PreflightChecks(ctx, dir, 1)
	if len(results) == 0 {
		t.Error("expected at least 1 preflight check")
	}

	// Docker check should always be present
	found := false
	for _, r := range results {
		if r.Name == "docker" {
			found = true
			break
		}
	}
	if !found {
		t.Error("missing docker check")
	}
}

func TestPreflightDiskSpace(t *testing.T) {
	dir := t.TempDir()
	result := checkDiskSpace(dir, 100000) // require absurd amount

	if result.Passed {
		t.Error("should not have 100000GB free")
	}
	if result.Name != "disk-space" {
		t.Errorf("name = %q, want disk-space", result.Name)
	}
}

func TestPreflightHFTokenNotSet(t *testing.T) {
	result := checkHFToken()
	if result.Name != "hf-token" {
		t.Errorf("name = %q, want hf-token", result.Name)
	}
	// Token might or might not be set; just verify it doesn't crash
}

func TestParseSizeToGB(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"100G", 100},
		{"1T", 1024},
		{"500M", 0},
		{"invalid", -1},
		{"50G", 50},
	}

	for _, tt := range tests {
		got := parseSizeToGB(tt.input)
		if got != tt.want {
			t.Errorf("parseSizeToGB(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestPrintPreflight(t *testing.T) {
	results := []PreflightResult{
		{Name: "test", Passed: true, Message: "ok"},
		{Name: "test2", Passed: false, Message: "fail"},
	}
	// Just verify it doesn't panic
	PrintPreflight(results)
}
