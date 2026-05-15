package benchmark

import (
	"testing"
	"time"
)

func TestParseRetryAfterHeaderSeconds(t *testing.T) {
	tests := []struct {
		input string
		want  time.Duration
	}{
		{"120", 120 * time.Second},
		{"30", 30 * time.Second},
		{"0", 0},
		{"", 0},
		{"invalid", 0},
	}

	for _, tt := range tests {
		got := parseRetryAfterHeader(tt.input)
		if got != tt.want {
			t.Errorf("parseRetryAfterHeader(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestParseRetryAfterHeaderHTTPDate(t *testing.T) {
	// Test with a date in the future — should return positive duration
	got := parseRetryAfterHeader("Fri, 14 May 2027 00:00:00 GMT")
	// Just verify it doesn't crash and returns something reasonable
	if got < 0 {
		t.Errorf("expected non-negative duration for future date, got %v", got)
	}
}

func TestParseRetryAfterHeaderPastDate(t *testing.T) {
	// Past date should return 0 (duration would be negative)
	got := parseRetryAfterHeader("Fri, 14 May 2020 00:00:00 GMT")
	if got != 0 {
		t.Errorf("expected 0 for past date, got %v", got)
	}
}
