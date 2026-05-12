// ── Custom errors ────────────────────────────────────────────────────────────
//
// FatalError: thrown to signal the process should exit.
// Caught at the top-level CLI entry point so library code stays testable.

export class FatalError extends Error {
  constructor(message?: string) {
    super(message ?? "Fatal error");
    this.name = "FatalError";
  }
}
