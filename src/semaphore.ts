// ── Async semaphore for concurrency limiting ────────────────────────────────

export class Semaphore {
  private queue: (() => void)[] = [];
  private _available: number;

  constructor(max: number) {
    this._available = max;
  }

  async acquire(): Promise<void> {
    if (this._available > 0) {
      this._available--;
      return;
    }
    return new Promise<void>((resolve) => {
      this.queue.push(resolve);
    });
  }

  release(): void {
    const next = this.queue.shift();
    if (next) {
      next();
    } else {
      this._available++;
    }
  }

  /** Current number of available permits (for diagnostics). */
  get available(): number {
    return this._available;
  }
}
