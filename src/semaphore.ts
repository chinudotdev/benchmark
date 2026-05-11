// ── Async semaphore for concurrency limiting ────────────────────────────────

export class Semaphore {
  private queue: (() => void)[] = [];

  constructor(private max: number) {}

  async acquire(): Promise<void> {
    if (this.max > 0) {
      this.max--;
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
      this.max++;
    }
  }
}
