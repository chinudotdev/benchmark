// ── Colored logging ─────────────────────────────────────────────────────────

const RED = "\x1b[0;31m";
const GREEN = "\x1b[0;32m";
const YELLOW = "\x1b[1;33m";
const BLUE = "\x1b[0;34m";
const CYAN = "\x1b[0;36m";
const BOLD = "\x1b[1m";
const NC = "\x1b[0m";

function ts(): string {
  return new Date().toTimeString().split(" ")[0]!;
}

export function log(msg: string): void {
  console.log(`${CYAN}[${ts()}]${NC} ${msg}`);
}

export function success(msg: string): void {
  console.log(`${GREEN}[${ts()}] ✓${NC} ${msg}`);
}

export function warn(msg: string): void {
  console.log(`${YELLOW}[${ts()}] ⚠${NC} ${msg}`);
}

export function error(msg: string): void {
  console.error(`${RED}[${ts()}] ✗${NC} ${msg}`);
}

export function header(msg: string): void {
  const line = "═".repeat(50);
  console.log(`\n${BOLD}${BLUE}${line}${NC}`);
  console.log(`${BOLD}${BLUE}  ${msg}${NC}`);
  console.log(`${BOLD}${BLUE}${line}${NC}\n`);
}

export function bold(msg: string): string {
  return `${BOLD}${msg}${NC}`;
}

export function green(msg: string): string {
  return `${GREEN}${msg}${NC}`;
}
