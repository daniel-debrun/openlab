/**
 * The token, and the one way to reach the workspace server.
 *
 * `openlab app` prints a tokened URL (`http://127.0.0.1:8321/?t=…`). The token is
 * captured once, stashed in sessionStorage, and stripped from the address bar —
 * so it is not in the history, not in a bookmark, and not in a screenshot of the
 * URL bar — then sent as `Authorization: Bearer <t>` on every request. Both
 * sockets take it as a query parameter instead, because a WebSocket handshake
 * cannot carry a header.
 *
 * There are no private powers here. Everything this UI can do is a POST to
 * /api/tool/{name}, which is the same dispatch the CLI and the agent's MCP call.
 */

const KEY = "openlab_token";

function captureToken(): string {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get("t");
  if (fromUrl) {
    sessionStorage.setItem(KEY, fromUrl);
    params.delete("t");
    const qs = params.toString();
    window.history.replaceState(
      {},
      "",
      window.location.pathname + (qs ? `?${qs}` : "") + window.location.hash,
    );
  }
  return sessionStorage.getItem(KEY) ?? "";
}

export const TOKEN = captureToken();

/**
 * The reviewer secret (`&r=` in the URL `openlab app` prints). The agent terminal never receives it, so a
 * request that carries it is the person at this browser: only such a request may approve or sign a claim.
 */
const RKEY = "openlab_reviewer";
function captureReviewer(): string {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get("r");
  if (fromUrl) {
    sessionStorage.setItem(RKEY, fromUrl);
    params.delete("r");
    const qs = params.toString();
    window.history.replaceState({}, "", window.location.pathname + (qs ? `?${qs}` : "") + window.location.hash);
  }
  return sessionStorage.getItem(RKEY) ?? "";
}
export const REVIEWER = captureReviewer();

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, body: unknown) {
    super(`API ${status}`);
    this.status = status;
    this.body = body;
  }
}

/**
 * Token loss is unrecoverable in-page, so it gets its own signal.
 *
 * The token is minted per boot unless `OPENLAB_APP_TOKEN` pins it. Restart the
 * server and every request 401s with no way for the page to re-acquire it;
 * without this that shows up as a screen full of identical failed panels and no
 * explanation. The shell subscribes and says what actually happened.
 */
type Listener = () => void;
const tokenLostWatchers = new Set<Listener>();
let tokenLost = false;

export function onTokenLost(fn: Listener): () => void {
  tokenLostWatchers.add(fn);
  if (tokenLost) fn();
  return () => tokenLostWatchers.delete(fn);
}

export function isTokenLost(): boolean {
  return tokenLost;
}

/** Adopt a token pasted by the user, then reload so every query refetches. */
export function adoptToken(token: string): void {
  sessionStorage.setItem(KEY, token.trim());
  window.location.reload();
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: {
      ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
      ...(REVIEWER ? { "X-Openlab-Reviewer": REVIEWER } : {}),
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const isJson = (res.headers.get("content-type") ?? "").includes("application/json");
  const data = isJson ? await res.json() : await res.text();
  if (res.status === 401 && !tokenLost) {
    tokenLost = true;
    tokenLostWatchers.forEach((fn) => fn());
  }
  if (!res.ok) throw new ApiError(res.status, data);
  return data as T;
}

export const api = {
  get: <T>(path: string) => req<T>("GET", path),
  post: <T>(path: string, body?: unknown) => req<T>("POST", path, body ?? {}),
};

/** The socket URL for a channel, with the token as a query parameter. */
export function socketUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${path}?t=${encodeURIComponent(TOKEN)}`;
}
