/**
 * A line typed into the agent pane from elsewhere in the app. The
 * pane registers its socket here; a line sent while the pane is closed waits until the agent has started.
 */
let socket: WebSocket | null = null;
let queued = "";
let queuedSubmit = true;

/** Called by the agent pane when its terminal opens (with a socket) and closes (null). */
export function registerAgentSocket(ws: WebSocket | null): void {
  socket = ws;
  if (ws && queued) {
    const line = queued + (queuedSubmit ? "\r" : "");
    queued = "";
    window.setTimeout(() => ws.readyState === WebSocket.OPEN && ws.send(line), 4000); // the CLI's start-up
  }
}

/**
 * Type `line` into the agent, and press Enter unless `submit` is false (a suggested prompt is typed and left
 * for the person to edit or send). `true` if sent now, `false` if queued for the pane's opening.
 */
export function sendToAgent(line: string, submit = true): boolean {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(line + (submit ? "\r" : ""));
    return true;
  }
  queued = line;
  queuedSubmit = submit;
  return false;
}
