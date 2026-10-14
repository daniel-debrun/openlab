/**
 * ONE `/ws/events` connection, fanned out to every subscriber.
 *
 * A naive implementation opens a socket per consumer, so a screen with three
 * live panels holds three connections each receiving every frame and filtering
 * client-side. The hub is refcounted and module-level instead: the first
 * subscriber opens the socket, the last one closes it, and reconnection keeps
 * its exponential backoff.
 *
 * What arrives here is the whole point of the app being live: a `workspace`
 * frame means a tool changed something, and it may well have been called by the
 * agent in the terminal below rather than by a click. The UI treats both the
 * same, because they went through the same dispatch.
 */
import { useEffect, useRef } from "react";
import { socketUrl } from "./client";

export type Event = Record<string, unknown>;
export type EventHandler = (ev: Event) => void;
export type EventFilter = (ev: Event) => boolean;

type Sub = { filter: EventFilter | null; handler: EventHandler };

const subs = new Set<Sub>();
let socket: WebSocket | null = null;
let retry = 500;
let wantOpen = false;

/** Connection state, for the status dot. */
export type LinkState = "connecting" | "open" | "closed";
let linkState: LinkState = "closed";
const linkWatchers = new Set<(s: LinkState) => void>();

function setLink(s: LinkState) {
  if (linkState === s) return;
  linkState = s;
  linkWatchers.forEach((w) => w(s));
}

export function getLinkState(): LinkState {
  return linkState;
}

export function watchLink(fn: (s: LinkState) => void): () => void {
  linkWatchers.add(fn);
  fn(linkState);
  return () => linkWatchers.delete(fn);
}

function dispatch(ev: Event) {
  // Copy: a handler may unsubscribe during iteration.
  for (const sub of Array.from(subs)) {
    try {
      if (!sub.filter || sub.filter(ev)) sub.handler(ev);
    } catch (e) {
      // one bad handler must not stop the others from seeing the frame
      console.error("[openlab] event handler threw", e);
    }
  }
}

function connect() {
  if (!wantOpen || socket) return;
  setLink("connecting");
  const ws = new WebSocket(socketUrl("/ws/events"));
  socket = ws;
  ws.onopen = () => {
    retry = 500;
    setLink("open");
  };
  ws.onmessage = (m) => {
    try {
      dispatch(JSON.parse(m.data));
    } catch {
      /* ignore malformed frames */
    }
  };
  ws.onclose = () => {
    socket = null;
    setLink("closed");
    if (wantOpen) {
      retry = Math.min(retry * 2, 8000);
      setTimeout(connect, retry);
    }
  };
  ws.onerror = () => {
    /* onclose follows; backoff is handled there */
  };
}

/** Subscribe to the shared socket. Returns an unsubscribe function. */
export function subscribe(handler: EventHandler, filter?: EventFilter): () => void {
  const sub: Sub = { filter: filter ?? null, handler };
  subs.add(sub);
  if (!wantOpen) {
    wantOpen = true;
    connect();
  }
  return () => {
    subs.delete(sub);
    if (!subs.size) {
      wantOpen = false;
      socket?.close();
      socket = null;
    }
  };
}

/**
 * A filtered subscription for a component.
 *
 * The handler is kept in a ref so a re-render does not tear down and rebuild the
 * subscription — which, with a shared refcounted socket, would otherwise close
 * and reopen the connection on every render of the last subscriber.
 */
export function useEvents(handler: EventHandler, filter?: EventFilter, deps: unknown[] = []) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;
  const filterRef = useRef(filter);
  filterRef.current = filter;

  useEffect(
    () =>
      subscribe(
        (ev) => handlerRef.current(ev),
        (ev) => (filterRef.current ? filterRef.current(ev) : true),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps,
  );
}

/* ------------------------------------------------------------------ frames */

export interface HelloEvent extends Event {
  channel: "hello";
  version?: string;
}

/** Something changed: refetch the file list and the open document. */
export interface WorkspaceEvent extends Event {
  channel: "workspace";
  tool: string;
  args?: Record<string, unknown>;
}

export const isHello = (ev: Event): ev is HelloEvent => ev.channel === "hello";
export const isWorkspace = (ev: Event): ev is WorkspaceEvent => ev.channel === "workspace";
