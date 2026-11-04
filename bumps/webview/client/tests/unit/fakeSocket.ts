import { vi } from "vitest";
import type { AsyncSocket } from "../../src/asyncSocket";

type Handler = (...args: any[]) => void;

/**
 * Stand-in for the server socket that components receive as a prop.
 * `asyncEmit` answers from the `responses` table (calling a trailing callback
 * like the real one does), and `serverEmit` simulates a server push.
 */
export function fakeSocket(responses: Record<string, unknown> = {}) {
  const handlers: Record<string, Handler[]> = {};
  const socket = {
    on: vi.fn((ev: string, fn: Handler) => {
      (handlers[ev] ??= []).push(fn);
    }),
    off: vi.fn((ev: string, fn: Handler) => {
      handlers[ev] = handlers[ev]?.filter((h) => h !== fn) ?? [];
    }),
    asyncEmit: vi.fn(async (ev: string, ...args: any[]) => {
      const callback = typeof args.at(-1) === "function" ? args.pop() : null;
      const result = responses[ev];
      if (callback) await callback(result);
      return result;
    }),
    serverEmit(ev: string, ...args: any[]) {
      handlers[ev]?.forEach((fn) => {
        fn(...args);
      });
    },
  };
  return socket as unknown as AsyncSocket & { serverEmit: typeof socket.serverEmit };
}
