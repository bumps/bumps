import { encode } from "@msgpack/msgpack";
import { Socket } from "socket.io-client";
import { expect, test, vi } from "vitest";
import "../../src/asyncSocket";

/** Call the real asyncEmit on an object whose `emit` replies with `reply`. */
function emitWithReply(reply: unknown, ...args: any[]) {
  const emit = vi.fn((_ev: string, ...emitArgs: any[]) => emitArgs.at(-1)(reply));
  const result = Socket.prototype.asyncEmit.call({ emit }, "some_event", ...args);
  return { emit, result };
}

test("Passes plain replies through", async () => {
  const { emit, result } = emitWithReply({ a: 1 }, "arg");
  await expect(result).resolves.toEqual({ a: 1 });
  expect(emit).toHaveBeenCalledWith("some_event", "arg", expect.any(Function));
});

test("Decodes msgpack replies (Uint8Array and ArrayBuffer)", async () => {
  const packed = encode({ x: [1, 2, 3] });
  await expect(emitWithReply(packed).result).resolves.toEqual({ x: [1, 2, 3] });
  const buffer = packed.buffer.slice(packed.byteOffset, packed.byteOffset + packed.byteLength);
  await expect(emitWithReply(buffer).result).resolves.toEqual({ x: [1, 2, 3] });
});

test("Awaits a trailing callback with the decoded result", async () => {
  const callback = vi.fn();
  const { emit, result } = emitWithReply(encode("hi"), "arg", callback);
  await expect(result).resolves.toBe("hi");
  expect(callback).toHaveBeenCalledWith("hi");
  // the callback is not forwarded to the server
  expect(emit).toHaveBeenCalledWith("some_event", "arg", expect.any(Function));
  expect(emit.mock.calls[0]).toHaveLength(3);
});
