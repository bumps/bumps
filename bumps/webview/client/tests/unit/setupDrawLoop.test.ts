import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { defineComponent, h, KeepAlive, nextTick, ref } from "vue";
import { notifications } from "../../src/app_state";
import { setupDrawLoop } from "../../src/setupDrawLoop";
import { fakeSocket } from "./fakeSocket";

const TOPIC = "updated_test";

// Animation frames only run when the test calls frame(), so each step of the loop is explicit.
let frames: FrameRequestCallback[] = [];
async function frame() {
  for (const cb of frames.splice(0)) cb(0);
  await flushPromises();
}

beforeEach(() => {
  frames = [];
  notifications.value = [];
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
  vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

/** Mount a panel using the draw loop inside <KeepAlive>; `active.value = false` switches away from it. */
async function mountPanel(draw: (value: any) => Promise<void>) {
  const socket = fakeSocket({ get_shared_setting: "initial" });
  const Panel = defineComponent(() => {
    setupDrawLoop(TOPIC, socket, draw, "Test");
    return () => h("div");
  });
  const Other = defineComponent(() => () => h("div"));
  const active = ref(true);
  const Host = defineComponent(() => () => h(KeepAlive, null, [h(active.value ? Panel : Other)]));
  const wrapper = mount(Host);
  await flushPromises();
  return { wrapper, socket, active };
}

test("Draws the current value when the panel is activated", async () => {
  const draw = vi.fn(async () => {});
  await mountPanel(draw);
  await frame();
  expect(draw).toHaveBeenCalledExactlyOnceWith("initial");
});

test("A draw error raises a notification and the loop keeps running", async () => {
  vi.useFakeTimers({ toFake: ["setTimeout"] });
  const draw = vi.fn().mockRejectedValueOnce(new Error("boom")).mockResolvedValue(undefined);
  const { socket } = await mountPanel(draw);

  await frame();
  expect(notifications.value).toMatchObject([{ title: "Draw Error", content: "Error drawing Test: Error: boom" }]);

  // the loop pauses for a second after an error, then carries on
  await vi.advanceTimersByTimeAsync(1000);
  socket.serverEmit(TOPIC, "next");
  await frame();
  expect(draw).toHaveBeenLastCalledWith("next");
});

test("A server push to the topic triggers a redraw with the latest value", async () => {
  const draw = vi.fn(async () => {});
  const { socket } = await mountPanel(draw);
  await frame();

  // nothing new, nothing drawn
  await frame();
  expect(draw).toHaveBeenCalledTimes(1);

  // several pushes between frames are drawn once, with the last value
  socket.serverEmit(TOPIC, "a");
  socket.serverEmit(TOPIC, "b");
  await frame();
  expect(draw).toHaveBeenCalledTimes(2);
  expect(draw).toHaveBeenLastCalledWith("b");
});

test("Deactivating the panel unsubscribes from the topic and stops the loop", async () => {
  const draw = vi.fn(async () => {});
  const { socket, active } = await mountPanel(draw);
  await frame();

  active.value = false;
  await nextTick();
  const handler = vi.mocked(socket.on).mock.calls[0]![1];
  expect(socket.off).toHaveBeenCalledWith(TOPIC, handler);

  socket.serverEmit(TOPIC, "while hidden");
  await frame();
  expect(draw).toHaveBeenCalledTimes(1);
  expect(frames).toHaveLength(0);

  // switching back fetches the current value and draws it
  active.value = true;
  await flushPromises();
  await frame();
  expect(draw).toHaveBeenCalledTimes(2);
});
