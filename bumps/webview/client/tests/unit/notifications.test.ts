import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { addNotification, cancelNotification, notifications } from "../../src/app_state";

beforeEach(() => {
  notifications.value = [];
});
afterEach(() => {
  vi.useRealTimers();
});

test("Notification without timeout shows a spinner and stays until cancelled", () => {
  const id = addNotification({ title: "Fitting", content: "running" });
  expect(notifications.value).toEqual([{ title: "Fitting", content: "running", id, spinner: true }]);
  cancelNotification(id);
  expect(notifications.value).toHaveLength(0);
});

test("Notification with timeout removes itself", () => {
  vi.useFakeTimers();
  addNotification({ title: "Saved", content: "ok", timeout: 1000, id: "saved" });
  expect(notifications.value[0]).toMatchObject({ id: "saved", spinner: false });
  vi.advanceTimersByTime(999);
  expect(notifications.value).toHaveLength(1);
  vi.advanceTimersByTime(1);
  expect(notifications.value).toHaveLength(0);
});

test("Cancelling an unknown id is a no-op", () => {
  addNotification({ title: "a", content: "b", id: "keep" });
  cancelNotification("missing");
  expect(notifications.value.map((n) => n.id)).toEqual(["keep"]);
});
