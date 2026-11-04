import { flushPromises, mount } from "@vue/test-utils";
import { expect, test } from "vitest";
import LogView from "../../src/components/LogView.vue";
import { fakeSocket } from "./fakeSocket";

const entry = (title: string, message: string, timestamp = "0") => ({ message: { title, message }, timestamp });

test("Shows existing log messages on mount", async () => {
  const socket = fakeSocket({ get_topic_messages: [entry("Init", "started")] });
  const wrapper = mount(LogView, { props: { socket } });
  await flushPromises();

  expect(socket.asyncEmit).toHaveBeenCalledWith("get_topic_messages", "log", expect.any(Function));
  expect(wrapper.findAll("tbody tr")).toHaveLength(1);
  expect(wrapper.text()).toContain("Init");
});

test("Appends messages pushed by the server", async () => {
  const socket = fakeSocket({ get_topic_messages: [] });
  const wrapper = mount(LogView, { props: { socket } });
  await flushPromises();

  socket.serverEmit("log", entry("Fit", "step 1"));
  socket.serverEmit("log", entry("Fit", "step 2"));
  await flushPromises();

  expect(wrapper.findAll("tbody tr")).toHaveLength(2);
});

test("Clear empties the table and Refresh reloads it", async () => {
  const socket = fakeSocket({ get_topic_messages: [entry("A", "a"), entry("B", "b")] });
  const wrapper = mount(LogView, { props: { socket } });
  await flushPromises();

  await wrapper.get("button.btn-danger").trigger("click");
  expect(wrapper.findAll("tbody tr")).toHaveLength(0);

  await wrapper.get("button.btn-primary").trigger("click");
  await flushPromises();
  expect(wrapper.findAll("tbody tr")).toHaveLength(2);
});
