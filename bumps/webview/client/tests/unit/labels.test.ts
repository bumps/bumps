import { mount } from "@vue/test-utils";
import { expect, test, vi } from "vitest";
import { type Component, KeepAlive, defineComponent, h } from "vue";
import ParameterView from "../../src/components/ParameterView.vue";
import SummaryView from "../../src/components/SummaryView.vue";
import { fakeSocket } from "./fakeSocket";

const parameter = (name: string, id: string) => ({
  name,
  id,
  tags: [],
  fittable: true,
  fixed: false,
  paths: [],
  link: "",
  writable: true,
  value01: 0.5,
  value_str: "1",
  min_str: "0",
  max_str: "2",
  slot_repr: "",
});

/** Panels only fetch and draw while active inside <KeepAlive>, as in PanelTabContainer. */
function mountPanel(panel: Component) {
  const socket = fakeSocket({
    get_shared_setting: "2026-01-01T00:00:00",
    get_parameters: [parameter("m", "p1"), parameter("b", "p2")],
  });
  const Host = defineComponent(() => () => h(KeepAlive, null, [h(panel, { socket })]));
  return mount(Host, { attachTo: document.body });
}

test.each([
  ["ParameterView", ParameterView],
  ["SummaryView", SummaryView],
])("%s labels point at exactly one, unique, control", async (_name, panel) => {
  const wrapper = mountPanel(panel);
  await vi.waitFor(() => expect(wrapper.findAll("tbody tr")).toHaveLength(2));

  const labels = wrapper.findAll("label[for]");
  expect(labels.length).toBeGreaterThan(0);
  for (const label of labels) {
    const id = label.attributes("for")!;
    expect(document.querySelectorAll(`[id="${CSS.escape(id)}"]`), `label for="${id}"`).toHaveLength(1);
  }
  wrapper.unmount();
});
