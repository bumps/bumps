import { expect, test, vi } from "vitest";
import ParameterView from "../../src/components/ParameterView.vue";
import SummaryView from "../../src/components/SummaryView.vue";
import { mountPanel } from "./mountPanel";

test.each([
  ["ParameterView", ParameterView],
  ["SummaryView", SummaryView],
])("%s labels point at exactly one, unique, control", async (_name, panel) => {
  const { wrapper } = mountPanel(panel);
  await vi.waitFor(() => expect(wrapper.findAll("tbody tr")).toHaveLength(2));

  const labels = wrapper.findAll("label[for]");
  expect(labels.length).toBeGreaterThan(0);
  for (const label of labels) {
    const id = label.attributes("for")!;
    expect(document.querySelectorAll(`[id="${CSS.escape(id)}"]`), `label for="${id}"`).toHaveLength(1);
  }
  wrapper.unmount();
});
