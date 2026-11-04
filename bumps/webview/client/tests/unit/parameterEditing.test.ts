import type { VueWrapper } from "@vue/test-utils";
import { afterEach, expect, test, vi } from "vitest";
import ParameterView from "../../src/components/ParameterView.vue";
import SummaryView from "../../src/components/SummaryView.vue";
import { mountPanel } from "./mountPanel";

const PANELS = [
  ["ParameterView", ParameterView],
  ["SummaryView", SummaryView],
] as const;

// Both panels put the value in the third column; SummaryView adds min and max after it.
const COLUMN = { value: 2, min: 3, max: 4 };

let wrapper: VueWrapper | undefined;
afterEach(() => wrapper?.unmount());

async function setup(panel: (typeof PANELS)[number][1]) {
  const mounted = mountPanel(panel);
  wrapper = mounted.wrapper;
  await vi.waitFor(() => expect(wrapper!.findAll("tbody tr")).toHaveLength(2));
  return mounted;
}

/** Type `text` into a parameter's editable cell and leave it, as the user would. */
async function edit(row: number, column: keyof typeof COLUMN, text: string) {
  const cell = wrapper!.findAll("tbody tr")[row]!.findAll("td")[COLUMN[column]]!;
  cell.element.textContent = text;
  await cell.trigger("blur");
  return cell;
}

const setParameterCalls = (socket: ReturnType<typeof mountPanel>["socket"]) =>
  vi.mocked(socket.asyncEmit).mock.calls.filter(([ev]) => ev === "set_parameter");

test.each(PANELS)("%s sends an edited value to the server", async (_name, panel) => {
  const { socket } = await setup(panel);
  await edit(1, "value", "1.5");
  expect(setParameterCalls(socket)).toEqual([["set_parameter", "p2", "value", "1.5"]]);
});

test.each(PANELS.flatMap(([name, panel]) => ["", "  ", "abc", "NaN"].map((text) => ({ name, panel, text }))))(
  '$name rejects value "$text" and restores the cell',
  async ({ panel, text }) => {
    const { socket } = await setup(panel);
    const cell = await edit(0, "value", text);
    expect(setParameterCalls(socket)).toEqual([]);
    expect(cell.text()).toBe("1");
  }
);

test.each(["inf", "-inf", "1e3"])("SummaryView accepts limit %j", async (text) => {
  const { socket } = await setup(SummaryView);
  await edit(0, "max", text);
  expect(setParameterCalls(socket)).toEqual([["set_parameter", "p1", "max", text]]);
});

test("ParameterView checkbox toggles whether a parameter is fitted", async () => {
  const { socket } = await setup(ParameterView);
  await wrapper!.get("#param-checkbox-0").trigger("click");
  expect(setParameterCalls(socket)).toEqual([["set_parameter", "p1", "fixed", true]]);
});
