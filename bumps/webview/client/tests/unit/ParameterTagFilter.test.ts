import { mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, expect, test } from "vitest";
import ParameterTagFilter from "../../src/components/ParameterTagFilter.vue";

const PARAMETERS = [
  { name: "m", tags: ["a"] },
  { name: "b", tags: ["b"] },
  { name: "c", tags: ["a", "b"] },
  { name: "d", tags: [] },
];

let wrapper: VueWrapper<any>;
afterEach(() => wrapper.unmount());

function mountFilter(parameters = PARAMETERS) {
  wrapper = mount(ParameterTagFilter, { props: { parameters } });
}

/** The tag's button in the Include or Exclude row. */
function tagButton(row: "Include" | "Exclude", tag: string) {
  const rowEl = wrapper.findAll(".row").find((r) => r.text().startsWith(row))!;
  return rowEl.findAll("button").find((b) => b.text() === tag)!;
}

async function click(row: "Include" | "Exclude", ...tags: string[]) {
  for (const tag of tags) await tagButton(row, tag).trigger("click");
}

const shown = () => wrapper.vm.filtered_parameters.map(({ parameter }: any) => parameter.name);

test("Shows every parameter when no tags are selected", () => {
  mountFilter();
  expect(shown()).toEqual(["m", "b", "c", "d"]);
});

test("Include shows parameters with any of the selected tags", async () => {
  mountFilter();
  await click("Include", "a");
  expect(shown()).toEqual(["m", "c"]);
  await click("Include", "b");
  expect(shown()).toEqual(["m", "b", "c"]);
});

test("Exclude hides parameters with any of the selected tags", async () => {
  mountFilter();
  await click("Exclude", "b");
  expect(shown()).toEqual(["m", "d"]);
});

test("Exclude takes precedence over Include", async () => {
  mountFilter();
  await click("Include", "a");
  await click("Exclude", "b");
  expect(shown()).toEqual(["m"]);
});

test("Clicking a selected tag again clears it", async () => {
  mountFilter();
  await click("Include", "a", "a");
  expect(shown()).toEqual(["m", "b", "c", "d"]);
});

test("Filtered parameters keep their index in the full list", async () => {
  // the panels use the index to edit the right parameter
  mountFilter();
  await click("Include", "b");
  expect(wrapper.vm.filtered_parameters.map(({ index }: any) => index)).toEqual([1, 2]);
});

test("A selected tag is highlighted only in its own row", async () => {
  mountFilter();
  const background = (row: "Include" | "Exclude") => (tagButton(row, "a").element as HTMLElement).style.backgroundColor;
  const unselected = background("Include");
  expect(background("Exclude")).toBe(unselected);

  await click("Include", "a");
  expect(background("Include")).not.toBe(unselected);
  expect(background("Exclude")).toBe(unselected);
});

test("A selected tag that no parameter has any more stops filtering", async () => {
  // e.g. after loading a different model; there is no button left to clear it with
  mountFilter();
  await click("Include", "a");
  await wrapper.setProps({ parameters: [{ name: "x", tags: ["b"] }, { name: "y", tags: [] }] });
  expect(shown()).toEqual(["x", "y"]);
});
