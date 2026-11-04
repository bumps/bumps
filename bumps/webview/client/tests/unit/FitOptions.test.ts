import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { autoupdate_state, default_fitter_settings } from "../../src/app_state";
import FitOptions from "../../src/components/FitOptions.vue";
import { fakeSocket } from "./fakeSocket";

// A subset of the server's get_fit_fields, one field of each stype.
const FIT_FIELDS = {
  steps: { name: "steps", label: "Number of steps", stype: "integer" },
  ftol: { name: "ftol", label: "f(x) tolerance", stype: "float" },
  alpha: { name: "alpha", label: "Convergence criteria", stype: { min: 0, max: 0.1 } },
  init: { name: "init", label: "Population initializer", stype: ["eps", "lhs", "cov", "random"] },
  trim: { name: "trim", label: "Burn-in trim", stype: "boolean" },
};

const DEFAULTS = {
  amoeba: { name: "Nelder-Mead Simplex", settings: { steps: 1000, ftol: 1e-8, radius: 0.15 } },
  dream: { name: "DREAM", settings: { steps: 0, alpha: 0.01, init: "eps", trim: false } },
};

let wrapper: VueWrapper<any>;
let socket: ReturnType<typeof fakeSocket>;

beforeEach(() => {
  // jsdom doesn't implement modal dialogs
  HTMLDialogElement.prototype.showModal ??= vi.fn();
  HTMLDialogElement.prototype.close ??= vi.fn();
  default_fitter_settings.value = DEFAULTS;
  autoupdate_state.shared_state.fitter_settings.value = undefined;
  autoupdate_state.shared_state.selected_fitter.value = undefined;
});
afterEach(() => wrapper.unmount());

/** Mount the dialog and open it on `fitter`, as the Fit Options menu does. */
async function open(fitter = "amoeba") {
  autoupdate_state.shared_state.selected_fitter.value = fitter;
  socket = fakeSocket({ get_fit_fields: FIT_FIELDS });
  wrapper = mount(FitOptions, { props: { socket }, attachTo: document.body });
  await flushPromises();
  wrapper.vm.open();
  await flushPromises();
}

const field = (name: string) => wrapper.get(`#fitter_setting_${name}`);
const button = (text: string) => wrapper.findAll("button").find((b) => b.text() === text)!;
const emitted = (ev: string) => vi.mocked(socket.asyncEmit).mock.calls.filter(([e]) => e === ev);

test("Shows only the fitter's settings that the server defines", async () => {
  await open("amoeba");
  expect(wrapper.findAll("[id^=fitter_setting_]").map((el) => el.attributes("id"))).toEqual([
    "fitter_setting_steps",
    "fitter_setting_ftol",
  ]);
});

test("Save Changes converts each field to its type", async () => {
  await open("dream");
  await field("steps").setValue("250");
  await field("alpha").setValue("0.05");
  await field("init").setValue("lhs");
  await field("trim").setValue(true);
  await button("Save Changes").trigger("click");
  await flushPromises();

  const settings = { steps: 250, alpha: 0.05, init: "lhs", trim: true };
  expect(emitted("set_shared_setting")).toEqual([
    ["set_shared_setting", "fitter_settings", { ...DEFAULTS, dream: { name: "dream", settings } }],
    ["set_shared_setting", "selected_fitter", "dream"],
  ]);
  expect(emitted("start_fit_thread")).toEqual([]);
});

test("Save and Start starts the fit with the new settings", async () => {
  await open("amoeba");
  await field("ftol").setValue("1e-6");
  await button("Save and Start").trigger("click");
  await flushPromises();
  expect(emitted("start_fit_thread")).toEqual([["start_fit_thread", "amoeba", { steps: 1000, ftol: 1e-6 }, false]]);
});

test.each([
  { fitter: "amoeba", name: "steps", value: "1.5" },
  { fitter: "amoeba", name: "steps", value: "abc" },
  { fitter: "amoeba", name: "steps", value: "" },
  { fitter: "amoeba", name: "ftol", value: "abc" },
  { fitter: "amoeba", name: "ftol", value: "" },
  { fitter: "amoeba", name: "ftol", value: " " },
  { fitter: "dream", name: "alpha", value: "0.2" },
  { fitter: "dream", name: "alpha", value: "-1" },
  { fitter: "dream", name: "alpha", value: "" },
])('Setting $name = "$value" is invalid and blocks saving', async ({ fitter, name, value }) => {
  await open(fitter);
  await field(name).setValue(value);
  expect(field(name).classes()).toContain("is-invalid");

  for (const text of ["Save Changes", "Save and Start"]) {
    await button(text).trigger("click");
  }
  await field(name).trigger("keydown", { key: "Enter" });
  await flushPromises();
  expect(emitted("set_shared_setting")).toEqual([]);
  expect(emitted("start_fit_thread")).toEqual([]);
});

test("Reset Defaults restores the fitter's default settings", async () => {
  await open("amoeba");
  await field("steps").setValue("5");
  await button("Reset Defaults").trigger("click");
  expect((field("steps").element as HTMLInputElement).value).toBe("1000");
});
