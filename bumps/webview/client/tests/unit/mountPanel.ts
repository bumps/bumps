import { mount } from "@vue/test-utils";
import { type Component, defineComponent, h, KeepAlive } from "vue";
import { fakeSocket } from "./fakeSocket";

/** A parameter as returned by the server's get_parameters. */
export const parameter = (name: string, id: string) => ({
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

/**
 * Mount a parameter panel with parameters m and b. Panels only fetch and draw
 * while active inside <KeepAlive>, as in PanelTabContainer.
 */
export function mountPanel(panel: Component) {
  const socket = fakeSocket({
    get_shared_setting: "2026-01-01T00:00:00",
    get_parameters: [parameter("m", "p1"), parameter("b", "p2")],
  });
  const Host = defineComponent(() => () => h(KeepAlive, null, [h(panel, { socket })]));
  const wrapper = mount(Host, { attachTo: document.body });
  return { wrapper, socket };
}
