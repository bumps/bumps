import type { Plugin } from "vue";
import tippy from "vue-tippy";
import { options } from "./tooltip";
import "tippy.js/dist/tippy.css";
import "tippy.js/dist/border.css";

/** list of global plugins and their options */
const plugins: [Plugin, unknown][] = [[tippy, options]];

export default plugins;
