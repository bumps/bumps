<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import * as mpld3 from "mpld3";
import * as Plotly from "plotly.js/lib/core";
import { v4 as uuidv4 } from "uuid";
import type { AsyncSocket } from "../asyncSocket";
import { setupDrawLoop } from "../setupDrawLoop";

// type ModelNameInfo = { name: string; model_index: number };
// const title = "Data";
const plot_div = ref<HTMLDivElement | null>(null);
const plot_div_id = ref(`div-${uuidv4()}`);
const show_multiple = ref(false);
const model_names = ref<string[]>([]);
const current_models = ref<number[]>([0]);
const current_model = ref(0);
const dropdown_open = ref(false);

const props = defineProps<{
  socket: AsyncSocket;
}>();

const current_selection = computed(() => {
  return show_multiple.value ? current_models.value : [current_model.value];
});

const { draw_requested } = setupDrawLoop("updated_parameters", props.socket, fetch_and_draw);

async function get_model_names() {
  const new_names = (await props.socket.asyncEmit("get_model_names")) as string[];
  if (new_names == null) {
    return;
  }
  model_names.value = new_names;
  const new_model = Math.min(current_model.value, model_names.value.length - 1);
  current_models.value = [new_model];
  current_model.value = new_model;
}

props.socket.on("model_loaded", get_model_names);
onMounted(async () => {
  await get_model_names();
});

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit("get_data_plot", current_selection.value);
  if (payload == null || plot_div.value == null) {
    return;
  }
  // get a handle to all child divs in plot_div
  const childDivs = plot_div.value.querySelectorAll("div.plot-grid-item");
  for (const [index, plotitem] of (payload as { fig_type: "plotly" | "mpld3"; plotdata: object }[]).entries()) {
    await draw_plot(childDivs[index] as HTMLDivElement, plotitem);
  }
}

async function draw_plot(childDiv: HTMLDivElement, plotitem: { fig_type: "plotly" | "mpld3"; plotdata: object }) {
  const { fig_type, plotdata } = plotitem;
  if (fig_type === "plotly") {
    const { data, layout } = plotdata as Plotly.PlotlyDataLayoutConfig;
    const config = { responsive: true };
    // draw into the child div with index index
    await Plotly.react(childDiv, [...data], layout, config);
  } else if (fig_type === "mpld3") {
    let mpld3_data = plotdata as { width: number; height: number };
    mpld3_data.width = Math.round(childDiv.clientWidth ?? 640) - 8;
    mpld3_data.height = Math.round(childDiv.clientHeight ?? 480) - 8;
    mpld3.draw_figure(childDiv.id, mpld3_data, false, true);
  }
}

function toggle_multiple() {
  if (show_multiple.value) {
    current_models.value = [current_model.value];
  } else {
    // then we're toggling from multiple to single...
    current_model.value = current_models.value.splice(-1)[0] ?? 0;
  }
  draw_requested.value = true;
}

function changeModel() {
  draw_requested.value = true;
}

function toggleModelSelection(model_index: number) {
  const index = current_models.value.indexOf(model_index);
  if (index > -1) {
    current_models.value.splice(index, 1);
  } else {
    current_models.value.push(model_index);
  }
  draw_requested.value = true;
}

function toggleDropdown() {
  dropdown_open.value = !dropdown_open.value;
}
</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <div class="form-check">
      <label class="form-check-label pe-2" for="multiple">Show multiple</label>
      <input
        id="multiple"
        v-model="show_multiple"
        class="form-check-input"
        type="checkbox"
        @change="toggle_multiple()"
      />
    </div>
    <label for="model-select">Models:</label>
    <div v-if="show_multiple" class="dropdown">
      <button
        class="btn btn-secondary dropdown-toggle"
        type="button"
        :aria-expanded="dropdown_open"
        @click="toggleDropdown"
      >
        {{ current_models.length }} model(s) selected
      </button>
      <ul class="dropdown-menu" :class="{ show: dropdown_open }">
        <li v-for="(model, model_index) in model_names" :key="model_index">
          <label class="dropdown-item">
            <input
              type="checkbox"
              class="form-check-input me-2"
              :value="model_index"
              :checked="current_models.includes(model_index)"
              @change="toggleModelSelection(model_index)"
            />
            Model {{ model_index + 1 }}: {{ model }}
          </label>
        </li>
      </ul>
    </div>
    <select v-else id="model-select" v-model="current_model" @change="changeModel()">
      <option v-for="(model, model_index) in model_names" :key="model_index" :value="model_index">
        Model {{ model_index + 1 }}: {{ model }}
      </option>
    </select>
    <div :id="plot_div_id" ref="plot_div" class="flex-grow-1 plot-container">
      <div
        v-for="model_index in current_selection"
        :id="'child_plot_' + model_index"
        :key="model_index"
        class="plot-grid-item"
      ></div>
    </div>
  </div>
</template>

<style>
.plot-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  grid-auto-rows: 1fr;
  align-content: start;
  overflow-x: hidden;
  overflow-y: auto;
  gap: 0rem;
}

.plot-grid-item {
  width: 100%;
  min-width: 400px;
  height: 100%;
  min-height: 300px;
  margin: 0;
  padding: 0;
  --bs-gutter-x: 0rem;
}

@media (prefers-color-scheme: dark) {
  .mpld3-figure {
    background-color: lightgray;
  }
}
</style>
