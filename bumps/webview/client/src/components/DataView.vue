<script setup lang="ts">
import { onMounted, ref } from "vue";
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

const props = defineProps<{
  socket: AsyncSocket;
}>();

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
  const models = show_multiple.value ? current_models.value : [current_model.value];
  const payload = await props.socket.asyncEmit("get_data_plot", models);
  let { fig_type, plotdata } = payload as { fig_type: "plotly" | "mpld3"; plotdata: object };
  if (fig_type === "plotly") {
    const { data, layout } = plotdata as Plotly.PlotlyDataLayoutConfig;
    const config = { responsive: true };
    await Plotly.react(plot_div.value as HTMLDivElement, [...data], layout, config);
  } else if (fig_type === "mpld3") {
    let mpld3_data = plotdata as { width: number; height: number };
    mpld3_data.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
    mpld3_data.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
    mpld3.draw_figure(plot_div_id.value, mpld3_data, false, true);
  }
}

function toggle_multiple() {
  if (show_multiple.value) {
    current_models.value = [current_model.value];
  } else {
    // then we're toggling from multiple to single...
    current_model.value = current_models.value.splice(-1)[0] ?? 0;
    draw_requested.value = true;
  }

  if (plot_div.value) {
    Plotly.Plots.resize(plot_div.value);
  }
}

function changeModel() {
  draw_requested.value = true;
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
    <select v-if="show_multiple" id="multi-model-select" v-model="current_models" multiple @change="changeModel()">
      <option v-for="(model, model_index) in model_names" :key="model_index" :value="model_index">
        Model {{ model_index + 1 }}: {{ model }}
      </option>
    </select>
    <select v-else id="model-select" v-model="current_model" @change="changeModel()">
      <option v-for="(model, model_index) in model_names" :key="model_index" :value="model_index">
        Model {{ model_index + 1 }}: {{ model }}
      </option>
    </select>
    <div :id="plot_div_id" ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style>
@media (prefers-color-scheme: dark) {
  .mpld3-figure {
    background-color: lightgray;
  }
}
</style>
