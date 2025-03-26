<script setup lang="ts">
import { onMounted, ref, shallowRef } from "vue";
import * as Plotly from "plotly.js/lib/core";
import { Socket } from "socket.io-client";
import { v4 as uuidv4 } from "uuid";
import { configWithSVGDownloadButton } from "../plotly_extras";
import { setupDrawLoop } from "../setupDrawLoop";

const plot_data = shallowRef<Plotly.Data[]>([]);
const plot_layout = shallowRef<Partial<Plotly.Layout>>();
const plot_config = shallowRef<Partial<Plotly.Config>>();
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const current_index = ref<number>(0);
const parameters_local = ref<string[]>([]);
const trace_opacity = ref(0.4);
const props = defineProps<{
  socket: Socket;
}>();

const { draw_requested } = setupDrawLoop("updated_uncertainty", props.socket, fetch_and_draw);

async function get_parameter_names() {
  const new_pars = (await props.socket.asyncEmit("get_parameter_labels")) as string[];
  if (new_pars == null) {
    return;
  }
  if (current_index.value > parameters_local.value.length) {
    current_index.value = 0;
  }
  parameters_local.value = new_pars;
}

props.socket.on("model_loaded", get_parameter_names);

onMounted(async () => {
  await get_parameter_names();
});

async function fetch_and_draw() {
  const payload = (await props.socket.asyncEmit(
    "get_parameter_trace_plot",
    current_index.value
  )) as Plotly.PlotlyDataLayoutConfig;

  let plotdata = { ...payload };
  const { data, layout } = plotdata;
  if (data == null || layout == null) {
    await Plotly.purge(plot_div_id.value);
    return;
  }
  plot_data.value = data;
  plot_layout.value = layout;
  const config: Partial<Plotly.Config> = {
    responsive: true,
    ...configWithSVGDownloadButton,
  };

  plot_config.value = config;

  plot_data.value = data;

  await draw_plot();
}

async function draw_plot() {
  const opacity = trace_opacity.value;

  let data = plot_data.value;
  data.forEach((line) => ((line as Plotly.ScatterData).opacity = opacity));

  await Plotly.react(plot_div.value as HTMLDivElement, [...data], plot_layout.value, plot_config.value);
}
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <label for="parameter_select" class="col-form-label">Trace Parameter: </label>
    <select id="parameter_select" v-model="current_index" @change="draw_requested = true">
      <option v-for="(parameter, index) in parameters_local" :key="index" :value="index">
        {{ parameter ?? "" }}
      </option>
    </select>
    <div class="row px-2 align-items-center">
      <div class="col-auto">
        <label for="opacity_control" class="col-form-label">Trace opacity:</label>
      </div>
      <div class="col">
        <input
          id="opacity_control"
          v-model.number="trace_opacity"
          type="range"
          min="0"
          max="1.0"
          step="0.01"
          class="form-range"
          @input="draw_plot"
        />
      </div>
    </div>
    <div :id="plot_div_id" ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>
