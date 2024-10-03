<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, shallowRef, onMounted } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from '../setupDrawLoop';
import * as Plotly from 'plotly.js/lib/core';
import { configWithSVGDownloadButton } from '../plotly_extras.mjs';

const plot_data = shallowRef<Plotly.Data[]>([]);
const plot_layout = shallowRef<Partial<Plotly.Layout>>();
const plot_config = shallowRef<Partial<Plotly.Config>>();
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const current_index = ref<number>(0);
const parameters_local = ref<String[]>([]);
const trace_opacity = ref(0.4);
const props = defineProps<{
  socket: Socket,
}>();

const { draw_requested } = setupDrawLoop('updated_uncertainty', props.socket, fetch_and_draw);

async function get_parameter_names() {
  const new_pars = await props.socket.asyncEmit("get_parameter_labels") as String[];
  if (new_pars == null) {
    return;
  }
  if (current_index.value > parameters_local.value.length) {
    current_index.value = 0
  }
  parameters_local.value = new_pars
}

props.socket.on('model_loaded', get_parameter_names);

onMounted(async () => {
  await get_parameter_names();
});

async function fetch_and_draw() {

  const payload = await props.socket.asyncEmit('get_parameter_trace_plot', current_index.value) as Plotly.PlotlyDataLayoutConfig;

  let plotdata = { ...payload };
  const { data, layout } = plotdata;
  plot_data.value = data
  plot_layout.value = layout
  const config: Partial<Plotly.Config> = {
    responsive: true,
    ...configWithSVGDownloadButton
  }

  plot_config.value = config

  plot_data.value = data

  await draw_plot()

}

async function draw_plot() {
  const opacity = trace_opacity.value

  let data = plot_data.value
  data.forEach(line => line.opacity = opacity)
  
  await Plotly.react(plot_div.value as HTMLDivElement, [...data], plot_layout.value, plot_config.value);

}

</script>
    
<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <select
      v-model="current_index"
      @change="draw_requested = true"
      >
      <option v-for="(parameter, index) in parameters_local" :key="index" :value="index">
        {{ parameter ?? "" }} </option>
    </select>
    <div class="row px-2 align-items-center">
      <div class="col-auto">
        <label for="opacity_control" class="col-form-label">Trace opacity</label>
      </div>
      <div class="col">
        <input type="range" min="0" max="1.0" step="0.01" id="opacity_control" class="form-range" v-model.number="trace_opacity" @input="draw_plot">
      </div>
    </div>
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>