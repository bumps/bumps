<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref, onMounted, watch, onUpdated, computed, shallowRef, ssrContextKey } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import * as mpld3 from 'mpld3';
import { v4 as uuidv4 } from 'uuid';
import type { AsyncSocket } from '../asyncSocket.ts';
import { setupDrawLoop } from '../setupDrawLoop';

type ModelNameInfo = {name: string, model_index: number};
const title = "Data";
const plot_div = ref<HTMLDivElement | null>(null);
const plot_div_id = ref(`div-${uuidv4()}`);
const show_multiple = ref(false);
const model_names = ref<string[]>([]);
const current_models = ref<number[][]>([[0]]);

// add types to mpld3
declare global {
  interface mpld3 {
    draw_figure: (div_id: string, data: { width: number, height: number }, process: boolean | Function, clearElem: boolean) => void;
  }
  var mpld3: mpld3;
}

const props = defineProps<{
  socket: AsyncSocket,
}>();

const { draw_requested } = setupDrawLoop('updated_parameters', props.socket, fetch_and_draw);

async function get_model_names() {
  const new_names = await props.socket.asyncEmit("get_model_names") as string[];
  if (new_names == null) {
    return;
  }
  model_names.value = new_names;
  current_models.value = [[0]]; //Array.from({length: num_models}).map((_, i) => [i]);
}

props.socket.on('model_loaded', get_model_names);
onMounted(async () => {
  await get_model_names();
});

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_data_plot', current_models.value.flat());
  let { fig_type, plotdata } = payload as { fig_type: 'plotly' | 'mpld3', plotdata: object};
  if (fig_type === 'plotly') {
    const { data, layout } = plotdata as Plotly.PlotlyDataLayoutConfig;
    const config = {responsive: true}
    await Plotly.react(plot_div.value as HTMLDivElement, [...data], layout, config);
  }
  else if (fig_type === 'mpld3') {
    let mpld3_data = plotdata as { width: number, height: number };
    mpld3_data.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
    mpld3_data.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
    mpld3.draw_figure(plot_div_id.value, mpld3_data, false, true);
  }
}

function toggle_multiple(value) {
  if (!show_multiple.value) {
    // then we're toggling from multiple to single...
    current_models.value.splice(0, current_models.value.length -1);
    draw_requested.value = true;
  }
  Plotly.Plots.resize(plot_div.value);
}

</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <div class="form-check">
      <label class="form-check-label pe-2" for="multiple">Show multiple</label>
      <input class="form-check-input" type="checkbox" v-model="show_multiple" id="multiple" @change="toggle_multiple"/>
    </div>
    <select v-if="show_multiple"
      v-model="current_models"
      @change="draw_requested = true"
      multiple
      >
      <option v-for="(name, model_index) in model_names" :key="model_index" :value="[model_index]">
        {{ model_index }}:  {{ name ?? "" }} </option>
    </select>
    <select v-else
      v-model="current_models[0]"
      @change="draw_requested = true"
      >
      <option v-for="(name, model_index) in model_names" :key="model_index" :value="[model_index]">
        {{ model_index }}:  {{ name ?? "" }} </option>
    </select>
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>
