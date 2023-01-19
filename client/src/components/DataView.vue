<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import type { Socket } from 'socket.io-client';
import { setupDrawLoop } from '../setupDrawLoop';

const title = "Reflectivity";
const plot_div = ref<HTMLDivElement | null>(null);

const props = defineProps<{
  socket: Socket,
}>();

setupDrawLoop('update_parameters', props.socket, fetch_and_draw);

const REFLECTIVITY_PLOTS = [
  "Fresnel",
  "Log Fresnel",
  "Linear",
  "Log",
  "Q4",
  "SA"
] as const;
type ReflectivityPlotEnum = typeof REFLECTIVITY_PLOTS;
type ReflectivityPlot = ReflectivityPlotEnum[number];
const reflectivity_type = ref<ReflectivityPlot>("Log");

function generate_new_traces(model_data, view: ReflectivityPlot) {
  let theory_traces: (Plotly.Data & { x: number, y: number })[] = [];
  let data_traces: (Plotly.Data & { x: number, y: number })[] = [];
  switch (view) {
    case "Log":
    case "Linear": {
      for (let model of model_data) {
        for (let xs of model) {
          theory_traces.push({ x: xs.Q, y: xs.theory, mode: 'lines', name: xs.label + ' theory', line: { width: 2 } });
          data_traces.push({ x: xs.Q, y: xs.R, error_y: {type: 'data', array: xs.dR, visible: true}, mode: 'markers', name: xs.label + ' data' });
        }
      }
      break;
    }
    case "Log Fresnel":
    case "Fresnel": {
      for (let model of model_data) {
        for (let xs of model) {
          const theory = xs.theory.map((y, i) => (y / (xs.fresnel[i])));
          const R = xs.R.map((y, i) => (y / (xs.fresnel[i])));
          theory_traces.push({ x: xs.Q, y: theory, mode: 'lines', name: xs.label + ' theory', line: { width: 2 } });
          data_traces.push({ x: xs.Q, y: R, mode: 'markers', name: xs.label + ' data' });
        }
      }
      break;
    }
    case "Q4": {
      // Q4 = 1e-8*Q**-4*self.intensity.value + self.background.value
      for (let model of model_data) {
        for (let xs of model) {
          const {intensity, background} = xs;
          const Q4 = xs.Q.map((qq) => (1e-8*Math.pow(qq, -4)*intensity + background));
          const theory = xs.theory.map((t, i) => (t / Q4[i]));
          const R = xs.R.map((r, i) => (r / Q4[i]));
          theory_traces.push({ x: xs.Q, y: theory, mode: 'lines', name: xs.label + ' theory', line: { width: 2 } });
          data_traces.push({ x: xs.Q, y: R, mode: 'markers', name: xs.label + ' data' });
        }
      }
      break;
    }
  }
  return {theory_traces, data_traces};
}

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_plot_data', 'linear');
  // console.log(payload);
  const { theory_traces, data_traces } = generate_new_traces(payload, reflectivity_type.value)
  const layout: Partial<Plotly.Layout> = {
    uirevision: reflectivity_type.value,
    xaxis: {
      title: {
        text: 'Q (Å<sup>-1</sup>)'
      },
      type: 'linear',
      autorange: true,
    },
    yaxis: {
      title: { text: 'Reflectivity' },
      exponentformat: 'e',
      showexponent: 'all',
      type: (/^(Log|Q4)/.test(reflectivity_type.value)) ? 'log' : 'linear',
      autorange: true,
    },
    margin: {
      l: 75,
      r: 50,
      t: 25,
      b: 75,
      pad: 4
    }
  };

  const config = {responsive: true}
  await Plotly.react(plot_div.value as HTMLDivElement, [...theory_traces, ...data_traces], layout, config);
}

</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <select v-model="reflectivity_type" @change="fetch_and_draw">
      <option v-for="refl_type in REFLECTIVITY_PLOTS" :key="refl_type" :value="refl_type">{{refl_type}}</option>
    </select>
    <div class="flex-grow-1" ref="plot_div" id="plot_div">

    </div>
  </div>
</template>
