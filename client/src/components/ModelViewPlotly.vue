<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import * as Plotly from 'plotly.js/lib/core';
import { setupDrawLoop } from '../setupDrawLoop';

const title = "Profile";
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const model_names = ref<string[]>([]);
const current_model = ref(0);

const props = defineProps<{
  socket: Socket,
}>();

const { draw_requested, drawing_busy, mounted } = setupDrawLoop('update_parameters', props.socket, fetch_and_draw);

props.socket.on('model_loaded', ({ message: { model_names: new_model_names } }) => {
  model_names.value = new_model_names;
});

onMounted(() => {
  props.socket.emit('get_topic_messages', 'model_loaded', (messages) => {
    const new_model_names = messages?.[0]?.message?.model_names;
    if (new_model_names != null) {
      model_names.value = new_model_names;
    }
  });
});

function generate_new_traces(profile_data) {
  let traces: (Plotly.Data & { x: number[], y: number[] })[] = [];
  const { step_profile, smooth_profile } = profile_data;
  console.log(profile_data);

  traces.push({ x: step_profile.z, y: step_profile.rho, mode: 'lines', name: 'rho', legendgroup: 'rho', line: { color: "green", width: 2, dash: "dot" }, showlegend: false });
  traces.push({ x: smooth_profile.z, y: smooth_profile.rho, mode: 'lines', name: 'rho', legendgroup: 'rho', line: { color: "green", width: 2 } });

  traces.push({ x: step_profile.z, y: step_profile.irho, mode: 'lines', name: 'irho', legendgroup: 'irho', line: { color: "blue", width: 2, dash: "dot" }, showlegend: false });
  traces.push({ x: smooth_profile.z, y: smooth_profile.irho, mode: 'lines', name: 'irho', legendgroup: 'irho', line: { color: "blue", width: 2 } });

  if (step_profile.rhoM && smooth_profile.rhoM) {
    traces.push({ x: step_profile.z, y: step_profile.rhoM, mode: 'lines', name: 'rhoM', legendgroup: 'rhoM', line: { color: "red", width: 2, dash: "dot" }, showlegend: false });
    traces.push({ x: smooth_profile.z, y: smooth_profile.rhoM, mode: 'lines', name: 'rhoM', legendgroup: 'rhoM', line: { color: "red", width: 2 } });
  }

  if (step_profile.thetaM && smooth_profile.thetaM) {
    traces.push({ x: step_profile.z, y: step_profile.thetaM, mode: 'lines', name: 'thetaM', legendgroup: 'thetaM', yaxis: "y2", line: { color: "gold", width: 2, dash: "dot" }, showlegend: false });
    traces.push({ x: smooth_profile.z, y: smooth_profile.thetaM, mode: 'lines', name: 'thetaM', legendgroup: 'thetaM', yaxis: "y2", line: { color: "gold", width: 2 } });
  }

  return traces;
}

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_profile_data', current_model.value);
  const traces = generate_new_traces(payload);
  const layout: Partial<Plotly.Layout> = {
    uirevision: 1,
    xaxis: {
      title: {
        text: 'depth (Å)'
      },
      type: 'linear',
      autorange: true,
    },
    yaxis: {
      title: { text: '$\\text{SLD: } \\rho, \\rho_i, \\rho_M / 10^{-6} \\text{Å}^{-2}$' },
      exponentformat: 'e',
      showexponent: 'all',
      type: 'linear',
      autorange: true,
    },
    yaxis2: {
      title: { text: '$\\text{Magnetic Angle } \\theta_M / {}^{\\circ}$' },
      type: 'linear',
      autorange: false,
      range: [0, 360],
      anchor: 'x',
      overlaying: 'y',
      side: 'right'
    },
    margin: {
      l: 75,
      r: 50,
      t: 25,
      b: 75,
      pad: 4
    }
  };

  const config = { responsive: true }
  // this function is only called when mounted, so plot_div.value exists:
  await Plotly.react(plot_div.value as HTMLDivElement, [...traces], layout, config);
}

</script>
    
<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <select v-model="current_model" @change="draw_requested = true">
      <option v-for="(name, index) in model_names" :key="index" :value="index">{{ index }}: {{ name ?? "" }}</option>
    </select>
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>