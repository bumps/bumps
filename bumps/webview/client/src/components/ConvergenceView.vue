<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref } from 'vue';
import type { AsyncSocket } from '../asyncSocket';
import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from '../setupDrawLoop';
import * as Plotly from 'plotly.js/lib/core';
import { SVGDownloadButton } from '../plotly_extras.mjs';

const title = "Convergence";
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('convergence_update', props.socket, fetch_and_draw, title);

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_convergence_plot') as Plotly.PlotlyDataLayoutConfig;
  let plotdata = { ...payload };
  // console.log({plotdata});
  const { data, layout } = plotdata;
  const config = { responsive: true, scrollZoom: true, modeBarButtonsToAdd: [ SVGDownloadButton ] };
  await Plotly.react(plot_div_id.value, [...data], layout, config);
}

</script>
    
<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>