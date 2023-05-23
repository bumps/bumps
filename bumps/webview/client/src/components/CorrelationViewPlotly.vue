<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref } from 'vue';
import type { AsyncSocket } from '../asyncSocket';
import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from '../setupDrawLoop';
import { cache } from '../plotcache';
import * as Plotly from 'plotly.js/lib/core';
import Heatmap from 'plotly.js/lib/heatmap';

// workaround to PlotlyModule not being exported as type!
type RegisterTypes = Parameters<typeof Plotly.register>[0];
type PlotlyModule = Exclude<RegisterTypes, any[]>;

Plotly.register([
  Heatmap as PlotlyModule
])

const title = "Correlations";
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);

const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('uncertainty_update', props.socket, fetch_and_draw, title);


async function fetch_and_draw(latest_timestamp: string) {
  let { timestamp, plotdata } = cache[title] as { timestamp: string, plotdata: Plotly.PlotlyDataLayoutConfig} ?? {};
  if (timestamp !== latest_timestamp) {
    const payload = await props.socket.asyncEmit('get_correlation_plot') as Plotly.PlotlyDataLayoutConfig;
    plotdata = { ...payload };
    cache[title] = {timestamp: latest_timestamp, plotdata};
  }
  const { data, layout } = plotdata;
  const config = { responsive: true, scrollZoom: true }
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