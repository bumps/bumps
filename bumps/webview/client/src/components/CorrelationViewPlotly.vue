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
const do_sort = ref(true);
const max_rows = ref(25);
const n_bins = ref(50);

const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('uncertainty_update', props.socket, fetch_and_draw, title);


async function fetch_and_draw(latest_timestamp: string) {
  let { timestamp, plotdata } = cache[title] as { timestamp: string, plotdata: Plotly.PlotlyDataLayoutConfig} ?? {};
  if (timestamp !== latest_timestamp) {
    const payload = await props.socket.asyncEmit('get_correlation_plot', do_sort.value, max_rows.value, n_bins.value) as Plotly.PlotlyDataLayoutConfig;
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
    <div class="row g-3">
      <div class="col-md-4 align-middle text-end">
        <label class="form-check-label pe-2" for="sort-by-correlation">Sort by correlation </label>
        <input class="form-check-input" type="checkbox" v-model="do_sort" id="sort-by-correlation" @change="fetch_and_draw" />
      </div>
      <div class="col-md-4 align-middle">
        <label class="form-label" for="max-rows">Max. rows</label>
        <input class="form-control" type="number" v-model="max_rows" id="max-rows" @change="fetch_and_draw" />
      </div>
      <div class="col-md-4 align-middle">
        <label class="form-label" for="n-bins">Num. bins</label>
        <input class="form-control" type="number" v-model="n_bins" id="n-bins" @change="fetch_and_draw" />
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