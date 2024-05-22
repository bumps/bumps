<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref } from 'vue';
import type { AsyncSocket } from '../asyncSocket.ts';
import { setupDrawLoop } from '../setupDrawLoop';
import * as Plotly from 'plotly.js/lib/core';
import Heatmap from 'plotly.js/lib/heatmap';
import { SVGDownloadButton } from '../plotly_extras.mjs';

// workaround to PlotlyModule not being exported as type!
type RegisterTypes = Parameters<typeof Plotly.register>[0];
type PlotlyModule = Exclude<RegisterTypes, any[]>;

Plotly.register([
  Heatmap as PlotlyModule
])

const title = "Correlations";
const plot_div = ref<Plotly.PlotlyHTMLElement | HTMLDivElement>();
const do_sort = ref(true);
const max_rows = ref(25);
const n_bins = ref(50);
const callback_registered = ref(false);
const stored_timestamp = ref("");
const vars = ref<number[]>([]);
// don't use the one from setupDrawLoop because we are calling
// fetch_and_draw locally:
const drawing_busy = ref(false);

const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('updated_uncertainty', props.socket, fetch_and_draw, title);


async function fetch_and_draw(latest_timestamp?: string) {
  // use stored value of timestamp if none is provided, otherwise update stored value:
  if (latest_timestamp !== undefined) {
    stored_timestamp.value = latest_timestamp;
  }
  // send timestamp to control server-side cache
  const timestamp = stored_timestamp.value;
  const output_vars = (vars.value.length > 0) ? [...vars.value] : null;
  const loading_delay = 50; // ms
  // if the plot loads faster than the timeout, don't show spinner
  const show_loader = setTimeout(() => {
    drawing_busy.value = true;
  }, loading_delay);
  const payload = await props.socket.asyncEmit('get_correlation_plot', do_sort.value, max_rows.value, n_bins.value, output_vars, timestamp) as Plotly.PlotlyDataLayoutConfig;
  const plotdata = { ...payload };
  const { data, layout } = plotdata;
  const config: Partial<Plotly.Config> = { responsive: true, scrollZoom: true, modeBarButtonsToAdd: [ SVGDownloadButton ] };
  await Plotly.react(plot_div.value as HTMLDivElement, [...data], layout, config);
  clearTimeout(show_loader);
  drawing_busy.value = false;

  if (!callback_registered.value) {
    if (plot_div.value?.on) {
      plot_div.value.on('plotly_click', (ev) => {
        // if we're already showing only one plot, bail out:
        if (vars.value.length === 2) {
          return
        }
        // we are putting an array of numbers in customdata on the server side.
        vars.value = (ev as Plotly.PlotMouseEvent).points[0].data.customdata as number[];
        fetch_and_draw();
      });
      callback_registered.value = true;
    }
  }
}

function reset_vars() {
  vars.value = [];
  fetch_and_draw();
}

</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div class="row g-1">
      <div class="col-3 form-check align-items-center align-middle text-start">
        <input class="form-check-input" type="checkbox" v-model="do_sort" id="sort-by-correlation" @change="fetch_and_draw()" />
        <label class="form-check-label pe-2 w-auto text-wrap" for="sort-by-correlation">Sort by correlation </label>
      </div>
      <div class="col-5 d-flex flex-row align-items-center">
        <label class="col-form-label w-auto text-nowrap flex-fill" for="max-rows">Max. rows</label>
        <input class="form-control flex-fill" type="number" v-model="max_rows" id="max-rows" @change="fetch_and_draw()" />
      </div>
      <div class="col-4 d-flex flex-row align-items-center">
        <label class="col-form-label w-auto" for="n-bins">Bins</label>
        <input class="form-control flex-fill" type="number" v-model="n_bins" id="n-bins" @change="fetch_and_draw()" />
      </div>
    </div>
    <button class="btn btn-primary" v-if="vars.length > 0" @click="reset_vars">Reset plot</button>
    <div class="flex-grow-1 position-relative">
      <div class="w-100 h-100 plot-div" ref="plot_div"></div>
      <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column align-items-center justify-content-center loading" v-if="drawing_busy">
        <span class="spinner-border text-primary"></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
div.plot-div :deep(g.hovertext > path) {
  opacity: 0.5;
}

div.loading {
  background-color: rgba(255, 255, 255, 0.4);
}
span.spinner-border {
  width: 3rem;
  height: 3rem;
}
</style>