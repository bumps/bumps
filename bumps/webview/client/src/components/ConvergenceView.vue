<script setup lang="ts">
import { ref } from "vue";
import * as Plotly from "plotly.js/lib/core";
import type { AsyncSocket } from "../asyncSocket.ts";
import { SVGDownloadButton } from "../plotly_extras";
import { setupDrawLoop } from "../setupDrawLoop";

const title = "Convergence";
const plot_div = ref<HTMLDivElement>();
const cutoff = ref(0.25);
const props = defineProps<{
  socket: AsyncSocket;
}>();

const { draw_requested } = setupDrawLoop("updated_convergence", props.socket, fetch_and_draw, title);

async function fetch_and_draw() {
  const payload = (await props.socket.asyncEmit("get_convergence_plot", cutoff.value)) as Plotly.PlotlyDataLayoutConfig;
  let plotdata = { ...payload };
  const { data, layout } = plotdata;
  const config = { responsive: true, scrollZoom: true, modeBarButtonsToAdd: [SVGDownloadButton] };
  if (plot_div.value == null) {
    return;
  }
  if (data == null) {
    Plotly.purge(plot_div.value);
  } else {
    await Plotly.react(plot_div.value, [...data], layout, config);
  }
}

function draw_plot() {
  draw_requested.value = true;
}
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div class="row px-2 align-items-center">
      <div class="col-auto">
        <label for="cutoff_control" class="col-form-label">Cutoff: </label>
        <span class="ps-1">{{ cutoff.toFixed(2) }}</span>
      </div>
      <div class="col">
        <input
          id="cutoff_control"
          v-model.number="cutoff"
          type="range"
          min="0"
          max="1.0"
          step="0.01"
          class="form-range"
          @input="draw_plot"
        />
      </div>
    </div>
    <div ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>
