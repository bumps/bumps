<script setup lang="ts">
import { ref } from "vue";
import * as Plotly from "plotly.js/lib/core";
import type { AsyncSocket } from "../asyncSocket.ts";
import { SVGDownloadButton } from "../plotly_extras";
import { setupDrawLoop } from "../setupDrawLoop";

const title = "Convergence";
const plot_div = ref<HTMLDivElement>();
const props = defineProps<{
  socket: AsyncSocket;
}>();

setupDrawLoop("updated_convergence", props.socket, fetch_and_draw, title);

async function fetch_and_draw() {
  const payload = (await props.socket.asyncEmit("get_convergence_plot")) as Plotly.PlotlyDataLayoutConfig;
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
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>
