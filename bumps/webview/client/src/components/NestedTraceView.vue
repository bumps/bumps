<script setup lang="ts">
import { ref, shallowRef } from "vue";
import * as Plotly from "plotly.js/lib/core";
import type { Socket } from "socket.io-client";
import { v4 as uuidv4 } from "uuid";
import { configWithSVGDownloadButton } from "../plotly_extras";
import { setupDrawLoop } from "../setupDrawLoop";

const plot_data = shallowRef<Plotly.Data[]>([]);
const plot_layout = shallowRef<Partial<Plotly.Layout>>();
const plot_config = shallowRef<Partial<Plotly.Config>>();
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const props = defineProps<{
  socket: Socket;
}>();

setupDrawLoop("updated_uncertainty", props.socket, fetch_and_draw);

async function fetch_and_draw() {
  const payload = (await props.socket.asyncEmit("get_nested_trace_plot")) as Plotly.PlotlyDataLayoutConfig;

  let plotdata = { ...payload };
  const { data, layout } = plotdata;
  if (data == null || layout == null) {
    await Plotly.purge(plot_div_id.value);
    return;
  }
  plot_data.value = data;
  plot_layout.value = layout;
  const config: Partial<Plotly.Config> = {
    responsive: true,
    ...configWithSVGDownloadButton,
  };

  plot_config.value = config;

  plot_data.value = data;

  await draw_plot();
}

async function draw_plot() {
  let data = plot_data.value;
  await Plotly.react(plot_div.value as HTMLDivElement, [...data], plot_layout.value, plot_config.value);
}
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div :id="plot_div_id" ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>
