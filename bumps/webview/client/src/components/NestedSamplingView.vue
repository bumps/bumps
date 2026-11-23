<script setup lang="ts">
import { ref } from "vue";
import * as Plotly from "plotly.js/lib/core";
import type { AsyncSocket } from "../asyncSocket.ts";
import { configWithSVGDownloadButton } from "../plotly_extras";
import { setupDrawLoop } from "../setupDrawLoop";

const title = "Nested Sampling";
const plot_div = ref<HTMLDivElement>();
const table = ref<[string, string][]>([]);

const props = defineProps<{
  socket: AsyncSocket;
}>();

interface NestedPlotData {
  plotdata: Plotly.PlotlyDataLayoutConfig;
  table: [string, string][];
}

setupDrawLoop("updated_nested", props.socket, fetch_and_draw, title);
const plot_config: Partial<Plotly.Config> = {
  ...configWithSVGDownloadButton,
  responsive: true,
  scrollZoom: true,
};

async function fetch_and_draw() {
  if (plot_div.value == null) {
    return;
  }
  const payload = (await props.socket.asyncEmit("get_nested_plot")) as NestedPlotData | null;
  table.value = payload?.table ?? [];
  if (payload == null) {
    Plotly.purge(plot_div.value);
  } else {
    const { data, layout } = payload.plotdata;
    await Plotly.react(plot_div.value, [...data], layout, plot_config);
  }
}
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <dl class="stats px-2 pt-2 mb-0">
      <div v-for="[label, value] in table" :key="label">
        <dt>{{ label }}</dt>
        <dd>{{ value }}</dd>
      </div>
    </dl>
    <div ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style scoped>
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(16rem, 1fr));
  column-gap: 2rem;
}
.stats div {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid var(--bs-border-color);
}
.stats dt {
  font-weight: normal;
  color: var(--bs-secondary-color);
}
.stats dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
}
</style>
