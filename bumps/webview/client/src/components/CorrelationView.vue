<script setup lang="ts">
import { ref } from "vue";
import * as mpld3 from "mpld3";
import { v4 as uuidv4 } from "uuid";
import type { AsyncSocket } from "../asyncSocket.ts";
import { cache } from "../plot_cache";
import { setupDrawLoop } from "../setupDrawLoop";

const title = "Correlations";
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);

const props = defineProps<{
  socket: AsyncSocket;
}>();

setupDrawLoop("updated_uncertainty", props.socket, fetch_and_draw, title);

type MplD3PlotData = {
  width?: number;
  height?: number;
};

async function fetch_and_draw(latest_timestamp: string) {
  let { timestamp, plotdata } = (cache[title] as { timestamp: string; plotdata: MplD3PlotData }) ?? {};
  if (timestamp !== latest_timestamp) {
    console.log("Fetching new correlation plot", timestamp, latest_timestamp);
    const payload = (await props.socket.asyncEmit("get_correlation_plot")) as MplD3PlotData;
    plotdata = { ...payload };
    cache[title] = { timestamp: latest_timestamp, plotdata };
  }
  let mpld3_data = plotdata as { width: number; height: number };
  mpld3_data.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
  mpld3_data.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
  mpld3.draw_figure(plot_div_id.value, mpld3_data, false, true);
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
