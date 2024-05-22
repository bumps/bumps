<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref } from 'vue';
import type { AsyncSocket } from '../asyncSocket.ts';
import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from '../setupDrawLoop';
import { cache } from '../plotcache';
import * as mpld3 from 'mpld3';

const title = "Model Uncertainty"
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);

const props = defineProps<{
  socket: AsyncSocket,
}>();

const { drawing_busy } = setupDrawLoop('updated_uncertainty', props.socket, fetch_and_draw, title);

type MplD3PlotData = {
  width?: number,
  height?: number,
}

async function fetch_and_draw(latest_timestamp: string) {
  let { timestamp, plotdata } = cache[title] as { timestamp: string, plotdata: MplD3PlotData } ?? {};
  if (timestamp !== latest_timestamp) {
    console.log("fetching new model uncertainty plot", timestamp, latest_timestamp);
    const payload = await props.socket.asyncEmit('get_model_uncertainty_plot') as MplD3PlotData;
    plotdata = { ...payload };
    cache[title] = {timestamp: latest_timestamp, plotdata};
  }
  plotdata.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
  plotdata.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
  mpld3.draw_figure(plot_div_id.value, plotdata, false, true);
}

</script>
    
<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div class="flex-grow-1 position-relative">
      <div class="w-100 h-100 plot-div" ref="plot_div" :id="plot_div_id"></div>
      <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column align-items-center justify-content-center loading" v-if="drawing_busy">
        <span class="spinner-border text-primary"></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
span.spinner-border {
  width: 3rem;
  height: 3rem;
}
</style>