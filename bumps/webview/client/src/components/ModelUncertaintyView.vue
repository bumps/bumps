<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from '../setupDrawLoop';
import { cache } from '../plotcache';
import mpld3 from 'mpld3';

const title = "Model Uncertainty"
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);

const props = defineProps<{
  socket: Socket,
}>();

setupDrawLoop('uncertainty_update', props.socket, fetch_and_draw, title);

async function fetch_and_draw(latest_timestamp: string) {
  let { timestamp, plotdata } = cache[title] ?? {};
  if (timestamp !== latest_timestamp) {
    console.log("fetching new model uncertainty plot", timestamp, latest_timestamp);
    const payload = await props.socket.asyncEmit('get_model_uncertainty_plot');
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
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>