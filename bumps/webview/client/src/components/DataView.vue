<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref, onMounted, watch, onUpdated, computed, shallowRef, ssrContextKey } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import mpld3 from 'mpld3';
import { v4 as uuidv4 } from 'uuid';
import type { AsyncSocket } from '../asyncSocket';
import { setupDrawLoop } from '../setupDrawLoop';

const title = "Data";
const plot_div = ref<HTMLDivElement | null>(null);
const plot_div_id = ref(`div-${uuidv4()}`);


const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('update_parameters', props.socket, fetch_and_draw);

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_data_plot');
  // console.log(payload);
  let { fig_type, plotdata } = payload as { fig_type: 'plotly' | 'mpld3', plotdata: object};
  if (fig_type === 'plotly') {
    const { data, layout } = plotdata as Plotly.PlotlyDataLayoutConfig;
    const config = {responsive: true}
    await Plotly.react(plot_div_id.value, [...data], layout, config);
  }
  else if (fig_type === 'mpld3') {
    let mpld3_data = plotdata as { width: number, height: number };
    mpld3_data.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
    mpld3_data.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
    mpld3.draw_figure(plot_div_id.value, mpld3_data, false, true);
  }
}

</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>
