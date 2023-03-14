<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from '../setupDrawLoop';
import mpld3 from 'mpld3';

const title = "Convergence";
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const props = defineProps<{
  socket: Socket,
}>();

setupDrawLoop('convergence_update', props.socket, fetch_and_draw, title);

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_convergence_plot');
  let plotdata = { ...payload };
  // console.log({plotdata});
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