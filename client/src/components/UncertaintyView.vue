<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import * as Plotly from 'plotly.js/lib/core';
import Bar from 'plotly.js/lib/bar';

Plotly.register([
  Bar,
])

const plot_div = ref<HTMLDivElement>();
const draw_requested = ref(false);
const drawing_busy = ref(false);
const plot_div_id = ref(`div-${uuidv4()}`);
const props = defineProps<{
  socket: Socket,
  visible: boolean
}>();

onMounted(() => {
  props.socket.on('uncertainty_update', () => {
    draw_requested.value = true;
  });
  window.requestAnimationFrame(draw_if_needed);
});


function fetch_and_draw() {
  props.socket.emit('get_uncertainty_plot', async (payload) => {
    if (plot_div.value == null) {
      return
    }
    if (props.visible) {
      let plotdata = { ...payload };
      const { data, layout } = plotdata;
      delete layout.width;
      delete layout.height;
      delete layout.heighth;
      const config = { responsive: true }
      const plot = await Plotly.react(plot_div_id.value, [...data], layout, config);
    }
    drawing_busy.value = false;
  });
}

function draw_if_needed(timestamp: number) {
  if (drawing_busy.value) {
    console.log("busy!");
  }
  if (draw_requested.value && props.visible && !drawing_busy.value) {
    drawing_busy.value = true;
    draw_requested.value = false;
    fetch_and_draw();
  }
  window.requestAnimationFrame(draw_if_needed);
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