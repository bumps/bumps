<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import mpld3 from 'mpld3';

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
  props.socket.emit('get_parameter_trace_plot', (payload) => {
    if (props.visible) {
      let plotdata = { ...payload };
      console.log({plotdata});
      plotdata.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
      plotdata.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
      /* Data Parsing Functions */
      // mpld3.draw_figure = function(figid, spec, process, clearElem) {}
      mpld3.draw_figure(plot_div_id.value, plotdata, false, true);
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