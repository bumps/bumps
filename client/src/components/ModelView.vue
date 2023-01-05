<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import * as Plotly from 'plotly.js/lib/core';

const title = "Profile";
const plot_div = ref<HTMLDivElement>();
const draw_requested = ref(false);
const plot_div_id = ref(`div-${uuidv4()}`);
const model_names = ref<string[]>([]);
const current_model = ref(0);
const props = defineProps<{
  socket: Socket,
  visible: boolean
}>();

props.socket.on('model_loaded', ({ message: { model_names: new_model_names } }) => {
  model_names.value = new_model_names;
});

onMounted(() => {
  props.socket.on('update_parameters', () => {
    draw_requested.value = true;
  });
  window.requestAnimationFrame(draw_if_needed);
});

function fetch_and_draw() {
  props.socket.emit('get_profile_plot', current_model.value, async (payload) => {
    console.log(payload);
    if (plot_div.value == null) {
      return
    }
    if (props.visible) {
      let plotdata = { ...payload };
      console.log(plotdata);
      const { data, layout } = plotdata;
      const config = { responsive: true }
      const plot = await Plotly.react(plot_div_id.value, [...data], layout, config);
    }
  });
}

function draw_if_needed(timestamp: number) {
  if (draw_requested.value && props.visible) {
    fetch_and_draw();
    draw_requested.value = false;
  }
  window.requestAnimationFrame(draw_if_needed);
}

// watch(() => props.visible, (value) => {
//   if (value) {
//     console.log('visible', value);
//     fetch_and_draw();
//   }
// });

</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <select v-model="current_model" @change="draw_requested = true">
      <option v-for="(name, index) in model_names" :key="index" :value="index">{{ index }}: {{ name ?? "" }}</option>
    </select>
    <div class="flex-grow-1" ref="plot_div" :id="plot_div_id">
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
</style>