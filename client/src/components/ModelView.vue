<script setup lang="ts">
/// <reference types="@types/uuid"/>
import { ref, onMounted, onBeforeUnmount, watch, onUpdated, computed, shallowRef } from 'vue';
import { Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import { setupDrawLoop } from '../setupDrawLoop';
import * as Plotly from 'plotly.js/lib/core';

const title = "Profile";
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const model_names = ref<string[]>([]);
const current_model = ref(0);

const props = defineProps<{
  socket: Socket,
}>();

const { draw_requested } = setupDrawLoop('update_parameters', props.socket, fetch_and_draw, title);

props.socket.on('model_loaded', ({ message: { model_names: new_model_names } }) => {
  model_names.value = new_model_names;
});

onMounted(async () => {
  const messages = await props.socket.asyncEmit('get_topic_messages', 'model_loaded');
  const new_model_names = messages?.[0]?.message?.model_names;
  if (new_model_names != null) {
    model_names.value = new_model_names;
  }
});

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_profile_plot', current_model.value);
  let plotdata = { ...payload };
  const { data, layout } = plotdata;
  const config = { responsive: true }
  await Plotly.react(plot_div_id.value, [...data], layout, config);
}

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