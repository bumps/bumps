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
const model_names = ref<{name: string, part_name: string, model_index: number, part_index: number}[]>([]);
const current_model = ref<[number, number]>([0, 0]);

const props = defineProps<{
  socket: Socket,
}>();

const { draw_requested } = setupDrawLoop('update_parameters', props.socket, fetch_and_draw, title);

async function get_model_names() {
  model_names.value = await props.socket.asyncEmit("get_model_names");
}

props.socket.on('model_loaded', get_model_names);

onMounted(async () => {
  await get_model_names();
});

async function fetch_and_draw() {
  const [model_index, sample_index] = current_model.value;
  const payload = await props.socket.asyncEmit('get_profile_plot', model_index, sample_index);
  let plotdata = { ...payload };
  const { data, layout } = plotdata;
  const config = { responsive: true }
  await Plotly.react(plot_div_id.value, [...data], layout, config);
}

</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <select v-model="current_model" @change="draw_requested = true">
      <option v-for="{name, part_name, model_index, part_index} in model_names" :key="`${model_index}:${part_index}`" :value="[model_index, part_index]">
        {{ model_index }}:{{ part_index }} --- {{ name ?? "" }}:{{ part_name ?? "" }}</option>
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