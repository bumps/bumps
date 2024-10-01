<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref, shallowRef, onMounted } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import * as mpld3 from 'mpld3';
import { v4 as uuidv4 } from 'uuid';
import type { AsyncSocket } from '../asyncSocket.ts';
import { setupDrawLoop } from '../setupDrawLoop';
import { configWithSVGDownloadButton } from '../plotly_extras.mjs';

type PlotInfo = {title: string, change_with: string, model_index: number};
const title = 'Custom'
const plot_div = ref<HTMLDivElement | null>(null);
const plot_div_id = ref(`div-${uuidv4()}`);
const plot_infos = ref<PlotInfo[]>([]);
//const current_plot_name = ref<PlotNameInfo>({"name": "", "change_with": "parameters", "model_index": 0});
const current_plot_index = ref<number>(0);
const error_text = ref<string>("")

// add types to mpld3
declare global {
  interface mpld3 {
    draw_figure: (div_id: string, data: { width: number, height: number }, process: boolean | Function, clearElem: boolean) => void;
  }
  var mpld3: mpld3;
}

const props = defineProps<{
  socket: AsyncSocket,
}>();

async function get_custom_plot_infos() {
  const new_infos = await props.socket.asyncEmit("get_custom_plot_info") as PlotInfo[];
  if (new_infos == null) {
    return;
  }
  plot_infos.value = new_infos.filter(a => a.change_with === 'parameter')
  current_plot_index.value = 0
}

props.socket.on('model_loaded', get_custom_plot_infos);
onMounted(async () => {
  await get_custom_plot_infos();
});

const { draw_requested } = setupDrawLoop('updated_parameters', props.socket, fetch_and_draw);

async function fetch_and_draw() {
  const { model_index, title } = plot_infos.value[current_plot_index.value];
  const payload = await props.socket.asyncEmit('get_custom_plot', model_index, title);
  let { fig_type, plotdata } = payload as { fig_type: 'plotly' | 'matplotlib' | 'error', plotdata: object};
  if (fig_type === 'plotly') {
    error_text.value = ""
    const { data, layout } = plotdata as Plotly.PlotlyDataLayoutConfig;
    const config: Partial<Plotly.Config> = {
    responsive: true,
    edits: {
      legendPosition: true
    }, 
    ...configWithSVGDownloadButton
    }

    await Plotly.react(plot_div.value as HTMLDivElement, [...data], layout, config);
  }
  else if (fig_type === 'matplotlib') {
    error_text.value = ""
    let mpld3_data = plotdata as { width: number, height: number };
    mpld3_data.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
    mpld3_data.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
    mpld3.draw_figure(plot_div_id.value, mpld3_data, false, true);
  }
  else if (fig_type === 'error') {
    error_text.value = String(plotdata).replace(/[\n]+/g, "<br>")
  }
  
}

</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <select
      v-model="current_plot_index"
      @change="draw_requested = true"
      >
      <option v-for="(plot_info, index) in plot_infos" :key="index" :value="index">
        {{ plot_info.model_index }}:  {{ plot_info.title ?? "" }} </option>
    </select>
    <div v-if="error_text" class="flex-grow-0" ref="error_div">
      <div style="color:red; font-size: larger; font-weight: bold;">
        Plotting error:
      </div>
      <div v-html="error_text"></div>
    </div>    
    <div v-if="!error_text" class="flex-grow-1" ref="plot_div" :id=plot_div_id>
    </div>
  </div>
</template>

<style scoped>
.plot-mode {
  width: 100%;
}
</style>