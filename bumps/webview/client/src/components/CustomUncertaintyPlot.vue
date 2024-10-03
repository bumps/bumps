<script setup lang="ts">
/// <reference types="@types/plotly.js" />
import { ref, shallowRef, onMounted, nextTick } from 'vue';
import * as Plotly from 'plotly.js/lib/core';
import * as mpld3 from 'mpld3';
import { v4 as uuidv4 } from 'uuid';
import type { AsyncSocket } from '../asyncSocket.ts';
import { setupDrawLoop } from '../setupDrawLoop';
import { cache } from '../plotcache';
import { configWithSVGDownloadButton } from '../plotly_extras.mjs';
import { TableData} from './CSVTable.vue'
import CSVTable from './CSVTable.vue'

type PlotInfo = {title: string, change_with: string, model_index: number};
const panel_title = "Custom Uncertainty"
const figtype = ref<String>("")
const plot_div = ref<HTMLDivElement | null>(null);
const plot_div_id = ref(`div-${uuidv4()}`);
const plot_infos = ref<PlotInfo[]>([]);
const current_plot_index = ref<number>(0);
const error_text = ref<string>("")
const n_samples = ref(50);
const table_data = ref<TableData>({raw: "", header: [], rows: [[]]})

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

async function get_custom_plot_info() {
  const new_infos = await props.socket.asyncEmit("get_custom_plot_info") as PlotInfo[];
  if (new_infos == null) {
    return;
  }
  plot_infos.value = new_infos.filter(a => a.change_with === "uncertainty")
  current_plot_index.value = 0
}

props.socket.on('model_loaded', get_custom_plot_info);
onMounted(async () => {
  await get_custom_plot_info();
});

const { draw_requested, drawing_busy } = setupDrawLoop('updated_uncertainty', props.socket, fetch_and_draw, panel_title);

async function fetch_and_draw(latest_timestamp?: string) {
  const { model_index, title } = plot_infos.value[current_plot_index.value];
  const cache_key = `${panel_title}:${model_index}:${title}:${n_samples.value}`;
  const read_cache = cache[cache_key] as { timestamp: string, plotdata: object } ?? {};
  const timestamp = read_cache.timestamp;
  let payload = read_cache.plotdata;
  //console.log([timestamp, latest_timestamp])
  if (latest_timestamp === undefined || timestamp !== latest_timestamp) {
    console.log("fetching new model uncertainty plot", timestamp, latest_timestamp);
    payload = await props.socket.asyncEmit('get_custom_plot', model_index, title, n_samples.value);
    cache[cache_key] = {timestamp: latest_timestamp, plotdata: payload};
  }
  //console.log(payload)
  const { fig_type, plotdata } = payload as { fig_type: 'plotly' | 'matplotlib' | 'table' | 'error', plotdata: object};
  figtype.value = fig_type
  if (fig_type === 'plotly') {
    await nextTick();
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
    await nextTick();
    let mpld3_data = plotdata as { width: number, height: number };
    mpld3_data.width = Math.round(plot_div.value?.clientWidth ?? 640) - 16;
    mpld3_data.height = Math.round(plot_div.value?.clientHeight ?? 480) - 16;
    mpld3.draw_figure(plot_div_id.value, mpld3_data, false, true);
  }
  else if (fig_type === 'table') {
    await nextTick();
    table_data.value = plotdata as TableData;
  }
  else if (fig_type === 'error') {
      error_text.value = String(plotdata).replace(/[\n]+/g, "<br>");
  }
  else {
    figtype.value = 'error';
    error_text.value = "Unknown figure type " + fig_type;
  }
}

</script>

<template>
  <div class="container d-flex flex-column flex-grow-1">
    <div class="row g-3">
      <div class="col-md-8">
        <select
          v-model="current_plot_index"
          @change="draw_requested = true"
          >
          <option v-for="(plot_info, index) in plot_infos" :key="index" :value="index">
            {{ plot_info.model_index }}:  {{ plot_info.title ?? "" }} </option>
        </select>
      </div>
      <div class="col-md-2 align-right justify-content-md-right">
        <label class="form-label" for="n_samples"
          title="Number of samples to draw from the uncertainty population">Num. samples:</label>
      </div>
      <div class="col-md-2 align-left">
        <input class="form-control" type="number" v-model="n_samples" id="n_samples" @change="draw_requested = true" />
      </div>
    </div>      
    <div v-if="figtype==='error'" class="flex-grow-0" ref="error_div">
      <div style="color:red; font-size: larger; font-weight: bold;">
        Plotting error:
      </div>
      <div v-html="error_text"></div>
    </div>
    <div v-else-if="figtype==='table'" class="flex-grow-0">
      <CSVTable :table_data="table_data"></CSVTable>
    </div>
    <div v-else class="flex-grow-1 position-relative">
      <div class="w-100 h-100 plot-div" ref="plot_div" :id=plot_div_id></div>
    </div>
    <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column align-items-center justify-content-center loading" v-if="drawing_busy">
      <span class="spinner-border text-primary"></span>
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
.hidden {
  display: none;
}
span.spinner-border {
  width: 3rem;
  height: 3rem;
}
</style>
