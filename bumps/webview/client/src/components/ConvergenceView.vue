<script setup lang="ts">
import { ref, watchEffect } from "vue";
import * as Plotly from "plotly.js/lib/core";
import { shared_state } from "../app_state.ts";
import type { AsyncSocket } from "../asyncSocket.ts";
import { SVGDownloadButton } from "../plotly_extras";
import { setupDrawLoop } from "../setupDrawLoop";

const title = "Convergence";
const plot_div = ref<HTMLDivElement>();
const cutoff = ref(0.25);
const negative_trim_portion = ref(-1.0);
const trim_is_set = ref(false);
const show_trim_controls = ref(false);
const trim_control_active = ref(false);

const props = defineProps<{
  socket: AsyncSocket;
}>();

interface ConvergencePlotData {
  plotdata: Plotly.PlotlyDataLayoutConfig | null;
  portion: number | null;
}

const { draw_requested } = setupDrawLoop("updated_convergence", props.socket, fetch_and_draw, title);

async function fetch_and_draw() {
  if (plot_div.value == null) {
    return;
  }
  const trim_portion_preview_value = trim_control_active.value ? -negative_trim_portion.value : null;
  const payload = (await props.socket.asyncEmit(
    "get_convergence_plot",
    cutoff.value,
    trim_portion_preview_value
  )) as ConvergencePlotData;
  const { plotdata, portion } = { ...payload };
  const { data, layout } = plotdata ?? {};
  if (portion != null) {
    if (!trim_control_active.value) {
      negative_trim_portion.value = -portion;
    }
    trim_is_set.value = true;
  } else {
    trim_is_set.value = false;
  }
  const config = { responsive: true, scrollZoom: true, modeBarButtonsToAdd: [SVGDownloadButton] };
  if (data == null) {
    Plotly.purge(plot_div.value);
  } else {
    await Plotly.react(plot_div.value, [...data], layout, config);
  }
}

async function setPortion() {
  // This function is called when the trim portion slider is changed.
  // The server sends "updated_convergence" after receiving it, so we will get
  // a new convergence plot with the updated trim portion.
  trim_control_active.value = false; // reset the control active state
  await props.socket.asyncEmit("set_trim_portion", -negative_trim_portion.value);
}

watchEffect(async () => {
  // only show trim controls if the fit is not active and the trim portion
  // is defined
  if (shared_state.active_fit?.fitter_id === undefined && trim_is_set.value) {
    show_trim_controls.value = true;
  } else {
    show_trim_controls.value = false;
  }
  if (plot_div.value != null) {
    // adjust the plot size when slider is shown or hidden above
    Plotly.Plots.resize(plot_div.value);
  }
});

function draw_plot() {
  draw_requested.value = true;
}

function previewPortion() {
  // prevent updating negative_trim_portion until the user releases the slider
  trim_control_active.value = true;
  draw_plot();
}
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div class="row px-2 align-items-center">
      <div class="col-auto">
        <label for="cutoff_control" class="col-form-label">Steps shown</label>
      </div>
      <div class="col">
        <input
          id="cutoff_control"
          v-model.number="cutoff"
          type="range"
          min="0"
          max="1.0"
          step="0.01"
          class="form-range"
          @input="draw_plot"
        />
      </div>
    </div>
    <div v-if="show_trim_controls" class="row px-2 align-items-center">
      <div class="col-auto">
        <label for="portion_control" class="col-form-label">Trim</label>
      </div>
      <div class="col">
        <input
          id="portion_control"
          v-model.number="negative_trim_portion"
          type="range"
          min="-1.0"
          max="0.0"
          step="0.01"
          class="form-range"
          @change="setPortion"
          @input="previewPortion"
        />
      </div>
    </div>
    <div ref="plot_div" class="flex-grow-1"></div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
.fixed-width-span {
  display: inline-block;
  width: 2em;
}
</style>
