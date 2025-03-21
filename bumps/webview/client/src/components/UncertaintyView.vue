<script setup lang="ts">
import { ref } from "vue";
// @ts-ignore - plotly.js does not define Bar as type
import Bar from "plotly.js/lib/bar";
import * as Plotly from "plotly.js/lib/core";
import { v4 as uuidv4 } from "uuid";
import type { AsyncSocket } from "../asyncSocket.ts";
import { cache } from "../plot_cache";
import { SVGDownloadButton } from "../plotly_extras";
import { setupDrawLoop } from "../setupDrawLoop";

// workaround to PlotlyModule not being exported as type!
type RegisterTypes = Parameters<typeof Plotly.register>[0];
type PlotlyModule = Exclude<RegisterTypes, any[]>;

// server sends annotations with name field
type CustomAnnotations = Plotly.Annotations & { name: string };

Plotly.register([Bar as PlotlyModule]);

const title = "Uncertainty";
const plot_div = ref<HTMLDivElement>();
const plot_div_id = ref(`div-${uuidv4()}`);
const props = defineProps<{
  socket: AsyncSocket;
}>();
const show_labels = ref(true);

async function on_change_show_labels() {
  let { plotdata } = (cache[title] as { timestamp: string; plotdata: Plotly.PlotlyDataLayoutConfig }) ?? {};
  if (plotdata) {
    apply_label_visibility(plotdata.layout as Plotly.Layout, show_labels.value);
    await Plotly.react(plot_div_id.value, [...plotdata.data], plotdata.layout);
  }
}

const { drawing_busy } = setupDrawLoop("updated_uncertainty", props.socket, fetch_and_draw, title);

async function fetch_and_draw(latest_timestamp: string): Promise<void> {
  let { timestamp, plotdata } = (cache[title] as { timestamp: string; plotdata: Plotly.PlotlyDataLayoutConfig }) ?? {};
  if (timestamp !== latest_timestamp) {
    console.log(`Fetching new model uncertainty plot: timestamp: ${timestamp}, latest: ${latest_timestamp}`);

    const payload = (await props.socket.asyncEmit(
      "get_uncertainty_plot",
      latest_timestamp
    )) as Plotly.PlotlyDataLayoutConfig;
    plotdata = { ...payload };
    cache[title] = { timestamp: latest_timestamp, plotdata };
  }
  const { data, layout } = plotdata;
  if (layout == null || data == null) {
    await Plotly.purge(plot_div_id.value);
    return;
  }
  delete layout?.width;
  delete layout?.height;
  const config = { responsive: true, scrollZoom: true, modeBarButtonsToAdd: [SVGDownloadButton] };
  apply_label_visibility(layout as Plotly.Layout, show_labels.value);
  await Plotly.react(plot_div_id.value, [...data], layout, config);
}

function apply_label_visibility(layout: Plotly.Layout, visible: boolean) {
  layout.showlegend = visible;
  for (let item_name in layout) {
    if (item_name.startsWith("xaxis")) {
      (layout as any)[item_name].showticklabels = visible;
    }
  }
  for (let annotation of layout?.annotations ?? []) {
    if ((annotation as CustomAnnotations)?.name === "label") {
      annotation.visible = visible;
    }
  }
}
</script>

<template>
  <div class="container d-flex flex-grow-1 flex-column">
    <div class="d-flex flex-row justify-content-between align-items-center">
      <div class="form-check">
        <input
          id="show_labels"
          v-model="show_labels"
          class="form-check-input"
          type="checkbox"
          @change="on_change_show_labels"
        />
        <label class="form-check-label" for="show_labels">Show labels</label>
      </div>
    </div>
    <div class="flex-grow-1 position-relative">
      <div :id="plot_div_id" ref="plot_div" class="w-100 h-100 plot-div"></div>
      <div
        v-if="drawing_busy"
        class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column align-items-center justify-content-center loading"
      >
        <span class="spinner-border text-primary"></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
svg {
  width: 100%;
}
span.spinner-border {
  width: 3rem;
  height: 3rem;
}
</style>
