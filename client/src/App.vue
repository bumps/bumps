<script setup lang="ts">
import { Button } from 'bootstrap/dist/js/bootstrap.esm.js';
import { onMounted, ref, shallowRef } from 'vue';
import { io } from 'socket.io-client';
import FitOptions from './components/FitOptions.vue';
import DataView from './components/DataView.vue';
import ModelView from './components/ModelView.vue';
import PanelTabContainer from './components/PanelTabContainer.vue';
import FileBrowser from './components/FileBrowser.vue';
import SummaryView from './components/SummaryView.vue';
import ModelInspect from './components/ModelInspect.vue';
import ModelViewPlotly from './components/ModelViewPlotly.vue';
import ParameterView from './components/ParameterView.vue';
import LogView from './components/LogView.vue';
import ConvergenceView from './components/ConvergenceView.vue';
import CorrelationView from './components/CorrelationView.vue';
import ParameterTraceView from './components/ParameterTraceView.vue';
import ModelUncertaintyView from './components/ModelUncertaintyView.vue';
import UncertaintyView from './components/UncertaintyView.vue';

// import { FITTERS as FITTER_DEFAULTS } from './fitter_defaults';

const panels = [
  {title: 'Reflectivity', component: DataView},
  {title: 'Summary', component: SummaryView},
  {title: 'Log', component: LogView},
  {title: 'Convergence', component: ConvergenceView},
  {title: 'Profile', component: ModelView},
  {title: 'Model', component: ModelInspect},
  {title: 'Profile2', component: ModelViewPlotly},
  {title: 'Parameters', component: ParameterView},
  {title: 'Correlations', component: CorrelationView},
  {title: 'Trace', component: ParameterTraceView},
  {title: 'Model Uncertainty', component: ModelUncertaintyView},
  {title: 'Uncertainty', component: UncertaintyView},
];

const connected = ref(false);
const menuToggle = ref<HTMLButtonElement>();
const fitOptions = ref<typeof FitOptions>();
const fileBrowser = ref<typeof FileBrowser>();
const fileBrowserSelectCallback = ref((pathlist: string[], filename: string) => { });
const model_loaded = shallowRef<{pathlist: string[], filename: string}>();

// Create a SocketIO connection, to be passed to child components
// so that they can do their own communications with the host.
const base_path = window.location.pathname;

const socket = io('', {
   // this is mostly here to test what happens on server fail:
   path: `${base_path}socket.io`,
   reconnectionAttempts: 10
});

socket.on('connect', () => {
  console.log(socket.id);
  connected.value = true;
});

socket.on('disconnect', (payload) => {
  console.log("disconnected!", payload);
  connected.value = false;
})

socket.on('model_loaded', ({message: {pathlist, filename}}) => {
  model_loaded.value = {pathlist, filename};
})

function disconnect() {
  socket.disconnect();
  connected.value = false;
}

function selectOpenFile() {
  if (fileBrowser.value) {
    fileBrowserSelectCallback.value = (pathlist, filename) => {
      socket.emit("load_problem_file", pathlist, filename);
    }
    fileBrowser.value.open();
  }
  // const path = prompt("full path to file:");
  // if (path != null) {
  //   let defaulted_path = (path == '') ? '/home/bbm/dev/refl1d-modelbuilder/ISIS_GGG/GGG_GdIG_MultiFit.py' : path;
  //   socket.emit("load_model_file", defaulted_path);
  // }
  // file_picker.value?.click();
}

async function saveFile(ev: Event) {
  if (model_loaded.value === undefined) {
    alert('no file to save');
    return;
  }
  const {pathlist, filename} = model_loaded.value;
  console.log('saving:', {pathlist, filename});
  socket.emit("save_problem_file", pathlist, filename, false, async(confirm_overwrite: boolean) => {
    if (confirm_overwrite) {
      const overwrite = await confirm(`File ${filename} exists: overwrite?`);
      if (overwrite) {
        socket.emit("save_problem_file", {pathlist, filename, overwrite: true});
      }
    }
  });
}

function reloadModel() {
  if (model_loaded.value) {
    const {pathlist, filename} = model_loaded.value;
    socket.emit("load_problem_file", pathlist, filename);
  }
}

function openFitOptions() {
  fitOptions.value?.open();
}

function startFit() {
  const fitter_active = fitOptions.value?.fitter_active;
  const fitter_settings = fitOptions.value?.fitter_settings;

  if (fitter_active && fitter_settings) {
    const fit_args = fitter_settings[fitter_active];
    socket.emit("start_fit_thread", fitter_active, fit_args.settings);
  }
}

function stopFit() {
  socket.emit("stop_fit")
}

onMounted(() => {
  const menuToggleButton = new Button(menuToggle.value);
});

</script>

<template>
  <div class="h-100 w-100 m-0 d-flex flex-column">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
      <div class="container-fluid">
        <div class="navbar-brand">
          <img src="./assets/refl1d-icon_256x256x32.png" alt="" height="24" class="d-inline-block align-text-middle">
          Refl1D
        </div>
        <button ref="menuToggle" class="navbar-toggler" type="button" data-bs-toggle="collapse"
          data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false"
          aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarSupportedContent">
          <ul class="navbar-nav me-auto mb-2 mb-lg-0">
            <!-- <li class="nav-item dropdown">
              <div class="nav-link dropdown-toggle"  role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                Session
              </div>
              <ul class="dropdown-menu">
                <li><button class="btn btn-link dropdown-item"  @click="connect">New</div></li>
                <li><button class="btn btn-link dropdown-item"  @click="disconnect">Disconnect</div></li>
                <li><button class="btn btn-link dropdown-item"  @click="reconnect">Existing</div></li>
              </ul>
            </li> -->
            <li class="nav-item dropdown">
              <button class="btn btn-link nav-link dropdown-toggle" role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                File
              </button>
              <ul class="dropdown-menu">
                <li><button class="btn btn-link dropdown-item" @click="selectOpenFile">Open</button></li>
                <li><button class="btn btn-link dropdown-item" :disabled="!model_loaded" @click="saveFile">Save</button></li>
                <li><button class="btn btn-link dropdown-item" :class="{disabled: model_loaded === undefined}"  @click="reloadModel">Reload</button></li>
                <li>
                  <hr class="dropdown-divider">
                </li>
                <li><button class="btn btn-link dropdown-item" >Quit</button></li>
              </ul>
            </li>
            <li class="nav-item dropdown">
              <button class="btn btn-link nav-link dropdown-toggle"  role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                Fitting
              </button>
              <ul class="dropdown-menu">
                <li><button class="btn btn-link dropdown-item"  @click="startFit">Start</button></li>
                <li><button class="btn btn-link dropdown-item"  @click="stopFit">Stop</button></li>
                <li>
                  <hr class="dropdown-divider">
                </li>
                <li><button class="btn btn-link dropdown-item"  @click="openFitOptions">Options...</button></li>
              </ul>
            </li>
            <!-- <li class="nav-item dropdown">
              <div class="nav-link dropdown-toggle"  role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                Reflectivity
              </div>
              <ul class="dropdown-menu">
                <li v-for="plot_type in REFLECTIVITY_PLOTS" :key="plot_type">
                  <div :class="{'dropdown-item': true, active: (plot_type === reflectivity_type)}" 
                    @click="set_reflectivity(plot_type)">{{plot_type}}</div>
                </li>
              </ul>
            </li> -->
          </ul>
          <div class="d-flex">
            <div id="connection_status"
              :class="{'btn': true, 'btn-outline-success': connected, 'btn-outline-danger': !connected}">
              {{(connected) ? 'connected' : 'disconnected'}}</div>

          </div>
        </div>
      </div>
    </nav>
    <div class="flex-grow-1 row overflow-hidden">
      <div class="col d-flex flex-column mh-100">
        <PanelTabContainer :panels="panels" :socket="socket" :initially_active="0"/>
      </div>
      <div class="col d-flex flex-column mh-100">
        <PanelTabContainer :panels="panels" :socket="socket" :initially_active="1"/>
      </div>
    </div>
  </div>
  <FitOptions ref="fitOptions" :socket="socket" />
  <FileBrowser ref="fileBrowser" :socket="socket" title="Load Model File" :callback="fileBrowserSelectCallback" />
</template>

<style>
html,
body,
#app {
  height: 100%;
  width: 100%;
  overflow-x: hidden;
}

div#connection_status {
  pointer-events: none;
}
</style>
