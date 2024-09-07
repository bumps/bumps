<script setup lang="ts">
import { Button } from 'bootstrap';
import { computed, onMounted, ref, shallowRef } from 'vue';
import { io } from 'socket.io-client';
import type { AsyncSocket } from './asyncSocket.ts';
import './asyncSocket.ts';  // patch Socket with asyncEmit
import {
  active_panel,
  active_layout,
  fitter_settings,
  active_fit,
  selected_fitter,
  fit_progress,
  fitOptions,
  fileBrowser,
  FileBrowserSettings,
  connected,
  model_file,
  notifications,
  menu_items,
  socket as socket_ref,
  addNotification,
  cancelNotification,
} from './app_state.ts';
import FitOptions from './components/FitOptions.vue';
import PanelTabContainer from './components/PanelTabContainer.vue';
import FileBrowser from './components/FileBrowser.vue';
import ServerShutdown from './components/ServerShutdown.vue';
import ServerStartup from './components/ServerStartup.vue';
import SessionMenu from './components/SessionMenu.vue';

const props = defineProps<{
  panels: {title: string, component: any }[],
  name?: string
}>();

type Message = {
  timestamp: string,
  message: object
}

const LAYOUTS = ["left-right", "top-bottom", "full"];
const menuToggle = ref<HTMLButtonElement>();
const nativefs = ref(false);

// Create a SocketIO connection, to be passed to child components
// so that they can do their own communications with the host.
const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);

const sio_base_path = urlParams.get('base_path') ?? window.location.pathname;
const sio_server = urlParams.get('server') ?? '';
const single_panel = urlParams.get('single_panel');
if (single_panel !== null) {
  active_layout.value = 'full';
  const panel_index = props.panels.findIndex(({title}) => (title.toLowerCase() == single_panel.toLowerCase()));
  if (panel_index > -1) {
    active_panel.value[0] = panel_index;
  }
  else {
    console.error(`Panel ${single_panel} not found`);
  }
}

const socket = io(sio_server, {
   path: `${sio_base_path}socket.io`,
}) as AsyncSocket;
socket_ref.value = socket;

const can_mount_local = (
  ('mountLocal' in socket) && ('showDirectoryPicker' in window)
)

socket.on('connect', async () => {
  console.log(socket.id);
  connected.value = true;
  const file_info = await socket.asyncEmit('get_shared_setting', 'model_file') as { pathlist: string[], filename: string } | undefined;
  model_file.value = file_info;
});

socket.on('disconnect', (payload) => {
  console.log("disconnected!", payload);
  connected.value = false;
})

socket.on('model_file', ( file_info: { filename: string, pathlist: string[] } ) => {
  model_file.value = file_info;
});

socket.on('active_fit', ({ fitter_id, options, num_steps }) => {
  active_fit.value = { fitter_id, options, num_steps };
});

socket.on('fit_progress', (event) => {
  fit_progress.value = event;
});

socket.on('add_notification', addNotification);
socket.on('cancel_notification', cancelNotification);

function disconnect() {
  socket.disconnect();
  connected.value = false;
}

function selectOpenFile() {
  if (fileBrowser.value) {
    const settings: FileBrowserSettings = {
      title: "Load Model File",
      callback: async (pathlist, filename) => {
        await socket.asyncEmit("load_problem_file", pathlist, filename);
      },
      chosenfile_in: model_file.value?.filename ?? "",
      show_name_input: false,
      require_name: true,
      show_files: true,
      search_patterns: [".py, .pickle, .json", ".py", ".json", ".pickle"],
    }
    fileBrowser.value.open(settings);
  }
}

function exportResults() {
  if (fileBrowser.value) {
    const settings: FileBrowserSettings = {
      title: "Export Results",
      callback: async (pathlist, filename) => {
        if (filename !== "") {
          pathlist.push(filename);
        }
        socket.asyncEmit("export_results", pathlist).then(() => {
          socket?.syncFS?.();
        })
      },
      show_name_input: true,
      name_input_label: "Subdirectory",
      require_name: false,
      show_files: false,
      chosenfile_in: "",
      search_patterns: [],
    }
    fileBrowser.value.open(settings);
  }
}

async function saveFileAs(ev: Event) {
  if (fileBrowser.value) {
    const { extension } = await socket.asyncEmit("get_serializer") as { extension: string };
    const filename_in = model_file.value?.filename ?? "";
    const new_filename = `${filename_in.replace(/(\.[^\.]+)$/, '')}.${extension}`;
    const settings: FileBrowserSettings = {
      title: "Save Problem",
      callback: async (pathlist, filename) => {
        saveFile(ev, {pathlist, filename});
      },
      show_name_input: true,
      name_input_label: "Filename",
      require_name: true,
      show_files: true,
      chosenfile_in: new_filename,
      search_patterns: [`.${extension}`],
    }
    fileBrowser.value.open(settings);
  }
}

async function saveFile(ev: Event, override?: {pathlist: string[], filename: string}) {
  if (model_file.value === undefined) {
    alert('no file to save');
    return;
  }
  const { filename, pathlist } = override ?? model_file.value;
  console.log('saving:', {pathlist, filename});
  await socket.asyncEmit("save_problem_file", pathlist, filename, false, async({filename, check_overwrite}: {filename: string, check_overwrite: boolean}) => {
    if (check_overwrite !== false) {
      const overwrite = await confirm(`File ${filename} exists: overwrite?`);
      if (overwrite) {
        await socket.asyncEmit("save_problem_file", pathlist, filename, overwrite);
      }
    }
    if (nativefs.value) {
      await socket.syncFS();
    }
  });
}

async function reloadModel() {
  if (model_file.value) {
    const { filename, pathlist } = model_file.value;
    await socket.asyncEmit("load_problem_file", pathlist, filename);
  }
}

async function applyParameters(ev: Event) {
  if (fileBrowser.value) {
    const settings: FileBrowserSettings = {
      title: "Apply Parameters",
      callback: async (pathlist, filename) => {
        await socket.asyncEmit("apply_parameters", pathlist, filename);
      },
      show_name_input: true,
      name_input_label: "Filename",
      require_name: true,
      show_files: true,
      chosenfile_in: "",
      search_patterns: [".par"],
    }
    fileBrowser.value.open(settings);
  }
}

async function saveParameters(ev: Event, override?: {pathlist: string[], filename: string}) {
  if (fileBrowser.value) {
    const settings: FileBrowserSettings = {
      title: "Save Parameters",
      callback: async (pathlist, filename) => {
        await socket.asyncEmit("save_parameters", pathlist, filename, false, async({filename, check_overwrite}: {filename: string, check_overwrite: boolean}) => {
          if (check_overwrite !== false) {
          const overwrite = await confirm(`File ${filename} exists: overwrite?`);
            if (overwrite) {
              await socket.asyncEmit("save_parameters", pathlist, filename, overwrite);
            }
          }
          if (nativefs.value) {
            await socket.syncFS();
          }
        });
      },
      show_name_input: true,
      name_input_label: "Filename",
      require_name: true,
      show_files: true,
      search_patterns: [".par"],
      chosenfile_in: "manual_save.par",
    }
    fileBrowser.value.open(settings);
  }
}

function openFitOptions() {
  fitOptions.value?.open();
}

async function startFit() {
  const active = selected_fitter.value;
  const settings = fitter_settings.value;
  if (active && settings) {
    const fit_args = settings[active];
    await socket.asyncEmit("start_fit_thread", active, fit_args.settings);
  }
}

async function stopFit() {
  await socket.asyncEmit("stop_fit")
}

async function quit() {
  await socket.asyncEmit("shutdown");
}

async function mountLocal() {
  const success = (await socket.mountLocal?.()) ?? false;
  nativefs.value = success;
}

const model_not_loaded = computed(() => model_file.value == null);

menu_items.value = [
  { text: "Load Problem", action: selectOpenFile },
  { text: "Save Parameters", action: saveParameters, disabled: model_not_loaded },
  { text: "Apply Parameters", action: applyParameters, disabled: model_not_loaded },
  { text: "Save Problem", action: saveFile, disabled: model_not_loaded },
  { text: "Save Problem As...", action: saveFileAs, disabled: model_not_loaded },
  { text: "Export Results", action: exportResults, disabled: model_not_loaded },
  { text: "Reload Model", action: reloadModel, disabled: model_not_loaded },
]

onMounted(() => {
  const menuToggleButton = new Button(menuToggle.value as HTMLElement);
});

</script>

<template>
  <div class="h-100 w-100 m-0 d-flex flex-column">
    <nav class="navbar navbar-expand-sm navbar-dark bg-dark" v-if="single_panel === null">
      <div class="container-fluid">
        <div class="navbar-brand">
          <img src="./assets/bumps-icon_256x256x32.png" alt="" height="24" class="d-inline-block align-text-middle">
          {{ name ?? "Bumps" }}
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
                <li><button v-if="can_mount_local" class="btn btn-link dropdown-item" @click="mountLocal">Mount Local Folder</button></li>
                <li v-for="menu_item of menu_items" :key="menu_item.text" :title="menu_item.help">
                  <button class="btn btn-link dropdown-item" @click="menu_item.action" :disabled="menu_item.disabled ?? false">{{ menu_item.text }}</button>
                </li>
                <li>
                  <hr class="dropdown-divider">
                </li>
                <li><button class="btn btn-link dropdown-item" @click="quit">Quit</button></li>
              </ul>
            </li>
            <SessionMenu :socket="socket" />
            <!-- <li class="nav-item dropdown">
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
            </li> -->
            <li class="nav-item dropdown">
              <button class="btn btn-link nav-link dropdown-toggle"  role="button" data-bs-toggle="dropdown"
                aria-expanded="false">
                Layout
              </button>
              <ul class="dropdown-menu">
                <li v-for="layout in LAYOUTS" :key="layout" :class="{layout: true, active: active_layout === layout}">
                  <button class="btn btn-link dropdown-item" :class="{layout: true, active: active_layout === layout}" @click="active_layout = layout">{{ layout }}</button>
                </li>
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
          <div class="flex-grow-1 px-4 m-0">
            <h4 class="m-0">
              <!-- <div class="rounded p-2 bg-primary">Fitting: </div> -->
              <div v-if="active_fit.fitter_id !== undefined" class="badge bg-secondary p-2 align-middle">
                <div class="align-middle pt-2 pb-1 px-1 d-inline-block">
                  <span>Fitting: {{ active_fit.fitter_id }} step {{ fit_progress?.step }} of
                  {{ active_fit?.num_steps }}, chisq={{ fit_progress.chisq }}</span>
                  <div class="progress mt-1" style="height:3px;">
                    <div class="progress-bar" role="progressbar" :aria-valuenow="fit_progress?.step" 
                      aria-valuemin="0" :aria-valuemax="active_fit?.num_steps ?? 100" :style="{width: ((fit_progress.step ?? 0) * 100 / (active_fit.num_steps ?? 1)).toFixed(1) + '%'}"></div>
                  </div>
                </div>
                <button class="btn btn-danger btn-sm" @click="stopFit">Stop</button>
              </div>
              <div v-else class="badge bg-secondary p-1">
                <div class="align-middle p-2 d-inline-block">
                  <span>Fitting: </span>
                </div>
                <button class="btn btn-light btn-sm me-2" @click="openFitOptions">
                  {{ selected_fitter ?? "" }}
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-gear" viewBox="0 0 16 16">
                    <path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"/>
                    <path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z"/>
                  </svg>
                </button>
                <button class="btn btn-success btn-sm" @click="startFit">Start</button>
              </div>
            </h4>
          </div>
          <div class="d-flex">
            <div id="connection_status"
              :class="{'btn': true, 'btn-outline-success': connected, 'btn-outline-danger': !connected}">
              {{(connected) ? 'connected' : 'disconnected'}}</div>

          </div>
        </div>
      </div>
    </nav>
    <div class="flex-grow-1 row overflow-hidden" v-if="active_layout === 'left-right'">
      <div class="col d-flex flex-column mh-100 border-end border-success border-3">
        <PanelTabContainer :panels="panels" :socket="socket" :active_panel="active_panel[0]" @panel_changed="(n: number) => { active_panel[0] = n }"/>
      </div>
      <div class="col d-flex flex-column mh-100">
        <PanelTabContainer :panels="panels" :socket="socket" :active_panel="active_panel[1]" @panel_changed="(n: number) => { active_panel[1] = n }"/>
      </div>
    </div>
    <div class="flex-grow-1 d-flex flex-column" v-if="active_layout === 'top-bottom'">
      <div class="d-flex flex-column flex-grow-1" style="overflow-y:scroll;">
        <PanelTabContainer :panels="panels" :socket="socket" :active_panel="active_panel[0]" @panel_changed="(n: number) => { active_panel[0] = n }"/>
      </div>
      <div class="d-flex flex-column flex-grow-1">
        <PanelTabContainer :panels="panels" :socket="socket" :active_panel="active_panel[1]" @panel_changed="(n: number) => { active_panel[1] = n }"/>
      </div>
    </div>
    <div class="flex-grow-1 row overflow-hidden" v-if="active_layout === 'full'">
      <div class="col d-flex flex-column mh-100">
        <PanelTabContainer :panels="panels" :socket="socket" :active_panel="active_panel[0]" @panel_changed="(n: number) => { active_panel[0] = n }" :hide_tabs="single_panel !== null"/>
      </div>
    </div>

  </div>
  <FitOptions ref="fitOptions" :socket="socket" />
  <FileBrowser ref="fileBrowser" :socket="socket" />
  <ServerShutdown :socket="socket" />
  <ServerStartup :socket="socket" />
  <div class="toast-container position-fixed bottom-0 end-0 p-3" style="z-index: 11">
    <div v-for="notification in notifications" :key="notification.id" class="toast show" role="alert" aria-live="assertive" aria-atomic="true">
      <div class="toast-header">
        <strong class="me-auto">{{notification.title}}</strong>
        <button type="button" class="btn-close" @click="cancelNotification(notification.id)" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
      <div class="toast-body">
        <div v-html="notification.content"></div>
        <div v-if="notification.spinner" class="spinner-border spinner-border-sm text-primary" role="status">
          <span class="visually-hidden">{{ notification.title }} ongoing...</span>
        </div>
      </div>
    </div>
  </div>
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

.dropdown-menu {
  z-index: 2000;
}
</style>
