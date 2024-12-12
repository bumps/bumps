<script setup lang="ts">
import { computed, ref } from "vue";
import { io } from "socket.io-client";
import {
  active_fit,
  active_layout,
  activePanel,
  addNotification,
  cancelNotification,
  connected,
  default_fitter,
  default_fitter_settings,
  file_menu_items,
  fileBrowser,
  FileBrowserSettings,
  fitOptions,
  fitter_settings,
  LAYOUTS,
  model_file,
  notifications,
  selected_fitter,
  socket as socket_ref,
} from "./app_state";
import Gear from "./assets/gear.svg?component";
import "./asyncSocket"; // patch Socket with asyncEmit
import type { AsyncSocket } from "./asyncSocket";
import DropDown from "./components/DropDown.vue";
import FileBrowser from "./components/FileBrowser.vue";
import FitOptions from "./components/FitOptions.vue";
import PanelTabContainer from "./components/PanelTabContainer.vue";
import ServerShutdown from "./components/ServerShutdown.vue";
import ServerStartup from "./components/ServerStartup.vue";
import SessionMenu from "./components/SessionMenu.vue";
import type { Panel } from "./panels";

const props = defineProps<{
  panels: Panel[];
  name?: string;
}>();

const show_menu = ref(false);
// const nativefs = ref(false);

// Create a SocketIO connection, to be passed to child components
// so that they can do their own communications with the host.
const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);

const sio_base_path = urlParams.get("base_path") ?? window.location.pathname;
const sio_server = urlParams.get("server") ?? "";
const single_panel = urlParams.get("single_panel");
if (single_panel !== null) {
  active_layout.value = "full";
  const panel_index = props.panels.findIndex(({ title }) => title.toLowerCase() == single_panel.toLowerCase());
  if (panel_index > -1) {
    activePanel.value[0] = panel_index;
  } else {
    console.error(`Panel ${single_panel} not found`);
  }
}

const socket = io(sio_server, {
  path: `${sio_base_path}socket.io`,
}) as AsyncSocket;
socket_ref.value = socket;

socket.on("connect", async () => {
  console.log(`Connected: Session ID ${socket.id}`);
  connected.value = true;
  const file_info = (await socket.asyncEmit("get_shared_setting", "model_file")) as
    | { pathlist: string[]; filename: string }
    | undefined;
  model_file.value = file_info;
  const current_active_fit = (await socket.asyncEmit("get_shared_setting", "active_fit")) as
    | {
        fitter_id?: string;
        options?: any;
        num_steps?: number;
        chisq?: string;
        step?: number;
        value?: number;
      }
    | undefined;
  if (current_active_fit) {
    active_fit.value = current_active_fit;
  }
});

socket.on("disconnect", (payload) => {
  console.log("Disconnected!", payload);
  connected.value = false;
});

socket.on("model_file", (file_info: { filename: string; pathlist: string[] }) => {
  model_file.value = file_info;
});

socket.on("active_fit", ({ fitter_id, options, num_steps, step, chisq }) => {
  active_fit.value = { fitter_id, options, num_steps, step, chisq };
});

socket.on("add_notification", addNotification);
socket.on("cancel_notification", cancelNotification);

// function disconnect() {
//   socket.disconnect();
//   connected.value = false;
// }

async function selectOpenFile() {
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
    };
    fileBrowser.value.open(settings);
  }
}

async function exportResults() {
  if (fileBrowser.value) {
    const settings: FileBrowserSettings = {
      title: "Export Results",
      callback: async (pathlist, filename) => {
        if (filename !== "") {
          pathlist.push(filename);
        }
        await socket.asyncEmit("export_results", pathlist);
      },
      show_name_input: true,
      name_input_label: "Subdirectory",
      require_name: false,
      show_files: false,
      chosenfile_in: "",
      search_patterns: [],
    };
    fileBrowser.value.open(settings);
  }
}

async function saveFileAs(ev: Event) {
  if (fileBrowser.value) {
    const { extension } = (await socket.asyncEmit("get_serializer")) as { extension: string };
    const filename_in = model_file.value?.filename ?? "";
    const new_filename = `${filename_in.replace(/(\.[^\.]+)$/, "")}.${extension}`;
    const settings: FileBrowserSettings = {
      title: "Save Problem",
      callback: async (pathlist, filename) => {
        saveFile(ev, { pathlist, filename });
      },
      show_name_input: true,
      name_input_label: "Filename",
      require_name: true,
      show_files: true,
      chosenfile_in: new_filename,
      search_patterns: [`.${extension}`],
    };
    fileBrowser.value.open(settings);
  }
}

async function saveFile(ev: Event, override?: { pathlist: string[]; filename: string }) {
  if (model_file.value === undefined) {
    alert("no file to save");
    return;
  }
  const { filename, pathlist } = override ?? model_file.value;
  console.debug(`Saving: ${pathlist.join("/")}/${filename}`);
  await socket.asyncEmit(
    "save_problem_file",
    pathlist,
    filename,
    false,
    async ({ filename, check_overwrite }: { filename: string; check_overwrite: boolean }) => {
      if (check_overwrite !== false) {
        const overwrite = await confirm(`File ${filename} exists: overwrite?`);
        if (overwrite) {
          await socket.asyncEmit("save_problem_file", pathlist, filename, overwrite);
        }
      }
    }
  );
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
    };
    fileBrowser.value.open(settings);
  }
}

async function saveParameters(ev: Event, override?: { pathlist: string[]; filename: string }) {
  if (fileBrowser.value) {
    const settings: FileBrowserSettings = {
      title: "Save Parameters",
      callback: async (pathlist, filename) => {
        await socket.asyncEmit(
          "save_parameters",
          pathlist,
          filename,
          false,
          async ({ filename, check_overwrite }: { filename: string; check_overwrite: boolean }) => {
            if (check_overwrite !== false) {
              const overwrite = await confirm(`File ${filename} exists: overwrite?`);
              if (overwrite) {
                await socket.asyncEmit("save_parameters", pathlist, filename, overwrite);
              }
            }
          }
        );
      },
      show_name_input: true,
      name_input_label: "Filename",
      require_name: true,
      show_files: true,
      search_patterns: [".par"],
      chosenfile_in: "manual_save.par",
    };
    fileBrowser.value.open(settings);
  }
}

function openFitOptions() {
  fitOptions.value?.open();
}

async function startFit() {
  const active = selected_fitter.value ?? default_fitter;
  const settings = fitter_settings.value ?? default_fitter_settings.value;
  if (active && settings) {
    const fit_args = settings[active];
    await socket.asyncEmit("start_fit_thread", active, fit_args.settings);
  }
}

async function stopFit() {
  await socket.asyncEmit("stop_fit");
}

async function quit() {
  await socket.asyncEmit("shutdown");
}

const model_not_loaded = computed(() => model_file.value == null);

file_menu_items.value = [
  { text: "Load Problem", action: selectOpenFile },
  { text: "Save Parameters", action: saveParameters, disabled: model_not_loaded },
  { text: "Apply Parameters", action: applyParameters, disabled: model_not_loaded },
  { text: "Save Problem", action: saveFile, disabled: model_not_loaded },
  { text: "Save Problem As...", action: saveFileAs, disabled: model_not_loaded },
  { text: "Export Results", action: exportResults, disabled: model_not_loaded },
  { text: "Reload Model", action: reloadModel, disabled: model_not_loaded },
  { text: "---" },
  { text: "Quit", action: quit },
];
</script>

<template>
  <div class="h-100 w-100 m-0 d-flex flex-column">
    <nav v-if="single_panel === null" class="navbar navbar-expand-sm navbar-dark bg-dark">
      <div class="container-fluid">
        <div class="navbar-brand">
          <img src="./assets/bumps-icon_256x256x32.png" alt="" height="24" class="d-inline-block align-text-middle" />
          {{ name ?? "Bumps" }}
        </div>
        <button
          class="navbar-toggler"
          type="button"
          aria-controls="navbarSupportedContent"
          aria-expanded="false"
          aria-label="Toggle navigation"
          @click="show_menu = !show_menu"
        >
          <span class="navbar-toggler-icon"></span>
        </button>
        <div id="navbarSupportedContent" class="collapse navbar-collapse" :class="{ show: show_menu }">
          <ul class="navbar-nav me-auto mb-2 mb-lg-0">
            <DropDown v-slot="{ hide }" title="File">
              <li v-for="menu_item of file_menu_items" :key="menu_item.text" :title="menu_item.help">
                <hr v-if="menu_item.text === '---'" class="dropdown-divider" />
                <button
                  v-else
                  class="btn btn-link dropdown-item"
                  :disabled="menu_item.disabled?.value ?? false"
                  @click="
                    menu_item.action?.();
                    hide();
                  "
                >
                  {{ menu_item.text }}
                </button>
              </li>
            </DropDown>
            <SessionMenu :socket="socket" />
            <DropDown v-slot="{ hide }" title="Layout">
              <li v-for="layout in LAYOUTS" :key="layout" :class="{ layout: true, active: active_layout === layout }">
                <button
                  class="btn btn-link dropdown-item"
                  :class="{ layout: true, active: active_layout === layout }"
                  @click="
                    active_layout = layout;
                    hide();
                  "
                >
                  {{ layout }}
                </button>
              </li>
            </DropDown>
          </ul>
          <div class="flex-grow-1 px-4 m-0">
            <h4 class="m-0">
              <!-- <div class="rounded p-2 bg-primary">Fitting: </div> -->
              <div v-if="active_fit.fitter_id !== undefined" class="badge bg-secondary p-2 align-middle">
                <div class="align-middle pt-2 pb-1 px-1 d-inline-block">
                  <span
                    >Fitting: {{ active_fit.fitter_id }} step {{ active_fit?.step }} of {{ active_fit?.num_steps }},
                    chisq={{ active_fit.chisq }}</span
                  >
                  <div class="progress mt-1" style="height: 3px">
                    <div
                      class="progress-bar"
                      role="progressbar"
                      :aria-valuenow="active_fit?.step"
                      aria-valuemin="0"
                      :aria-valuemax="active_fit?.num_steps ?? 100"
                      :style="{
                        width: (((active_fit.step ?? 0) * 100) / (active_fit.num_steps ?? 1)).toFixed(1) + '%',
                      }"
                    ></div>
                  </div>
                </div>
                <button class="btn btn-danger btn-sm" @click="stopFit">Stop</button>
              </div>
              <div v-else class="badge bg-secondary p-1">
                <div class="align-middle p-2 d-inline-block">
                  <span>Fitting: </span>
                </div>
                <button class="btn btn-light btn-sm me-2" @click="openFitOptions">
                  {{ selected_fitter ?? default_fitter }}
                  <Gear />
                </button>
                <button class="btn btn-success btn-sm" @click="startFit">Start</button>
              </div>
            </h4>
          </div>
          <div class="d-flex">
            <div
              id="connection_status"
              :class="{ btn: true, 'btn-outline-success': connected, 'btn-outline-danger': !connected }"
            >
              {{ connected ? "connected" : "disconnected" }}
            </div>
          </div>
        </div>
      </div>
    </nav>
    <div v-if="active_layout === 'left-right'" class="flex-grow-1 row overflow-hidden">
      <div class="col d-flex flex-column mh-100 border-end border-success border-3">
        <PanelTabContainer
          :panels="panels"
          :socket="socket"
          :active-panel="activePanel[0]"
          @panel_changed="
            (n: number) => {
              activePanel[0] = n;
            }
          "
        />
      </div>
      <div class="col d-flex flex-column mh-100">
        <PanelTabContainer
          :panels="panels"
          :socket="socket"
          :active-panel="activePanel[1]"
          @panel_changed="
            (n: number) => {
              activePanel[1] = n;
            }
          "
        />
      </div>
    </div>
    <div v-if="active_layout === 'top-bottom'" class="flex-grow-1 d-flex flex-column">
      <div class="d-flex flex-column flex-grow-1" style="overflow-y: scroll">
        <PanelTabContainer
          :panels="panels"
          :socket="socket"
          :active-panel="activePanel[0]"
          @panel_changed="
            (n: number) => {
              activePanel[0] = n;
            }
          "
        />
      </div>
      <div class="d-flex flex-column flex-grow-1">
        <PanelTabContainer
          :panels="panels"
          :socket="socket"
          :active-panel="activePanel[1]"
          @panel_changed="
            (n: number) => {
              activePanel[1] = n;
            }
          "
        />
      </div>
    </div>
    <div v-if="active_layout === 'full'" class="flex-grow-1 row overflow-hidden">
      <div class="col d-flex flex-column mh-100">
        <PanelTabContainer
          :panels="panels"
          :socket="socket"
          :active-panel="activePanel[0]"
          :hide-tabs="single_panel !== null"
          @panel_changed="
            (n: number) => {
              activePanel[0] = n;
            }
          "
        />
      </div>
    </div>
  </div>
  <FitOptions ref="fitOptions" :socket="socket" />
  <FileBrowser ref="fileBrowser" :socket="socket" />
  <ServerShutdown :socket="socket" />
  <ServerStartup :socket="socket" />
  <div class="toast-container position-fixed bottom-0 end-0 p-3" style="z-index: 11">
    <div
      v-for="notification in notifications"
      :key="notification.id"
      class="toast show"
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
    >
      <div class="toast-header">
        <strong class="me-auto">{{ notification.title }}</strong>
        <button
          type="button"
          class="btn-close"
          data-bs-dismiss="toast"
          aria-label="Close"
          @click="cancelNotification(notification.id)"
        ></button>
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
  width: 100%;
  height: 100%;
  overflow-x: hidden;
}

div#connection_status {
  pointer-events: none;
}

.dropdown-menu {
  z-index: 2000;
}
</style>
