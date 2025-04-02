<script setup lang="ts">
import DropDown from "./DropDown.vue";
import { fileBrowser, shared_state } from "../app_state";
import type { FileBrowserSettings } from "../app_state";
import type { AsyncSocket } from "../asyncSocket";

// const title = "Session";
const props = defineProps<{
  socket: AsyncSocket;
}>();

async function toggle_autosave() {
  // if turning on autosave, but session_file is not set, then prompt for a file
  if (!shared_state.autosave_session && !shared_state.session_output_file) {
    // open the file browser to select output file, and then set autosave
    await setOutputFile(true, false);
    // closeMenu();
    return;
  }
  await props.socket.asyncEmit("set_shared_setting", "autosave_session", !shared_state.autosave_session);
  // setTimeout(closeMenu, 1000);
}

async function set_interval(new_interval: number) {
  await props.socket.asyncEmit("set_shared_setting", "autosave_session_interval", new_interval);
  closeMenu();
}

async function saveSession() {
  if (shared_state.session_output_file) {
    await props.socket.asyncEmit("save_session");
  } else {
    await setOutputFile(false, true);
  }
}

function closeMenu() {
  const menu = document.getElementById("session-dropdown-menu");
  if (menu) {
    menu.classList.remove("show");
  }
}

async function readSession(readOnly: boolean) {
  if (shared_state.active_fit?.fitter_id) {
    const confirmation = confirm("Loading session will stop current fit. Continue?");
    if (!confirmation) {
      return;
    }
  }
  if (fileBrowser.value) {
    const settings: FileBrowserSettings = {
      title: "Read Session",
      callback: async (pathlist, filename) => {
        props.socket.asyncEmit("load_session", pathlist, filename, readOnly);
      },
      show_name_input: true,
      name_input_label: "Session Filename",
      require_name: true,
      show_files: true,
      search_patterns: [".h5"],
      chosenfile_in: "",
      pathlist_in:
        shared_state.session_output_file?.pathlist ? [...shared_state.session_output_file.pathlist] : undefined,
    };
    fileBrowser.value.open(settings);
  }
}

async function setOutputFile(enable_autosave = true, immediate_save = false) {
  if (fileBrowser.value) {
    const settings: FileBrowserSettings = {
      title: "Set Output File",
      callback: async (pathlist, filename) => {
        let input = filename;
        if (input.includes("/")) {
          alert("FileBrowser: '/' in filename not yet supported");
          return;
        }
        if (!input.endsWith(".h5") && !input.endsWith(".hdf5")) {
          filename = `${filename}.session.h5`;
        }
        await props.socket.asyncEmit("set_shared_setting", "session_output_file", { filename, pathlist });
        if (enable_autosave) {
          await props.socket.asyncEmit("set_shared_setting", "autosave_session", true);
        }
        if (immediate_save) {
          await props.socket.asyncEmit("save_session");
        }
      },
      show_name_input: true,
      name_input_label: "Output Filename",
      require_name: true,
      show_files: true,
      search_patterns: [".h5"],
      chosenfile_in: shared_state.session_output_file?.filename,
      pathlist_in:
        shared_state.session_output_file?.pathlist ? [...shared_state.session_output_file.pathlist] : undefined,
    };
    fileBrowser.value.open(settings);
  }
}

// async function unsetOutputFile() {
//   await props.socket.asyncEmit("set_shared_setting", "session_output_file", null);
// }
</script>

<template>
  <DropDown v-slot="{ hide }" title="Session">
    <li class="dropdown-item">
      <div class="form-check form-switch">
        <label class="form-check-label" for="autoSaveSessionCheckbox">Autosave</label>
        <input
          id="autoSaveSessionCheckbox"
          class="form-check-input"
          type="checkbox"
          role="switch"
          :checked="shared_state.autosave_session"
          @change="toggle_autosave"
        />
      </div>
    </li>
    <li>
      <span v-if="shared_state.session_output_file" class="">
        <span class="dropdown-item-text">{{ shared_state.session_output_file.filename }}</span>
      </span>
    </li>
    <li class="dropdown-item">
      <div class="row">
        <label for="autosaveIntervalInput" class="col col-form-label col-form-label-sm">Interval (s)</label>
        <div class="col-auto">
          <input
            id="autosaveIntervalInput"
            type="number"
            class="form-control form-control"
            :value="shared_state.autosave_session_interval"
            @change="set_interval(($event.target as HTMLInputElement).valueAsNumber)"
          />
        </div>
      </div>
    </li>
    <li>
      <hr class="dropdown-divider" />
    </li>
    <li>
      <button
        class="btn btn-link dropdown-item"
        :class="{ disabled: shared_state.active_fit?.fitter_id }"
        @click="
          readSession(false);
          hide();
        "
      >
        Open Session
      </button>
    </li>
    <li>
      <button
        class="btn btn-link dropdown-item"
        @click="
          saveSession();
          hide();
        "
      >
        Save Session
      </button>
    </li>
    <li>
      <button
        class="btn btn-link dropdown-item"
        @click="
          setOutputFile(false, true);
          hide();
        "
      >
        Save Session As
      </button>
    </li>
  </DropDown>
</template>
