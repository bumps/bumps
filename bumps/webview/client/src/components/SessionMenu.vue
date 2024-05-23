<script setup lang="ts">
import { onMounted, ref } from 'vue';
import type { AsyncSocket } from '../asyncSocket';
import { fileBrowser, fileBrowserSettings, active_fit } from '../app_state';

const title = "Session";
const props = defineProps<{
  socket: AsyncSocket,
}>();
const session_pathlist = ref<string[]>([]);
const session_output_file = ref<string>('');
const autosave_session = ref(false);

function handle_file_message(payload: string) {
  console.log("session_output_file", payload);
  session_output_file.value = payload;
}
function handle_pathlist_message(payload: string[]) {
  session_pathlist.value = payload;
}
function handle_autosave_message(payload: boolean) {
  console.log("autosave_session", payload);
  autosave_session.value = payload;
}

props.socket.asyncEmit('get_shared_setting', 'session_output_file', handle_file_message);
props.socket.asyncEmit('get_shared_setting', 'session_pathlist', handle_pathlist_message);
props.socket.asyncEmit('get_shared_setting', 'autosave_session', handle_autosave_message);
props.socket.on('session_output_file', handle_file_message);
props.socket.on('session_pathlist', handle_pathlist_message);
props.socket.on('autosave_session', handle_autosave_message);

async function toggle_autosave() {
  // if turning on autosave, but session_file is not set, then prompt for a file
  if (!autosave_session.value && !session_output_file.value) {
    // open the file browser to select output file, and then set autosave
    await setOutputFile(true, false);
    // closeMenu();
    return;
  }
  await props.socket.asyncEmit('set_shared_setting', 'autosave_session', !autosave_session.value);
  // setTimeout(closeMenu, 1000); 
}

async function saveSession() {
  if (session_output_file.value) {
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
  if (active_fit.value && active_fit.value?.fitter_id) {
    const confirmation = confirm("Loading session will stop current fit. Continue?");
    if (!confirmation) {
      return;
    }
  }
  if (fileBrowser.value) {
    const settings = fileBrowserSettings.value;
    settings.title = "Read Session"
    settings.callback = (pathlist, filename) => {
      props.socket.asyncEmit("load_session", pathlist, filename, readOnly);
    }
    settings.show_name_input = true;
    settings.name_input_label = "Session Filename";
    settings.require_name = true;
    settings.show_files = true;
    settings.search_patterns = [".h5"];
    settings.chosenfile_in = "";
    settings.pathlist_in = session_pathlist.value;
    fileBrowser.value.open();
  }
}

async function saveSessionCopy() {
  if (fileBrowser.value) {
    const settings = fileBrowserSettings.value;
    settings.title = "Save Session (Copy)"
    settings.callback = async (pathlist, filename) => {
      await props.socket.asyncEmit("save_session_copy", pathlist, filename);
      await props.socket.syncFS();
    }
    settings.show_name_input = true;
    settings.name_input_label = "Filename";
    settings.require_name = true;
    settings.show_files = true;
    settings.search_patterns = [".h5"];
    settings.chosenfile_in = "";
    if (session_pathlist.value?.length > 0) {
      settings.pathlist_in = session_pathlist.value;
    }
    fileBrowser.value.open();
  }
}

async function setOutputFile(enable_autosave = true, immediate_save = false) {
  if (fileBrowser.value) {
    const settings = fileBrowserSettings.value;
    settings.title = "Set Output File"
    settings.callback = async (pathlist, filename) => {
      await props.socket.asyncEmit("set_shared_setting", "session_output_file", filename);
      await props.socket.asyncEmit("set_shared_setting", "session_pathlist", pathlist);
      if (enable_autosave) {
        await props.socket.asyncEmit("set_shared_setting", "autosave_session", true);
      }
      if (immediate_save) {
        await props.socket.asyncEmit("save_session", pathlist, filename);
        await props.socket.syncFS();
      }
    }
    settings.show_name_input = true;
    settings.name_input_label = "Output Filename";
    settings.require_name = true;
    settings.show_files = true;
    settings.search_patterns = [".h5"];
    settings.chosenfile_in = session_output_file.value;
    if (session_pathlist.value?.length > 0) {
      settings.pathlist_in = session_pathlist.value;
    }
    fileBrowser.value.open();
  }

}


async function unsetOutputFile() {
  await props.socket.asyncEmit('set_shared_setting', 'session_output_file', null);
}

</script>

<template>
  <li class="nav-item dropdown">
    <button class="btn btn-link nav-link dropdown-toggle" role="button" data-bs-toggle="dropdown"
      aria-expanded="false">
      Session
      <input class="form-check-input" type="checkbox" :checked="autosave_session" readonly >
    </button>
    <ul class="dropdown-menu" id="session-dropdown-menu">
      <li class="dropdown-item">
        <div class="form-check form-switch">
          <input class="form-check-input" type="checkbox" role="switch" id="autoSaveSessionCheckbox"
            :checked="autosave_session" @change="toggle_autosave"
            >
          <label class="form-check-label" for="autoSaveSessionCheckbox">Autosave</label>
        </div>
      </li>
      <li>
        <span class="" v-if="session_output_file">
          <span class="dropdown-item-text">{{ session_output_file }}</span>
        </span>
      </li>
      <li>
        <hr class="dropdown-divider">
      </li>
      <li>
        <button class="btn btn-link dropdown-item" :class="{'disabled': active_fit?.fitter_id }" @click="readSession(false)">Open Session</button>
      </li>
      <li>
        <button class="btn btn-link dropdown-item" @click="saveSession">Save Session</button>
      </li>
      <li>
        <button class="btn btn-link dropdown-item" @click="setOutputFile(false, true)">Save Session As</button>
      </li>
    </ul>
  </li>
</template>