<script setup lang="ts">
import { onMounted, ref } from "vue";
import { addNotification, autosave_history, autosave_history_length } from "../app_state";
import type { AsyncSocket } from "../asyncSocket";

// const title = "History";

const props = defineProps<{
  socket: AsyncSocket;
}>();

type HistoryItem = {
  timestamp: string;
  label: string;
  chisq_str: string;
  keep: boolean;
  has_population: boolean;
  has_uncertainty: boolean;
};

const history = ref<HistoryItem[]>([]);
const history_label = ref("");
const current_chisq_str = ref("");
const warning_seen = ref(false);

props.socket.on("updated_history", async () => {
  const new_history = (await props.socket.asyncEmit("get_history")) as { problem_history: HistoryItem[] };
  history.value = new_history.problem_history;
});
props.socket.asyncEmit("get_history", (new_history: { problem_history: HistoryItem[] }) => {
  history.value = new_history.problem_history;
});

props.socket.on("updated_parameters", () => {
  props.socket.asyncEmit("get_chisq", (chisq_str: string) => {
    current_chisq_str.value = `chisq: ${chisq_str}`;
  });
});
props.socket.asyncEmit("get_chisq", (chisq_str: string) => {
  current_chisq_str.value = `chisq: ${chisq_str}`;
});

props.socket.on("autosave_history", (autosave: boolean) => {
  autosave_history.value = autosave;
});

props.socket.on("autosave_history_length", (length: number) => {
  autosave_history_length.value = length;
});

async function remove_history_item(timestamp: string, keep: boolean) {
  console.log(`remove_history_item: ${timestamp}`);
  if (keep) {
    addNotification({
      title: "Forbidden",
      content: "Cannot remove history item marked to keep",
      timeout: 5000,
    });
    return;
  } else {
    await props.socket.asyncEmit("remove_history_item", timestamp);
  }
}

async function manual_save() {
  await props.socket.asyncEmit("save_to_history", history_label.value || "manual save");
  history_label.value = "";
}

async function reload_history(timestamp: string) {
  const confirmation = warning_seen.value || confirm("Reloading overwrites current state: continue?");
  if (confirmation) {
    await props.socket.asyncEmit("reload_history_item", timestamp);
    warning_seen.value = true;
  }
}

async function toggle_keep(timestamp: string, current_keep: boolean) {
  await props.socket.asyncEmit("set_keep_history", timestamp, !current_keep);
}

async function update_label(timestamp: string, new_label: string) {
  console.log(`update_label(${timestamp}, ${new_label})`);
  await props.socket.asyncEmit("update_history_label", timestamp, new_label);
}

async function toggle_autosave_history() {
  await props.socket.asyncEmit("set_shared_setting", "autosave_history", !autosave_history.value);
}

async function set_autosave_history_length(value_str: string) {
  const new_value = parseInt(value_str, 10);
  if (!isNaN(new_value) && new_value > 0) {
    await props.socket.asyncEmit("set_shared_setting", "autosave_history_length", new_value);
  }
}

onMounted(async () => {
  autosave_history.value = await props.socket.asyncEmit("get_shared_setting", "autosave_history");
  autosave_history_length.value = await props.socket.asyncEmit("get_shared_setting", "autosave_history_length");
});
</script>

<template>
  <div class="history container d-flex flex-column flex-grow-1">
    <div class="row p-1 align-items-center">
      <div class="col-auto">
        <button type="button" class="btn btn-primary" @click="manual_save">Save current state</button>
        <span>{{ current_chisq_str }}</span>
      </div>
      <div class="col text-end">
        <input
          id="auto_save"
          class="form-check-input"
          type="checkbox"
          :checked="autosave_history"
          @click="toggle_autosave_history"
        />
        <label class="form-check-label" for="auto_save" title="Append problem state to history on load and at fit end"
          >Auto append</label
        >
      </div>
      <div class="col-auto">
        <input
          id="auto_save_length"
          class="form-control-sm"
          type="number"
          :value="autosave_history_length"
          min="1"
          step="1"
          @change="set_autosave_history_length(($event.target as HTMLInputElement).value)"
        />
        <label class="col-form-label" for="auto_save_length">history length</label>
      </div>
    </div>
    <div class="px-2 flex-grow-1 flex-shrink-1 overflow-auto">
      <table class="table table-sm flex-grow-0">
        <thead class="border-bottom py-1 sticky-top text-white bg-secondary">
          <tr class="text-center">
            <th scope="col"></th>
            <th scope="col">Label</th>
            <th scope="col">Chisq</th>
            <th scope="col"></th>
            <th scope="col">Keep</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="{ timestamp, label, chisq_str, keep, has_population, has_uncertainty } of history"
            :key="timestamp"
            class="py-1 align-middle"
          >
            <td class="align-items-center">
              <button
                type="button"
                class="btn-close"
                aria-label="Close"
                @click="remove_history_item(timestamp, keep)"
              ></button>
            </td>
            <td
              class="px-2"
              contenteditable="true"
              spellcheck="false"
              plaintext-only
              :title="timestamp"
              @blur="update_label(timestamp, ($event.target as HTMLTextAreaElement).innerText)"
            >
              {{ label }}
            </td>
            <td>
              <small>{{ chisq_str }}</small>
            </td>
            <td class="text-nowrap">
              <button class="btn btn-secondary btn-sm mx-1 text-nowrap" @click="reload_history(timestamp)">Load</button>
              <span v-show="has_population" class="badge bg-success" title="has population">P</span>
              <span v-show="has_uncertainty" class="badge bg-warning" title="has uncertainty">U</span>
            </td>
            <td class="text-center">
              <input
                :id="`keep-${label}`"
                class="form-check-input"
                type="checkbox"
                :checked="keep"
                @click="toggle_keep(timestamp, keep)"
              />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
input#auto_save_length {
  width: 4em;
}

.btn-close {
  cursor: pointer;
}
</style>
