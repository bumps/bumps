<script setup lang="ts">
import { ref } from 'vue';
import type { AsyncSocket } from '../asyncSocket';
import { add_notification } from '../app_state';

const title = "History";

const props = defineProps<{
  socket: AsyncSocket,
}>();

type HistoryItem = {
  timestamp: string,
  label: string,
  chisq_str: string,
  keep: boolean,
  has_population: boolean,
  has_uncertainty: boolean,
};

const history = ref<HistoryItem[]>([]);
const history_label = ref('');

props.socket.on('history_update', async () => {
  console.log('history_update');
  const new_history = await props.socket.asyncEmit('get_history') as { problem_history: HistoryItem[] };
  console.log(new_history);
  history.value = new_history.problem_history;
  // props.socket.asyncEmit('get_history', (new_history: HistoryItem[]) => { history.value = new_history })
});
props.socket.asyncEmit('get_history', (new_history: { problem_history: HistoryItem[]}) => { history.value = new_history.problem_history });

async function remove_history_item(timestamp: string, keep: boolean) {
  console.log('remove_history_item', timestamp);
  if (keep) {
    add_notification({
      "title": "Forbidden", 
      "content": "Cannot remove history item marked to keep",
      "timeout": 4000 });
    return;
  } else {
    await props.socket.asyncEmit('remove_history_item', timestamp);
  }
}

async function manual_save() {
  await props.socket.asyncEmit('save_to_history', history_label.value || 'manual save');
  history_label.value = '';
}

async function reload_history(timestamp: string) {
  const confirmation = confirm('Reloading overwrites current state: continue?');
  console.log('reload_history', timestamp, confirmation);
  if (confirmation) {
    await props.socket.asyncEmit('reload_history_item', timestamp);
  }
}

async function toggle_keep(timestamp: string, current_keep: boolean) {
  console.log('toggle_keep', timestamp);
  await props.socket.asyncEmit('set_keep_history', timestamp, !current_keep);
}

async function update_label(timestamp: string, new_label: string) {
  console.log('update_label', timestamp, new_label);
  await props.socket.asyncEmit('update_history_label', timestamp, new_label);
}

</script>
        
<template>
  <div class="history">
    <div class="container-fluid">
      <div class="row p-2">
        <div class="col-auto">
          <button
            type="button"
            class="btn btn-primary"
            @click="manual_save">
            Save current problem state
          </button>
        </div>
        <div class="col">
          <input type="text" class="form-control" id="history_label" v-model="history_label" placeholder="save label">
        </div>
      </div>
    </div>
    <h2 class="mx-auto">History</h2>
    <ul class="list-group">
      <li 
        v-for="({timestamp, label, chisq_str, keep, has_population, has_uncertainty}, index) of history"
        :key="timestamp" class="list-group-item"
      >
        <div class="d-flex w-100 justify-content-between">
          <button
            type="button"
            class="btn-close"
            aria-label="Close"
            @click="remove_history_item(timestamp, keep)">
          </button>
          <span class="px-2" contenteditable="true" plaintext-only @blur="update_label(timestamp, $event.target.innerText)">{{ label }}</span>
          <span>
            <small class="me-1">{{ chisq_str }}</small>
            <button class="btn btn-secondary btn-sm mx-1" @click="reload_history(timestamp)">
              Load
              <span v-if="has_population" class="badge bg-primary" title="has population">P</span>
              <span v-if="has_uncertainty" class="badge bg-warning" title="has uncertainty">U</span>
            </button>
            <span class="form-check form-check-inline">
              <input class="form-check-input" type="checkbox" :checked="keep" @click="toggle_keep(timestamp, keep)" :id="`keep-${index}`">
              <label class="form-check-label" :for="`keep-${index}`">
                keep
              </label>
            </span>
          </span>
        </div>
      </li>
    </ul>
  </div>
</template>
    
<style scoped>
.history {
  padding: 1em;
}
.btn-close {
  cursor: pointer;
}
</style>