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
  <div class="history container d-flex flex-column flex-grow-1">
    <div class="row p-1 align-items-center">
      <div class="col-auto">
        <button
          type="button"
          class="btn btn-primary"
          @click="manual_save">
          Save current state
        </button>
      </div>
      <div class="col text-end">
        <input class="form-check-input" type="checkbox" id="auto_save" checked>
        <label class="form-check-label" for="auto_save">Auto save</label>
      </div>
      <div class="col-auto">
        <input class="form-control-sm" type="number" id="auto_save_length" value="10" min="1" step="1">
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
            <tr class="py-1 align-middle" v-for="({timestamp, label, chisq_str, keep, has_population, has_uncertainty}, index) of history" :key="timestamp">
              <td class="align-items-center">
                <button
                  type="button"
                  class="btn-close"
                  aria-label="Close"
                  @click="remove_history_item(timestamp, keep)">
                </button>
              </td>
              <td class="px-2" contenteditable="true" plaintext-only @blur="update_label(timestamp, $event.target.innerText)">{{ label }}</td>
              <td>
                <small>{{ chisq_str }}</small>
              </td>
              <td class="text-nowrap">
                <button class="btn btn-secondary btn-sm mx-1 text-nowrap" @click="reload_history(timestamp)">
                  Load
                </button>
                <span v-show="has_population" class="badge bg-success" title="has population">P</span>
                <span v-show="has_uncertainty" class="badge bg-warning" title="has uncertainty">U</span>
              </td>
              <td class="text-center">
                  <input class="form-check-input" type="checkbox" :checked="keep" @click="toggle_keep(timestamp, keep)">
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