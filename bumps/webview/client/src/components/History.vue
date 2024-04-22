<script setup lang="ts">
import { ref } from 'vue';
import type { AsyncSocket } from '../asyncSocket';

const title = "History";

const props = defineProps<{
  socket: AsyncSocket,
}>();

type HistoryItem = {
  timestamp: string,
  label: string,
  chisq_str: string,
  keep: boolean,
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

async function remove_history_item(timestamp: string) {
  console.log('remove_history_item', timestamp);
  await props.socket.asyncEmit('remove_history_item', timestamp);
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
      <li v-for="({timestamp, label, chisq_str, keep}, index) of history" :key="index" class="list-group-item">
        <div class="d-flex w-100 justify-content-between">
          <button
            type="button"
            class="btn-close"
            aria-label="Close"
            @click="remove_history_item(timestamp)">
          </button>
          <span class="me-1">{{ label }}</span>
          <span>
            <small class="me-1">{{ chisq_str }}</small>
            <button class="btn btn-secondary btn-sm" @click="reload_history(timestamp)">Load</button>
          </span>
          <div class="form-check">
            <input class="form-check-input" type="checkbox" :value="keep" @click="toggle_keep(timestamp, keep)" :id="`keep-${index}`">
            <label class="form-check-label" :for="`keep-${index}`">
              keep
            </label>
          </div>
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