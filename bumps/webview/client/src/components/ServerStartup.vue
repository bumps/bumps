<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { Modal } from 'bootstrap/dist/js/bootstrap.esm.js';
import type { AsyncSocket } from '../asyncSocket.ts';

const props = defineProps<{socket:AsyncSocket}>();

const loading_dialog = ref<HTMLDivElement>();
const connected = ref(false);
const status_string = ref<string>();
const percent = ref<number>();

let modal: Modal;

onMounted(() => {
  modal = new Modal(loading_dialog.value, { backdrop: 'static', keyboard: false });
  loading_dialog.value?.addEventListener("shown.bs.modal", (e) => {
    if (connected.value) {
      modal.hide();
    }
  });
  modal.show();
});

props.socket.on('connect', () => {
  connected.value = true;
  console.log('connected!');
  modal?.hide();
});

interface ServerStartupStatus {
  status: string;
  percent: number; // 0-100
}

props.socket.on('server_startup_status', (status: ServerStartupStatus) => {
  status_string.value = status.status;
  percent.value = status.percent;
});

console.log('connected handler registered.');

</script>

<template>
  <div ref="loading_dialog" class="modal" id="serverStartupModal" tabindex="-1" aria-labelledby="serverStartupLabel"
    :aria-hidden="connected">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="serverStartupLabel">
            Server Connecting...
            <div class="spinner-border text-primary" role="status" v-if="percent === undefined">
              <span class="visually-hidden">Connecting...</span>
            </div>
          </h5>
        </div>
        <div class="modal-body" v-if="status_string !== undefined">
          <div class="progress">
            <div class="progress-bar" role="progressbar" :style="{ width: percent + '%' }" :aria-valuenow="percent"
              aria-valuemin="0" aria-valuemax="100"></div>
          </div>
          <p>{{ status_string }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
</style>