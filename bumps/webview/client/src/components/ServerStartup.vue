<script setup lang="ts">
import { ref } from "vue";
import { connected } from "../app_state.ts";
import type { AsyncSocket } from "../asyncSocket.ts";

const props = defineProps<{ socket: AsyncSocket }>();

const loading_dialog = ref<HTMLDivElement>();
// const connected = ref(false);
const status_string = ref<string>();
const percent = ref<number>();

interface ServerStartupStatus {
  status: string;
  percent: number; // 0-100
}

props.socket.on("server_startup_status", (status: ServerStartupStatus) => {
  status_string.value = status.status;
  percent.value = status.percent;
});

console.log("Connected handler registered.");
</script>

<template>
  <dialog ref="loading_dialog" :open="!connected">
    <div
      id="serverStartupModal"
      class="modal show"
      tabindex="-1"
      aria-labelledby="serverStartupLabel"
      :aria-hidden="connected"
    >
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-header">
            <h5 id="serverStartupLabel" class="modal-title">
              Server Connecting...
              <div v-if="percent === undefined" class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Connecting...</span>
              </div>
            </h5>
          </div>
          <div v-if="status_string !== undefined" class="modal-body">
            <div class="progress">
              <div
                class="progress-bar"
                role="progressbar"
                :style="{ width: percent + '%' }"
                :aria-valuenow="percent"
                aria-valuemin="0"
                aria-valuemax="100"
              ></div>
            </div>
            <p>{{ status_string }}</p>
          </div>
        </div>
      </div>
    </div>
  </dialog>
</template>

<style scoped>
div.modal {
  display: block;
}
</style>
