<script setup lang="ts">
import { computed, onMounted, ref, shallowRef } from "vue";
import { Modal } from "bootstrap/dist/js/bootstrap.esm.js";
import type { AsyncSocket } from "../asyncSocket.ts";

const props = defineProps<{ socket: AsyncSocket }>();

const dialog = ref<HTMLDivElement>();
const isOpen = ref(false);
const closeCancelled = ref(false);
const shutdownTimer = ref(0);
const CLOSE_DELAY = 2000; // try to auto-close window after 2 seconds.

let modal: Modal;

onMounted(() => {
  modal = new Modal(dialog.value, { backdrop: "static", keyboard: false });
});

props.socket.on("server_shutting_down", () => {
  modal?.show();
  shutdownTimer.value = setTimeout(() => {
    window.close();
  }, CLOSE_DELAY);
});

function cancelClose() {
  clearTimeout(shutdownTimer.value);
  modal?.hide();
}
</script>

<template>
  <div
    id="serverShutdownModal"
    ref="dialog"
    class="modal fade"
    tabindex="-1"
    aria-labelledby="serverShutdownLabel"
    :aria-hidden="isOpen"
  >
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 id="serverShutdownLabel" class="modal-title">Server Disconnected</h5>
          <button type="button" class="btn-close" aria-label="dismiss dialog" @click="cancelClose"></button>
        </div>
        <div class="modal-body">
          <h3>This client window can be closed</h3>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-primary" @click="cancelClose">Dismiss</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped></style>
