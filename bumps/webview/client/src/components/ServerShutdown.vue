<script setup lang="ts">
import { ref } from 'vue';
import type { AsyncSocket } from '../asyncSocket.ts';

const props = defineProps<{socket:AsyncSocket}>();

const dialog = ref<HTMLDialogElement>();
const isOpen = ref(false);
const closeCancelled = ref(false);
const shutdownTimer = ref<ReturnType<typeof setTimeout>>();
const CLOSE_DELAY = 2000; // try to auto-close window after 2 seconds.

props.socket.on('server_shutting_down', () => {
  isOpen.value = true;
  dialog.value?.showModal();
  shutdownTimer.value = setTimeout(() => {
    window.close();
  }, CLOSE_DELAY);
});

function cancelClose() {
  clearTimeout(shutdownTimer.value);
  isOpen.value = false;
  dialog.value?.close();
}

</script>

<template>
  <dialog ref="dialog">
    <div class="modal" id="serverShutdownModal" tabindex="-1" aria-labelledby="serverShutdownLabel"
      :aria-hidden="!isOpen">
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="serverShutdownLabel">Server Disconnected</h5>
            <button type="button" class="btn-close" @click="cancelClose" aria-label="dismiss dialog"></button>
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
  </dialog>
</template>

<style scoped>
div.modal {
  display: block;
}
</style>