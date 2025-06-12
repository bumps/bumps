<script setup lang="ts">
import { ref, watch } from "vue";
import { disconnected } from "../app_state.ts";

const dialog = ref<HTMLDialogElement>();
const closeCancelled = ref(false);
const attemptingAutoClose = ref(false);
const shutdownTimer = ref<ReturnType<typeof setInterval>>();
const CLOSE_DELAY = 3; // try to auto-close window after n seconds.
const timeRemaining = ref(CLOSE_DELAY);

watch(disconnected, () => {
  if (disconnected.value) {
    dialog.value?.showModal();
    closeCancelled.value = false;
    attemptingAutoClose.value = true;
    timeRemaining.value = CLOSE_DELAY;
    shutdownTimer.value = setInterval(() => {
      timeRemaining.value -= 1;
      if (timeRemaining.value <= 0) {
        timeRemaining.value = 0;
        window.close();
        attemptingAutoClose.value = false;
        clearInterval(shutdownTimer.value);
      }
    }, 1000);
  } else {
    dialog.value?.close();
    clearInterval(shutdownTimer.value);
    attemptingAutoClose.value = false;
  }
});

function cancelClose() {
  clearInterval(shutdownTimer.value);
  attemptingAutoClose.value = false;
}
</script>

<template>
  <dialog ref="dialog">
    <div
      id="serverShutdownModal"
      class="modal"
      tabindex="-1"
      aria-labelledby="serverShutdownLabel"
      :aria-hidden="!disconnected"
    >
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-header">
            <h5 id="serverShutdownLabel" class="modal-title">Server disconnected</h5>
          </div>
          <div v-if="attemptingAutoClose" class="modal-body">
            <h3>Will attempt to auto-close client window...</h3>
            <div class="progress">
              <div
                class="progress-bar"
                role="progressbar"
                :style="{ width: (100 * timeRemaining) / CLOSE_DELAY + '%' }"
                aria-valuemin="0"
                :aria-valuenow="timeRemaining"
                :aria-valuemax="CLOSE_DELAY"
              >
                {{ timeRemaining }}
              </div>
            </div>
          </div>
          <div v-if="attemptingAutoClose" class="modal-footer">
            <button type="button" class="btn btn-primary" @click="cancelClose">Cancel</button>
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

dialog::backdrop {
  background-color: rgba(0, 0, 0, 0.5);
}
</style>
