<script setup lang="ts">
import { ref, onMounted, computed, shallowRef } from 'vue';
import { Modal } from 'bootstrap/dist/js/bootstrap.esm';
import type { Socket } from 'socket.io-client';

const props = defineProps<{socket: Socket}>();

const dialog = ref<HTMLDivElement>();
const isOpen = ref(false);
const closeCancelled = ref(false);
const timeRemaining = ref(5); // seconds
const intervalHandle = ref(0);

let modal: Modal;

onMounted(() => {
  modal = new Modal(dialog.value, { backdrop: 'static', keyboard: false });

});

props.socket.on('server_shutting_down', () => {
  timeRemaining.value = 5;
  modal?.show();
  intervalHandle.value = setInterval(() => {
    timeRemaining.value--;
    if (timeRemaining.value < 0) {
      window.close();
    }
  }, 1000);
});

function closeWindow() {
  window.close();
}

function cancelClose() {
  clearInterval(intervalHandle.value);
  modal?.hide();
}

</script>

<template>
  <div ref="dialog" class="modal fade" id="serverShutdownModal" tabindex="-1" aria-labelledby="serverShutdownLabel"
    :aria-hidden="isOpen">
    <div class="modal-dialog modal-dialog-centered">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="serverShutdownLabel">Server Shutting Down</h5>
          <button type="button" class="btn-close" @click="closeWindow" aria-label="Close Window"></button>
        </div>
        <div class="modal-body">
          <h3>This window will automatically close in {{ timeRemaining }} seconds.</h3>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-success" @click="closeWindow">Close Now</button>
          <button type="button" class="btn btn-primary" @click="cancelClose">Cancel</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
</style>