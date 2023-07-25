<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { Modal } from 'bootstrap/dist/js/bootstrap.esm';
import type { AsyncSocket } from '../asyncSocket';

const props = defineProps<{socket:AsyncSocket}>();

const loading_dialog = ref<HTMLDivElement>();
const connected = ref(false);

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
            <div class="spinner-border text-primary" role="status">
              <span class="visually-hidden">Connecting...</span>
            </div>
          </h5>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
</style>