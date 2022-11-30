<script setup lang="ts">
import { ref, onMounted } from 'vue';
import type { Socket } from 'socket.io-client';

const title = "Log";

const props = defineProps<{
  socket: Socket,
  visible: Boolean
}>();

const log_info = ref<string[]>([]);

onMounted(() => {
  props.socket.on('log', ({message}) => {
    log_info.value.push(message);
  });
});

</script>
        
<template>
  <div class="log">
    <pre class="message" v-for="(message, index) of log_info" :key="index">{{message}}</pre>
  </div>
</template>
    
<style scoped>

div.log {
  white-space: nowrap;
}

.message {
  font-family: 'Courier New', Courier, monospace;
  line-height: 1em;
}
</style>