<script setup lang="ts">
import { onMounted, ref } from "vue";
import type { AsyncSocket } from "../asyncSocket.ts";

const title = "Log";

const props = defineProps<{
  socket: AsyncSocket;
}>();

const log_info = ref<{ message: string; title?: string }[]>([]);

props.socket.on("log", ({ message: { message, title } }) => {
  log_info.value.push({ message, title });
});
props.socket.asyncEmit("get_topic_messages", "log", (messages) => {
  log_info.value = [...log_info.value, ...messages.map((m) => m.message)];
});
</script>

<template>
  <div class="log">
    <div v-for="({ message, title }, index) of log_info" :key="index" class="message">
      <details v-if="title != null">
        <summary>{{ title }}</summary>
        <pre>{{ message }}</pre>
      </details>
      <span v-else>{{ message }}</span>
    </div>
  </div>
</template>

<style scoped>
div.log {
  white-space: nowrap;
}

.message {
  line-height: 1em;
  font-family: "Courier New", Courier, monospace;
}
</style>
