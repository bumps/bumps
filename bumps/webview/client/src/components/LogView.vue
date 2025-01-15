<script setup lang="ts">
import { ref } from "vue";
import type { AsyncSocket } from "../asyncSocket.ts";

// const title = "Log";

const props = defineProps<{
  socket: AsyncSocket;
}>();

type LogMessage = { message: string; title?: string };
type LogEntry = { message: LogMessage; timestamp: string };

const log_info = ref<LogEntry[]>([]);

props.socket.on("log", (message: LogEntry) => {
  log_info.value.push(message);
});

props.socket.asyncEmit("get_topic_messages", "log", (messages: LogEntry[]) => {
  log_info.value = [...log_info.value, ...messages];
});

function timestampToDate(timestamp: string) {
  // cast to number
  let dt = Number(timestamp) * 1000;
  return new Date(dt).toLocaleString();
}
</script>

<template>
  <div class="container">
    <button
      class="btn btn-primary btn-sm me-2"
      @click="
        () =>
          props.socket.asyncEmit('get_topic_messages', 'log', (messages: LogEntry[]) => {
            log_info = messages;
          })
      "
    >
      Refresh
    </button>
    <button class="btn btn-danger btn-sm me-2" @click="() => (log_info = [])">Clear</button>

    <table class="table">
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Message</th>
        </tr>
      </thead>

      <tbody>
        <tr v-for="({ message: entry, timestamp }, index) of log_info" :key="index">
          <td>{{ timestampToDate(timestamp) }}</td>
          <td class="log_entry">
            <details v-if="entry.title !== null">
              <summary>{{ entry.title }}</summary>
              <pre>{{ entry.message }}</pre>
            </details>
            <p v-else>{{ entry.message }}</p>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.container {
  align-self: center;
  width: 90%;
  padding: 1rem;
}

.log_entry {
  /* line-height: 1rem; */
  font-family: "Courier New", Courier, monospace;

  p {
    margin: 0;
  }
}

table {
  margin: 0;
  border-collapse: collapse;
  font-size: 0.9rem;
  table-layout: auto;
  /* table-layout: fixed; */
}

th:nth-child(1) {
  width: 15%;
}

th,
td {
  padding: 0.5em;
  border-bottom: solid 2px #838383;
  /* white-space: nowrap; */
}

th {
  padding-bottom: 3px;
  font-weight: 600;
  text-transform: capitalize;
}

td {
  border: 1px solid #838383;
}
</style>
