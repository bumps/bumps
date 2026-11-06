<!-- Button to toggle theme -->
<template>
  <button type="button" :aria-label="`Switch to ${nextTheme} theme`" class="theme-toggle" @click="toggleTheme">
    <span v-if="theme === 'light'">🌞</span>
    <span v-else>🌜</span>
  </button>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import { theme } from "../app_state";

const nextTheme = computed(() => (theme.value === "light" ? "dark" : "light"));

function toggleTheme() {
  theme.value = nextTheme.value;
}
</script>

<style scoped>
.theme-toggle {
  border: none;
  background: none;
  font-size: 1.5rem;
  cursor: pointer;
}
</style>
