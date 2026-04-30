<!-- Button to toggle theme -->
<template>
  <button :aria-label="`Switch to ${nextTheme} theme`" class="theme-toggle" @click="toggleTheme">
    <span v-if="currentTheme === 'light'">🌞</span>
    <span v-else>🌜</span>
  </button>
</template>

<script lang="ts" setup>
import { computed, ref } from "vue";

/** Themes come from html data-bs-theme */
const currentTheme = ref(document.documentElement.getAttribute("data-bs-theme") || "light");
const nextTheme = computed(() => (currentTheme.value === "light" ? "dark" : "light"));

function toggleTheme() {
  const newTheme = nextTheme.value;
  document.documentElement.setAttribute("data-bs-theme", newTheme);
  currentTheme.value = newTheme;
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
